max_events = None
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # needed for 3D scatter plots
%matplotlib inline 
import seaborn as sns
import gc
import warnings
warnings.filterwarnings("ignore")

PATH='../input/'

import os
print(os.listdir(PATH))
train = pd.read_csv('{}/train.csv'.format(PATH), nrows=max_events)
test  = pd.read_csv('{}/test.csv'.format(PATH), nrows=max_events)

y = train['Cover_Type']
train.drop('Cover_Type', axis=1, inplace=True)

train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)
print('Train shape: {}'.format(train.shape))
print('Test  shape: {}'.format(test.shape))
train.info(verbose=False)
def convert_OHE2LE(df):
    tmp_df = df.copy(deep=True)
    for s_ in ['Soil_Type', 'Wilderness_Area']:
        cols_s_ = [f_ for f_ in df.columns if f_.startswith(s_)]
        sum_ohe = tmp_df[cols_s_].sum(axis=1).unique()
        #deal with those OHE, where there is a sum over columns == 0
        if 0 in sum_ohe:
            print('The OHE in {} is incomplete. A new column will be added before label encoding'
                  .format(s_))
            # dummy colmn name to be added
            col_dummy = s_+'_dummy'
            # add the column to the dataframe
            tmp_df[col_dummy] = (tmp_df[cols_s_].sum(axis=1) == 0).astype(np.int8)
            # add the name to the list of columns to be label-encoded
            cols_s_.append(col_dummy)
            # proof-check, that now the category is complete
            sum_ohe = tmp_df[cols_s_].sum(axis=1).unique()
            if 0 in sum_ohe:
                 print("The category completion did not work")
        tmp_df[s_ + '_LE'] = tmp_df[cols_s_].idxmax(axis=1).str.replace(s_,'').astype(np.uint16)
        tmp_df.drop(cols_s_, axis=1, inplace=True)
    return tmp_df



def train_test_apply_func(train_, test_, func_):
    xx = pd.concat([train_, test_])
    xx_func = func_(xx)
    train_ = xx_func.iloc[:train_.shape[0], :]
    test_  = xx_func.iloc[train_.shape[0]:, :]

    del xx, xx_func
    return train_, test_
train_x, test_x = train_test_apply_func(train, test, convert_OHE2LE)
train_x.head()
def preprocess(df_):
    df_['fe_E_Min_02HDtH'] = (df_['Elevation']- df_['Horizontal_Distance_To_Hydrology']*0.2).astype(np.float32)
    df_['fe_Distance_To_Hydrology'] = np.sqrt(df_['Horizontal_Distance_To_Hydrology']**2 + 
                                              df_['Vertical_Distance_To_Hydrology']**2).astype(np.float32)
    
    feats_sub = [('Elevation_Min_VDtH', 'Elevation', 'Vertical_Distance_To_Hydrology'),
                 ('HD_Hydrology_Min_Roadways', 'Horizontal_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways'),
                 ('HD_Hydrology_Min_Fire', 'Horizontal_Distance_To_Hydrology', 'Horizontal_Distance_To_Fire_Points')]
    feats_add = [('Elevation_Add_VDtH', 'Elevation', 'Vertical_Distance_To_Hydrology')]
    
    for f_new, f1, f2 in feats_sub:
        df_['fe_' + f_new] = (df_[f1] - df_[f2]).astype(np.float32)
    for f_new, f1, f2 in feats_add:
        df_['fe_' + f_new] = (df_[f1] + df_[f2]).astype(np.float32)
        
    # The feature is advertised in https://douglas-fraser.com/forest_cover_management.pdf
    df_['fe_Shade9_Mul_VDtH'] = (df_['Hillshade_9am'] * df_['Vertical_Distance_To_Hydrology']).astype(np.float32)
    
    # this mapping comes from https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.info
    climatic_zone = {}
    geologic_zone = {}
    for i in range(1,41):
        if i <= 6:
            climatic_zone[i] = 2
            geologic_zone[i] = 7
        elif i <= 8:
            climatic_zone[i] = 3
            geologic_zone[i] = 5
        elif i == 9:
            climatic_zone[i] = 4
            geologic_zone[i] = 2
        elif i <= 13:
            climatic_zone[i] = 4
            geologic_zone[i] = 7
        elif i <= 15:
            climatic_zone[i] = 5
            geologic_zone[i] = 1
        elif i <= 17:
            climatic_zone[i] = 6
            geologic_zone[i] = 1
        elif i == 18:
            climatic_zone[i] = 6
            geologic_zone[i] = 7
        elif i <= 21:
            climatic_zone[i] = 7
            geologic_zone[i] = 1
        elif i <= 23:
            climatic_zone[i] = 7
            geologic_zone[i] = 2
        elif i <= 34:
            climatic_zone[i] = 7
            geologic_zone[i] = 7
        else:
            climatic_zone[i] = 8
            geologic_zone[i] = 7
            
    df_['Climatic_zone_LE'] = df_['Soil_Type_LE'].map(climatic_zone).astype(np.uint8)
    df_['Geologic_zone_LE'] = df_['Soil_Type_LE'].map(geologic_zone).astype(np.uint8)
    return df_
train_x = preprocess(train_x)
test_x = preprocess(test_x)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from  sklearn.linear_model import LogisticRegression
import lightgbm as lgb
y = y-1
X_train, X_test, y_train, y_test = train_test_split(train_x, y, test_size=0.15, random_state=315, stratify=y)
def learning_rate_decay_power_0995(current_iter):
    base_learning_rate = 0.15
    lr = base_learning_rate  * np.power(.995, current_iter)
    return lr if lr > 1e-2 else 1e-2

clfs = {'rf': (RandomForestClassifier(n_estimators=200, max_depth=1, random_state=314, n_jobs=4),
               {'max_depth': [20,25,30,35,40,45,50]}, 
               {}),
        'xt': (ExtraTreesClassifier(n_estimators=200, max_depth=1, max_features='auto',random_state=314, n_jobs=4),
               {'max_depth': [20,25,30,35,40,45,50]},
               {}),
        'lgbm': (lgb.LGBMClassifier(max_depth=-1, min_child_samples=400, 
                                 random_state=314, silent=True, metric='None', 
                                 n_jobs=4, n_estimators=5000, learning_rate=0.1), 
                 {'colsample_bytree': [0.75], 'min_child_weight': [0.1,1,10], 'num_leaves': [18, 20,22], 'subsample': [0.75]}, 
                 {'eval_set': [(X_test, y_test)], 
                  'eval_metric': 'multi_error', 'verbose':500, 'early_stopping_rounds':100, 
                  'callbacks':[lgb.reset_parameter(learning_rate=learning_rate_decay_power_0995)]}
                )
       }
gss = {}
for name, (clf, clf_pars, fit_pars) in clfs.items():
    print('--------------- {} -----------'.format(name))
    gs = GridSearchCV(clf, param_grid=clf_pars,
                            scoring='accuracy',
                            cv=5,
                            n_jobs=1,
                            refit=True,
                            verbose=True)
    gs = gs.fit(X_train, y_train, **fit_pars)
    print('{}:  train = {:.4f}, test = {:.4f}+-{:.4f} with best params {}'.format(name, 
                                                                                  gs.cv_results_['mean_train_score'][gs.best_index_],
                                                                                  gs.cv_results_['mean_test_score'][gs.best_index_],
                                                                                  gs.cv_results_['std_test_score'][gs.best_index_],
                                                                                  gs.best_params_
                                                                                 ))
    print("Valid+-Std     Train  :   Parameters")
    for i in np.argsort(gs.cv_results_['mean_test_score'])[-5:]:
        print('{1:.3f}+-{3:.3f}     {2:.3f}   :  {0}'.format(gs.cv_results_['params'][i], 
                                        gs.cv_results_['mean_test_score'][i], 
                                        gs.cv_results_['mean_train_score'][i],
                                        gs.cv_results_['std_test_score'][i]))
    gss[name] = gs
# gss = {}
# for name, (clf, clf_pars, fit_pars) in clfs.items():
#     if name == 'lgbm':
#         continue
#     print('--------------- {} -----------'.format(name))
#     gs = GridSearchCV(clf, param_grid=clf_pars,
#                             scoring='accuracy',
#                             cv=5,
#                             n_jobs=1,
#                             refit=True,
#                             verbose=True)
#     gs = gs.fit(X_train, y_train, **fit_pars)
#     print('{}:  train = {:.4f}, test = {:.4f}+-{:.4f} with best params {}'.format(name, 
#                                                                                   gs.cv_results_['mean_train_score'][gs.best_index_],
#                                                                                   gs.cv_results_['mean_test_score'][gs.best_index_],
#                                                                                   gs.cv_results_['std_test_score'][gs.best_index_],
#                                                                                   gs.best_params_
#                                                                                  ))
#     print("Valid+-Std     Train  :   Parameters")
#     for i in np.argsort(gs.cv_results_['mean_test_score'])[-5:]:
#         print('{1:.3f}+-{3:.3f}     {2:.3f}   :  {0}'.format(gs.cv_results_['params'][i], 
#                                         gs.cv_results_['mean_test_score'][i], 
#                                         gs.cv_results_['mean_train_score'][i],
#                                         gs.cv_results_['std_test_score'][i]))
#     gss[name] = gs
