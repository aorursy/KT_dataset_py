import numpy as np # linear algebra
import pandas as pd 

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 

import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder

def encode_data(df):
    '''
    The function does not return, but transforms the input pd.DataFrame
    
    Encodes the Costa Rican Household Poverty Level data 
    following studies in https://www.kaggle.com/mlisovyi/categorical-variables-in-the-data
    and the insight from https://www.kaggle.com/c/costa-rican-household-poverty-prediction/discussion/61403#359631
    
    The following columns get transformed: edjefe, edjefa, dependency, idhogar
    The user most likely will simply drop idhogar completely (after calculating houshold-level aggregates)
    '''
    
    yes_no_map = {'no': 0, 'yes': 1}
    
    df['dependency'] = df['dependency'].replace(yes_no_map).astype(np.float32)
    
    df['edjefe'] = df['edjefe'].replace(yes_no_map).astype(np.float32)
    df['edjefa'] = df['edjefa'].replace(yes_no_map).astype(np.float32)
    
    df['idhogar'] = LabelEncoder().fit_transform(df['idhogar'])

    
def do_features(df):
    feats_div = [('children_fraction', 'r4t1', 'r4t3'), 
                 ('working_man_fraction', 'r4h2', 'r4t3'),
                 ('all_man_fraction', 'r4h3', 'r4t3'),
                 ('human_density', 'tamviv', 'rooms'),
                 ('human_bed_density', 'tamviv', 'bedrooms'),
                 ('rent_per_person', 'v2a1', 'r4t3'),
                 ('rent_per_room', 'v2a1', 'rooms'),
                 ('mobile_density', 'qmobilephone', 'r4t3'),
                 ('tablet_density', 'v18q1', 'r4t3'),
                 ('mobile_adult_density', 'qmobilephone', 'r4t2'),
                 ('tablet_adult_density', 'v18q1', 'r4t2'),
                 #('', '', ''),
                ]
    
    feats_sub = [('people_not_living', 'tamhog', 'tamviv'),
                 ('people_weird_stat', 'tamhog', 'r4t3')]

    for f_new, f1, f2 in feats_div:
        df['fe_' + f_new] = (df[f1] / df[f2]).astype(np.float32)       
    for f_new, f1, f2 in feats_sub:
        df['fe_' + f_new] = (df[f1] - df[f2]).astype(np.float32)
    
    # aggregation rules over household
    aggs_num = {'age': ['min', 'max', 'mean', 'count'],
                'escolari': ['min', 'max', 'mean']
               }
    aggs_cat = {'dis': ['mean', 'sum']}
    for s_ in ['estadocivil', 'parentesco', 'instlevel']:
        for f_ in [f_ for f_ in df.columns if f_.startswith(s_)]:
            aggs_cat[f_] = ['mean']
    # aggregation over household
    for name_, df_ in [('18', df.query('age >= 18'))]:
        df_agg = df_.groupby('idhogar').agg({**aggs_num, **aggs_cat}).astype(np.float32)
        df_agg.columns = pd.Index(['agg' + name_ + '_' + e[0] + "_" + e[1].upper() for e in df_agg.columns.tolist()])
        df = df.join(df_agg, how='left', on='idhogar')
        del df_agg
    # do something advanced above...
    
    # Drop SQB variables, as they are just squres of other vars 
    df.drop([f_ for f_ in df.columns if f_.startswith('SQB') or f_ == 'agesq'], axis=1, inplace=True)
    # Drop id's
    df.drop(['Id'], axis=1, inplace=True)
    # Drop repeated columns
    df.drop(['hhsize', 'female', 'area2'], axis=1, inplace=True)
    return df
    
def convert_OHE2LE(df):
    tmp_df = df.copy(deep=True)
    for s_ in ['pared', 'piso', 'techo', 'abastagua', 'sanitario', 'energcocinar', 'elimbasu', 
               'epared', 'etecho', 'eviv', 'estadocivil', 'parentesco', 
               'instlevel', 'lugar', 'tipovivi',
               'manual_elec']:
        if 'manual_' not in s_:
            cols_s_ = [f_ for f_ in df.columns if f_.startswith(s_)]
        elif 'elec' in s_:
            cols_s_ = ['public', 'planpri', 'noelec', 'coopele']
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
        tmp_cat = tmp_df[cols_s_].idxmax(axis=1)
        tmp_df[s_ + '_LE'] = LabelEncoder().fit_transform(tmp_cat).astype(np.int16)
        if 'parentesco1' in cols_s_:
            cols_s_.remove('parentesco1')
        tmp_df.drop(cols_s_, axis=1, inplace=True)
    return tmp_df
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
def process_df(df_):
    # fix categorical features
    encode_data(df_)
    #fill in missing values based on https://www.kaggle.com/mlisovyi/missing-values-in-the-data
    for f_ in ['v2a1', 'v18q1', 'meaneduc', 'SQBmeaned']:
        df_[f_] = df_[f_].fillna(0)
    df_['rez_esc'] = df_['rez_esc'].fillna(-1)
    # do feature engineering and drop useless columns
    return do_features(df_)

train = process_df(train)
test = process_df(test)
def train_test_apply_func(train_, test_, func_):
    test_['Target'] = 0
    xx = pd.concat([train_, test_])

    xx_func = func_(xx)
    train_ = xx_func.iloc[:train_.shape[0], :]
    test_  = xx_func.iloc[train_.shape[0]:, :].drop('Target', axis=1)

    del xx, xx_func
    return train_, test_
train, test = train_test_apply_func(train, test, convert_OHE2LE)
train.info(max_cols=20)
cols_2_ohe = ['eviv_LE', 'etecho_LE', 'epared_LE', 'elimbasu_LE', 
              'energcocinar_LE', 'sanitario_LE', 'manual_elec_LE',
              'pared_LE']
cols_nums = ['age', 'meaneduc', 'dependency', 
             'hogar_nin', 'hogar_adul', 'hogar_mayor', 'hogar_total',
             'bedrooms', 'overcrowding']

def convert_geo2aggs(df_):
    tmp_df = pd.concat([df_[(['lugar_LE', 'idhogar']+cols_nums)],
                        pd.get_dummies(df_[cols_2_ohe], 
                                       columns=cols_2_ohe)],axis=1)
    #print(pd.get_dummies(train[cols_2_ohe], 
    #                                   columns=cols_2_ohe).head())
    #print(tmp_df.head())
    #print(tmp_df.groupby(['lugar_LE','idhogar']).mean().head())
    geo_agg = tmp_df.groupby(['lugar_LE','idhogar']).mean().groupby('lugar_LE').mean().astype(np.float32)
    geo_agg.columns = pd.Index(['geo_' + e + '_MEAN' for e in geo_agg.columns.tolist()])
    
    #print(gb.T)
    del tmp_df
    return df_.join(geo_agg, how='left', on='lugar_LE')

train, test = train_test_apply_func(train, test, convert_geo2aggs)
X = train.query('parentesco1==1')

# pull out the target variable
y = X['Target'] - 1 # this is done to bing input labels [1,2,3,4] in agreement with lightgbm [0,1,2,3]
X = X.drop(['Target'], axis=1)
#cols_2_drop = ['agg18_estadocivil1_MEAN', 'agg18_estadocivil3_COUNT', 'agg18_estadocivil4_COUNT', 'agg18_estadocivil5_COUNT', 'agg18_estadocivil6_COUNT', 'agg18_estadocivil7_COUNT', 'agg18_instlevel1_COUNT', 'agg18_instlevel2_COUNT', 'agg18_instlevel3_COUNT', 'agg18_instlevel4_COUNT', 'agg18_instlevel5_COUNT', 'agg18_instlevel6_COUNT', 'agg18_instlevel7_COUNT', 'agg18_instlevel8_COUNT', 'agg18_instlevel9_COUNT', 'agg18_parentesco10_COUNT', 'agg18_parentesco10_MEAN', 'agg18_parentesco11_COUNT', 'agg18_parentesco11_MEAN', 'agg18_parentesco12_COUNT', 'agg18_parentesco12_MEAN', 'agg18_parentesco1_COUNT', 'agg18_parentesco2_COUNT', 'agg18_parentesco3_COUNT', 'agg18_parentesco4_COUNT', 'agg18_parentesco4_MEAN', 'agg18_parentesco5_COUNT', 'agg18_parentesco6_COUNT', 'agg18_parentesco6_MEAN', 'agg18_parentesco7_COUNT', 'agg18_parentesco7_MEAN', 'agg18_parentesco8_COUNT', 'agg18_parentesco8_MEAN', 'agg18_parentesco9_COUNT', 'fe_people_weird_stat', 'hacapo', 'hacdor', 'mobilephone', 'parentesco1', 'parentesco_LE', 'rez_esc', 'v14a', 'v18q']
cols_2_drop = ['abastagua_LE', 'agg18_estadocivil1_MEAN', 'agg18_instlevel6_MEAN', 'agg18_parentesco10_MEAN', 'agg18_parentesco11_MEAN', 'agg18_parentesco12_MEAN', 'agg18_parentesco4_MEAN', 'agg18_parentesco5_MEAN', 'agg18_parentesco6_MEAN', 'agg18_parentesco7_MEAN', 'agg18_parentesco8_MEAN', 'agg18_parentesco9_MEAN', 'fe_people_not_living', 'fe_people_weird_stat', 'geo_elimbasu_LE_3_MEAN', 'geo_elimbasu_LE_4_MEAN', 'geo_energcocinar_LE_0_MEAN', 'geo_energcocinar_LE_1_MEAN', 'geo_energcocinar_LE_2_MEAN', 'geo_epared_LE_0_MEAN', 'geo_epared_LE_2_MEAN', 'geo_etecho_LE_2_MEAN', 'geo_eviv_LE_0_MEAN', 'geo_hogar_mayor_MEAN', 'geo_hogar_nin_MEAN', 'geo_manual_elec_LE_1_MEAN', 'geo_manual_elec_LE_2_MEAN', 'geo_manual_elec_LE_3_MEAN', 'geo_pared_LE_0_MEAN', 'geo_pared_LE_1_MEAN', 'geo_pared_LE_3_MEAN', 'geo_pared_LE_4_MEAN', 'geo_pared_LE_5_MEAN', 'geo_pared_LE_6_MEAN', 'geo_pared_LE_7_MEAN', 'hacapo', 'hacdor', 'mobilephone', 'parentesco1', 'parentesco_LE', 'rez_esc', 'techo_LE', 'v14a', 'v18q']

X.drop((cols_2_drop+['idhogar']), axis=1, inplace=True)
test.drop((cols_2_drop+['idhogar']), axis=1, inplace=True)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=314, stratify=y)
X_test.info(max_cols=20)
from sklearn.metrics import f1_score
def evaluate_macroF1_lgb(truth, predictions):  
    # this follows the discussion in https://github.com/Microsoft/LightGBM/issues/1483
    pred_labels = predictions.reshape(len(np.unique(truth)),-1).argmax(axis=0)
    f1 = f1_score(truth, pred_labels, average='macro')
    return ('macroF1', f1, True) 

import lightgbm as lgb
fit_params={"early_stopping_rounds":300, 
            "eval_metric" : evaluate_macroF1_lgb, 
            "eval_set" : [(X_test,y_test)],
            'eval_names': ['valid'],
            #'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],
            'verbose': False,
            'categorical_feature': 'auto'}

def learning_rate_power_0997(current_iter):
    base_learning_rate = 0.1
    min_learning_rate = 0.02
    lr = base_learning_rate  * np.power(.995, current_iter)
    return max(lr, min_learning_rate)

fit_params['callbacks'] = [lgb.reset_parameter(learning_rate=learning_rate_power_0997)]

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
param_test ={'num_leaves': sp_randint(12, 20), 
             'min_child_samples': sp_randint(40, 100), 
             #'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': sp_uniform(loc=0.75, scale=0.25), 
             'colsample_bytree': sp_uniform(loc=0.8, scale=0.15)#,
             #'reg_alpha': [0, 1e-3, 1e-1, 1, 10, 50, 100],
             #'reg_lambda': [0, 1e-3, 1e-1, 1, 10, 50, 100]
            }
#This parameter defines the number of HP points to be tested
n_HP_points_to_test = 100

import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
clf = lgb.LGBMClassifier(max_depth=-1, learning_rate=0.05, objective='multiclass',
                         random_state=314, silent=True, metric='None', 
                         n_jobs=4, n_estimators=5000, class_weight='balanced')

gs = RandomizedSearchCV(
    estimator=clf, param_distributions=param_test, 
    n_iter=n_HP_points_to_test,
    scoring='f1_macro',
    cv=5,
    refit=True,
    random_state=314,
    verbose=True)
_ = gs.fit(X_train, y_train, **fit_params)
print("PERFORMANCE IMPROVES FROM TOP TO BOTTOM")
print("Valid+-Std     Train  :   Parameters")
for i in np.argsort(gs.cv_results_['mean_test_score'])[-5:]:
    print('{1:.3f}+-{3:.3f}     {2:.3f}   :  {0}'.format(gs.cv_results_['params'][i], 
                                    gs.cv_results_['mean_test_score'][i], 
                                    gs.cv_results_['mean_train_score'][i],
                                    gs.cv_results_['std_test_score'][i]))

opt_parameters = gs.best_params_
clf_final = lgb.LGBMClassifier(**clf.get_params())
clf_final.set_params(**opt_parameters)

def learning_rate_power_0997(current_iter):
    base_learning_rate = 0.1
    min_learning_rate = 0.02
    lr = base_learning_rate  * np.power(.997, current_iter)
    return max(lr, min_learning_rate)

#Train the final model with learning rate decay
fit_params['verbose'] = 200
_ = clf_final.fit(X_train, y_train, **fit_params)#, callbacks=[lgb.reset_parameter(learning_rate=learning_rate_power_0997)])
y_subm = pd.read_csv('../input/sample_submission.csv')
y_subm['Target'] = clf_final.predict(test) + 1
from datetime import datetime
now = datetime.now()
global_score = f1_score(y_test, clf_final.predict(X_test), average='macro')

sub_file = 'submission_LGB_{:.4f}_{}.csv'.format(global_score, str(now.strftime('%Y-%m-%d-%H-%M')))

y_subm.to_csv(sub_file, index=False)
