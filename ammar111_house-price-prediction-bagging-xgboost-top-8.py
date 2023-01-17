import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
from sklearn.ensemble import BaggingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import os
from pprint import pprint


%matplotlib inline
%config InlineBackend.figure_format = 'retina'
# warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)

sns.set(style="ticks")
plt.rc('figure', figsize=(8, 5), dpi=100)
plt.rc('axes', facecolor="#ffffff", linewidth=0.4, grid=True, labelpad=8, labelcolor='#616161')
plt.rc('patch', linewidth=0)
plt.rc('xtick.major', width=0.2)
plt.rc('ytick.major', width=0.2)
plt.rc('grid', color='#9E9E9E', linewidth=0.4)
plt.rc('text', color='#282828')
plt.rc('savefig', pad_inches=0.3, dpi=300)
tr = pd.read_csv('../input/train.csv')
ts = pd.read_csv('../input/test.csv')
tr.head(10)
tr.shape, ts.shape
train_null = tr.isna().sum()
print('Columns that contain missing values in training dataset')
train_null[train_null > 0]

test_null = ts.isna().sum()
print('Columns that contain missing values in test dataset')
test_null[test_null > 0]
tt = [tr, ts]

# Filling in missing values for columns that contain missing values in both
# training and test datasets
for df in tt:
    df['LotFrontage'].fillna(df['LotFrontage'].mean(), inplace=True)
    df['Alley'].fillna('No alley', inplace=True)
    df['MasVnrArea'].fillna(df['MasVnrArea'].median(), inplace=True)
    df['FireplaceQu'].fillna('No fireplace', inplace=True)
    df['PoolQC'].fillna('No pool', inplace=True)
    df['Fence'].fillna('No fence', inplace=True)
    df['MiscFeature'].fillna('No', inplace=True)
    df['GarageYrBlt'].fillna(1, inplace=True)
    for a in ['MasVnrType', 'BsmtExposure']:
        df[a].fillna(df[a].mode().iloc[0], inplace=True)
    for a in ['BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2']:
        df[a].fillna('No basement', inplace=True)
    for a in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        df[a].fillna('No Garage', inplace=True)
        
# Some columns contain missing values in only one of the two datasets
tr['Electrical'].fillna(tr['Electrical'].mode().iloc[0], inplace=True)

for a in ['Utilities', 'MSZoning', 'Exterior1st', 'Exterior2nd', 'BsmtFullBath', 'BsmtHalfBath', 'KitchenQual',
         'Functional', 'GarageCars', 'SaleType']:
    ts[a].fillna(ts[a].mode().iloc[0], inplace=True)

for a in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageArea']:
    ts[a].fillna(ts[a].median(), inplace=True)

# GarageYrBlt_years = tr['GarageYrBlt'].value_counts().index.tolist()
# GarageYrBlt_prob = tr['GarageYrBlt'].value_counts(normalize=True).values.tolist()
# for df in tt:
#     df['GarageYrBlt'] = df['GarageYrBlt'].apply(lambda x: 
#                                                 np.random.choice(GarageYrBlt_years, p=GarageYrBlt_prob) 
#                                                 if (pd.isnull(x)) else x)
for df in tt:
    df['MSSubClass'] = df['MSSubClass'].astype(str)
    mp1 = {'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':0}
    df['ExterQual'] = df['ExterQual'].map(mp1)
    df['ExterCond'] = df['ExterCond'].map(mp1)
    mp3 = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No basement':0}
    df['BsmtQual'] = df['BsmtQual'].map(mp3)
    df['BsmtCond'] = df['BsmtCond'].map(mp3)
    df['BsmtExposure'] = df['BsmtExposure'].map({'Gd':4,'Av':3,'Mn':2,'No':1,'No basement':0})
    mp2 = {'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'No basement':0}
    df['BsmtFinType1'] = df['BsmtFinType1'].map(mp2)
    df['BsmtFinType2'] = df['BsmtFinType2'].map(mp2)
    df['HeatingQC'] = df['HeatingQC'].map(mp1)
    df['CentralAir'] = df['CentralAir'].map({'Y':1,'N':0})
    df['KitchenQual'] = df['KitchenQual'].map(mp1)
    df['Functional'] = df['Functional'].map({'Typ':7,'Min1':6,'Min2':5,'Mod':4,'Maj1':3,'Maj2':2,'Sev':1,'Sal':0})
    df['FireplaceQu'] = df['FireplaceQu'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No fireplace':0})
    df['GarageFinish'] = df['GarageFinish'].map({'Fin':3,'RFn':2,'Unf':1,'No Garage':0})
    df['GarageQual'] = df['GarageQual'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No Garage':0})
    df['GarageCond'] = df['GarageCond'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No Garage':0})
    df['PoolQC'] = df['PoolQC'].map({'Ex':4,'Gd':3,'TA':2,'Fa':1,'No pool':0})
    df['Fence'] = df['Fence'].map({'GdPrv':4,'MnPrv':3,'GdWo':2,'MnWw':1,'No fence':0})
    
tr['TotalSF'] = tr['TotalBsmtSF'] + tr['GrLivArea']
ts['TotalSF'] = ts['TotalBsmtSF'] + ts['GrLivArea']
tr.drop('Id', axis=1, inplace=True)
# saving test dataset Ids for submission later
ts_ids = ts['Id']
ts.drop('Id', axis=1, inplace=True)
fig, ax = plt.subplots()
ax.scatter(x = tr['GrLivArea'], y = tr['SalePrice']);
# outlier_indexes = tr[(tr['GrLivArea'] > 4000) & (tr['SalePrice'] < 200000)].index
# tr = tr.drop(outlier_indexes, axis=0)
## Outlier removal

# from pandas.api.types import is_numeric_dtype
# from scipy import stats
# for c in tr.columns.values:
#     if is_numeric_dtype(tr[c]):
#         z = np.abs(stats.zscore(tr[c]))
#         s = tr[(z > 7) & (tr[c] > 1000)].shape[0]
#         if s <= 5 and s > 0:
#             tr.drop(tr[(z > 7) & (tr[c] > 1000)].index, inplace=True)

# Or

# for df in tt:
#     for c in df.columns.values:
#         if is_numeric_dtype(df[c]):
#             Q1 = df[c].quantile(0.25)
#             Q3 = df[c].quantile(0.75)
#             IQR = Q3 - Q1
#             tmp = df[(df[c] < (Q1 - 10 * IQR)) | (df[c] > (Q3 + 10 * IQR))].shape
#             if tmp[0] > 0:
#                 print(c)
#                 print(tmp)
#                 print()
tr_X = tr.drop('SalePrice', axis=1)
tr_y = tr['SalePrice']
tr_X = pd.get_dummies(tr_X)
ts = pd.get_dummies(ts)
# get names of columns that exist in tr_X but not in ts after applying get_dummies()
ts_diff_cols = set(tr_X.columns.values) - set(ts.columns.values)
tr_diff_cols = set(ts.columns.values) - set(tr_X.columns.values)
# add them to ts with values of zero
for c in ts_diff_cols:
    ts[c] = 0
for c in tr_diff_cols:
    tr_X[c] = 0
# make sure that columns have the same order in tr_X and ts
tr_X, ts = tr_X.align(ts, axis=1)
tr_X_colnames = list(tr_X.columns.values)
scaler = StandardScaler().fit(tr_X)
tr_X = scaler.transform(tr_X)
ts = scaler.transform(ts)
x_train, x_test, y_train, y_test = \
    train_test_split(tr_X, tr_y, test_size=0.25, random_state=3)
def f(model, name):
    '''
    fits the model on x_train and y_train, then prints
    the model prediction score on x_train and x_test
    '''
    print('-------- ', name, ' --------')
    model.fit(x_train, y_train)
    pred = model.predict(x_train)
    score = np.sqrt(metrics.mean_squared_error(y_train, pred))
    print('Training: ', score)
    pred = model.predict(x_test)
    all_pred.append(list(pred))
    score = np.sqrt(metrics.mean_squared_error(y_test, pred))
    print('Test: ', score)
    print()

def get_predictions(model):
    '''
    fits the model on x_train and y_train, then
    returns the predictions of the model on x_train
    and x_test
    '''
    model.fit(x_train, y_train)
    return (list(model.predict(x_train)), list(model.predict(x_test)))
def parameterSearch(search_type='random', parameter_space={}, 
                    model=None, cv=4, n_iter=100, scoring='neg_mean_squared_error', 
                    n_jobs=4, verbose=1, fitting_data=None, 
                    best_parameters=[], best_scores=[], 
                    n_repeat=1, print_rest=False, round_decimals=6,
                    modify_score=None):
    
    '''
    Parameters:
        search_type should be 'random' or 'grid'
        parameter_space is a dictionary
    
    best_parameters = [{'p1': v1, 'p2': v2}, {'p1': v1, 'p2': v2},...]
    best_scores = [(test_mean, test_std, train_mean, train_std), 
                    (test_mean, test_std, train_mean, train_std),...]
    '''
    
    for i in range(n_repeat):
        if search_type == 'grid':
            clf = GridSearchCV(model, parameter_space, cv=cv, return_train_score=True,
                       scoring=scoring, n_jobs=n_jobs, verbose=verbose)
        else:
            clf = RandomizedSearchCV(model, parameter_space, cv=cv, n_iter=n_iter, 
                                     return_train_score=True, scoring=scoring, 
                                     n_jobs=n_jobs, verbose=verbose)
            
        clf.fit(*(fitting_data))
        print("Best parameters:" if n_repeat==1 else "Round {} | Best parameters:".format(i+1))
        print()
        pprint(clf.best_params_) 
        print()
        best_parameters.append(clf.best_params_)
        bi = clf.best_index_
        tsmeans = np.round(clf.cv_results_['mean_test_score'], decimals=round_decimals)
        tsstds = np.round(clf.cv_results_['std_test_score'], decimals=round_decimals)
        trmeans = np.round(clf.cv_results_['mean_train_score'], decimals=round_decimals)
        trstds = np.round(clf.cv_results_['std_train_score'], decimals=round_decimals)
        if modify_score != None:
            tsmeans = list(map(modify_score, tsmeans))
            tsstds = list(map(modify_score, tsstds))
            trmeans = list(map(modify_score, trmeans))
            trstds = list(map(modify_score, trstds))
        print("Mean test score = ", tsmeans[bi])
        print("with std = ", tsstds[bi])
        print("Mean train score = ", trmeans[bi])
        print("with std = ", trstds[bi])
        best_scores.append((tsmeans[bi], tsstds[bi], trmeans[bi], trstds[bi]))
        print()
        if print_rest:
            print('*'*30)
            print()
            print("All parameters: ")
            parameters = clf.cv_results_['params']
            for tsmean, tsstd, trmean, trstd, params in zip(tsmeans, tsstds, trmeans, trstds, parameters):
                print(params)
                print(tsmean, tsstd, trmean, trstd, sep=' | ')
                print('-'*20)
        print('='*50)
        print('='*50)
        print()
xgb_tuned_parameters = {
                     'gamma': np.round(np.random.uniform(0,100,size=(100,)), decimals=4),
                     'subsample': np.round(np.random.uniform(0.5,0.9,(100,)), decimals=4),
                     'colsample_bytree': np.round(np.random.uniform(0.5,0.9,(100,)), decimals=4),
                     'colsample_bylevel': np.round(np.random.uniform(0.5,0.9,(100,)), decimals=4),
                     'learning_rate': np.round(np.random.uniform(0.05, 0.2, (100,)), decimals=4),
                     'n_estimators': np.random.randint(10,1500,(100,)),
                     'reg_alpha': np.round(np.random.uniform(1,500,(100,)), decimals=4),
                     'reg_lambda': np.round(np.random.uniform(1,500,(100,)), decimals=4),
                     'random_state': np.random.randint(1,20,(100,)),
                    }

xgb_bestpars = []
xgb_bestscores = []

# Finding good parameters for XGBRegressor
# parameterSearch(model=XGBRegressor(), best_parameters=xgb_bestpars, 
#                 best_scores=xgb_bestscores, parameter_space=xgb_tuned_parameters, 
#                 fitting_data=(tr_X, tr_y), cv=4, n_iter=1000, n_jobs=4, n_repeat=1, 
#                 search_type='random', print_rest=True, round_decimals=4, verbose=1, 
#                 modify_score=lambda v: np.sqrt(np.absolute(v)))

xgb = XGBRegressor(**{'colsample_bylevel': 0.5308,
                     'colsample_bytree': 0.861,
                     'gamma': 54.0637,
                     'learning_rate': 0.0577,
                     'n_estimators': 1291,
                     'random_state': 11,
                     'reg_alpha': 384.5839,
                     'reg_lambda': 15.3661,
                     'subsample': 0.8521})
xgb.fit(tr_X, tr_y)
feat_imp = xgb.feature_importances_
feat_imp_s = pd.Series(feat_imp, index=tr_X_colnames).sort_values(ascending=False)
# # plot a bar chart of feature importances
# fig, ax = plt.subplots(figsize=(18,80))
# sns.barplot(x=feat_imp_s, y=feat_imp_s.index, palette=sns.color_palette('deep', n_colors=284));
# plt.xlabel('Feature Importance Score');
# plt.ylabel('Features');
fi = sorted( [i for i, x in enumerate(feat_imp) if x < 0.0001] )
tr_X = np.delete(tr_X, fi, 1)
ts = np.delete(ts, fi, 1)
# re-split the training data
x_train, x_test, y_train, y_test = \
    train_test_split(tr_X, tr_y, test_size=0.25, random_state=0)
xgb_bestpars = []
xgb_bestscores = []

# Finding good parameters for XGBRegressor
# parameterSearch(model=XGBRegressor(), best_parameters=xgb_bestpars, 
#                 best_scores=xgb_bestscores, parameter_space=xgb_tuned_parameters, 
#                 fitting_data=(tr_X, tr_y), cv=4, n_iter=5000, n_jobs=4, n_repeat=1, 
#                 search_type='random', print_rest=True, round_decimals=4, verbose=1, 
#                 modify_score=lambda v: np.sqrt(np.absolute(v)))
# Finding good parameters for Bagging with XGBRegressor
ops = [
    {'colsample_bylevel': 0.5037, 'colsample_bytree': 0.5102, 'gamma': 8.5888, 'learning_rate': 0.0573,
     'n_estimators': 1194, 'random_state': 17, 'reg_alpha': 12.3419, 'reg_lambda': 15.3661, 'subsample': 0.8243},
    {'subsample': 0.5656, 'reg_lambda': 302.4913, 'reg_alpha': 482.5555, 'random_state': 2, 'n_estimators': 641, 
     'learning_rate': 0.1327, 'gamma': 50.2968, 'colsample_bytree': 0.8274, 'colsample_bylevel': 0.7549},
    {'subsample': 0.8526, 'reg_lambda': 223.798, 'reg_alpha': 497.4114, 'random_state': 12, 'n_estimators': 245, 
     'learning_rate': 0.124, 'gamma': 58.3453, 'colsample_bytree': 0.8874, 'colsample_bylevel': 0.7565}]
bes = [XGBRegressor(**op) for op in ops]
bg_tuned_parameters = {
    'base_estimator': bes,
    'n_estimators': np.random.randint(7,30,(100,)),
    'max_samples': np.round(np.random.uniform(0.7,1.0,(100,)), decimals=4),
    'max_features': np.round(np.random.uniform(0.7,1.0,(100,)), decimals=4),
    'bootstrap': (True, False),
    'bootstrap_features': (True, False),
    'random_state': np.random.randint(1,50,(100,))}
bg_bestpars = []
bg_bestscores = []
# parameterSearch(model=BaggingRegressor(), best_parameters=bg_bestpars, 
#                 best_scores=bg_bestscores, parameter_space=bg_tuned_parameters, 
#                 fitting_data=(tr_X, tr_y), cv=4, n_iter=30, n_jobs=4, n_repeat=1, 
#                 search_type='random', print_rest=True, round_decimals=4, verbose=1, 
#                 modify_score=lambda v: np.sqrt(np.absolute(v)))
bg_s = [BaggingRegressor(**{'random_state': 36, 'n_estimators': 14, 'max_samples': 0.8894, 'max_features': 0.8545, 
                            'bootstrap_features': False, 'bootstrap': False, 
                            'base_estimator': XGBRegressor(**{'subsample': 0.5773, 'reg_lambda': 197.4628, 
                                                              'reg_alpha': 106.8185, 'random_state': 15, 'n_estimators': 990, 
                                                              'learning_rate': 0.07, 'gamma': 13.8871, 'colsample_bytree': 0.5363, 
                                                              'colsample_bylevel': 0.8607})})]*16
for i in range(1, 17):
    if i % 2 != 0:
        a = {'colsample_bylevel': 0.5037, 'colsample_bytree': 0.5102, 'gamma': 8.5888, 'learning_rate': 0.0573,
         'n_estimators': 1194, 'random_state': 17, 'reg_alpha': 12.3419, 'reg_lambda': 15.3661, 'subsample': 0.8243}
        a['random_state'] = i
    else:
        a = {'colsample_bylevel': 0.4045,
             'colsample_bytree': 0.3997,
             'gamma': 32.6015,
             'learning_rate': 0.1286,
             'n_estimators': 4190,
             'random_state': 19,
             'reg_alpha': 186.2586,
             'reg_lambda': 505.4068,
             'subsample': 0.3381}
    b = {'base_estimator': XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.5037,
                           colsample_bytree=0.5102, gamma=8.5888, learning_rate=0.0573,
                           max_delta_step=0, max_depth=3, min_child_weight=1, missing=None,
                           n_estimators=1194, n_jobs=1, nthread=None, objective='reg:linear',
                           random_state=17, reg_alpha=12.3419, reg_lambda=15.3661,
                           scale_pos_weight=1, seed=None, silent=True, subsample=0.8243),
         'bootstrap': False,
         'bootstrap_features': True,
         'max_features': 0.9193,
         'max_samples': 0.948,
         'n_estimators': 19,
         'random_state': 16}
    b['base_estimator'] = XGBRegressor(**a)
    b['random_state'] = i + 3
    bg_s.append(BaggingRegressor(**b))
preds = []
for b in bg_s:
    preds.append(list(b.fit(tr_X, tr_y).predict(ts)))
final_p = list(np.mean(preds, axis=0))
submission = pd.DataFrame({
        "Id": ts_ids,
        "SalePrice": final_p
})

submission.to_csv('submission.csv', index=False)
