import numpy as np

import pandas as pd

import math

from scipy import stats
train = pd.read_csv('../input/train.csv')

train.info()
import seaborn as sns

import matplotlib.pyplot as plt
sns.heatmap(data=train.corr())

plt.show()

plt.gcf().clear()
train.skew()
train_d = train.copy()

train_d = pd.get_dummies(train_d)
keep_cols = train_d.select_dtypes(include=['number']).columns

train_d = train_d[keep_cols]
train_d = train_d.fillna(train_d.mean())
test = pd.read_csv('../input/test.csv')
test_d = test.copy()

test_d = pd.get_dummies(test_d)
test_d = test_d.fillna(test_d.mean())
for col in keep_cols:

    if col not in test_d:

        test_d[col] = 0
test_d = test_d[keep_cols]
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV
rf_test = RandomForestRegressor(max_depth=30, n_estimators=500, max_features = 100, oob_score=True, random_state=1234)

cv_score = cross_val_score(rf_test, train_d.drop('SalePrice', axis = 1), train_d['SalePrice'], cv = 5, n_jobs = -1)
print('CV Score is: '+ str(np.mean(cv_score)))
train_0 = train.copy()
null_index = train_0.LotFrontage.isnull()

train_0.loc[null_index,'LotFrontage'] = 0
train_0 = pd.get_dummies(train_0)
keep_cols = train_0.select_dtypes(include=['number']).columns

train_0 = train_0[keep_cols]
train_0 = train_0.fillna(train_0.mean())
rf_test = RandomForestRegressor(max_depth=30, n_estimators=500, max_features = 100, oob_score=True, n_jobs=-1, random_state=1234)

cv_score = cross_val_score(rf_test, train_0.drop('SalePrice', axis = 1), train_0['SalePrice'], cv = 5, n_jobs=-1)
print('CV Score is: '+ str(np.mean(cv_score)))
sns.barplot(data=train,x='Neighborhood',y='LotFrontage', estimator=np.median)

plt.xticks(rotation=90)

plt.show()

plt.gcf().clear()
gb_neigh_LF = train['LotFrontage'].groupby(train['Neighborhood'])
train_LFm = train.copy()
# for the key (the key is neighborhood in this case), and the group object (group is LotFrontage grouped by Neighborhood) 

# associated with it...

for key,group in gb_neigh_LF:

    # find where we are both simultaneously missing values and where the key exists

    lot_f_nulls_nei = train['LotFrontage'].isnull() & (train['Neighborhood'] == key)

    # fill in those blanks with the median of the key's group object

    train_LFm.loc[lot_f_nulls_nei,'LotFrontage'] = group.median()
train_LFm = pd.get_dummies(train_LFm)
keep_cols = train_LFm.select_dtypes(include=['number']).columns

train_LFm = train_LFm[keep_cols]
train_LFm = train_LFm.fillna(train_LFm.mean())
rf_test = RandomForestRegressor(max_depth=30, n_estimators=500, max_features = 100, oob_score=True, n_jobs=-1, random_state= 1234)

cv_score = cross_val_score(rf_test, train_LFm.drop('SalePrice', axis = 1), train_LFm['SalePrice'], cv = 5, n_jobs=-1)
print('CV Score is: '+ str(np.mean(cv_score)))
train_med = train.copy()
# for the key (the key is neighborhood in this case), and the group object (group is LotFrontage grouped by Neighborhood) 

# associated with it...

for key,group in gb_neigh_LF:

    # find where we are both simultaneously missing values and where the key exists

    lot_f_nulls_nei = train['LotFrontage'].isnull() & (train['Neighborhood'] == key)

    # fill in those blanks with the median of the key's group object

    train_med.loc[lot_f_nulls_nei,'LotFrontage'] = group.median()
train_med = pd.get_dummies(train_med)
keep_cols = train_med.select_dtypes(include=['number']).columns

train_med = train_med[keep_cols]
train_med = train_med.fillna(train_med.median())
rf_test = RandomForestRegressor(max_depth=30, n_estimators=500, max_features = 100, oob_score=True, n_jobs=-1, random_state=1234)

cv_score = cross_val_score(rf_test, train_med.drop('SalePrice', axis = 1), train_med['SalePrice'], cv = 5, n_jobs=-1)
print('CV Score is: '+ str(np.mean(cv_score)))
rf_test.fit(train_med.drop('SalePrice',axis = 1),train_med['SalePrice'])
sns.barplot(data=train, x='MSSubClass', y='SalePrice')

plt.show()

plt.gcf().clear()
from xgboost.sklearn import XGBRegressor
xgb_test = XGBRegressor(learning_rate=0.05,n_estimators=500,max_depth=3,colsample_bytree=0.4)

cv_score = cross_val_score(xgb_test, train_med.drop(['SalePrice','Id'], axis = 1), train_med['SalePrice'], cv = 5, n_jobs = -1)
print('CV Score is: '+ str(np.mean(cv_score)))
has_rank = [col for col in train if 'TA' in list(train[col])]
dic_num = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
train_c2n = train.copy()
train_c2n['MSSubClass'] = train_c2n['MSSubClass'].astype('category')
# for the key (the key is neighborhood in this case), and the group object (group is LotFrontage grouped by Neighborhood) 

# associated with it...

for key,group in gb_neigh_LF:

    # find where we are both simultaneously missing values and where the key exists

    lot_f_nulls_nei = train['LotFrontage'].isnull() & (train['Neighborhood'] == key)

    # fill in those blanks with the median of the key's group object

    train_c2n.loc[lot_f_nulls_nei,'LotFrontage'] = group.median()
for col in has_rank:

    train_c2n[col+'_2num'] = train_c2n[col].map(dic_num)
train_c2n = pd.get_dummies(train_c2n)
train_cols = train_c2n.select_dtypes(include=['number']).columns

train_c2n = train_c2n[train_cols]
train_c2n = train_c2n.fillna(train_c2n.median())
xgb_test = XGBRegressor(learning_rate=0.05,n_estimators=500,max_depth=3,colsample_bytree=0.4)

cv_score = cross_val_score(xgb_test, train_c2n.drop(['SalePrice','Id'], axis = 1), train_c2n['SalePrice'], cv = 5, n_jobs=-1)
print('CV Score is: '+ str(np.mean(cv_score)))
from statistics import mode
low_var_cat = [col for col in train.select_dtypes(exclude=['number']) if 1 - sum(train[col] == mode(train[col]))/len(train) < 0.03]

low_var_cat
has_rank = [col for col in train if 'TA' in list(train[col])]
dic_num = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
train_col = train.copy()
train_col = train_col.drop(['Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating'], axis = 1)
train_col['MSSubClass'] = train_col['MSSubClass'].astype('category')
# for the key (the key is neighborhood in this case), and the group object (group is LotFrontage grouped by Neighborhood) 

# associated with it...

for key,group in gb_neigh_LF:

    # find where we are both simultaneously missing values and where the key exists

    lot_f_nulls_nei = train['LotFrontage'].isnull() & (train['Neighborhood'] == key)

    # fill in those blanks with the median of the key's group object

    train_col.loc[lot_f_nulls_nei,'LotFrontage'] = group.median()
for col in has_rank:

    train_col[col+'_2num'] = train_col[col].map(dic_num)
train_col = pd.get_dummies(train_col)
train_cols = train_col.select_dtypes(include=['number']).columns

train_col = train_col[train_cols]
train_col = train_col.fillna(train_col.median())
xgb_test = XGBRegressor(learning_rate=0.05,n_estimators=500,max_depth=3,colsample_bytree=0.4)

cv_score = cross_val_score(xgb_test, train_col.drop(['SalePrice','Id'], axis = 1), train_col['SalePrice'], cv = 5, n_jobs=-1)
print('CV Score is: '+ str(np.mean(cv_score)))
cat_hasnull = [col for col in train.select_dtypes(['object']) if train[col].isnull().any()]

cat_hasnull
cat_hasnull.remove('Electrical')
mode_elec = mode(train['Electrical'])

mode_elec
has_rank = [col for col in train if 'TA' in list(train[col])]
dic_num = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
cat_hasnull = [col for col in train.select_dtypes(['object']) if train[col].isnull().any()]
cat_hasnull.remove('Electrical')
train_none = train.copy()
train_none = train_none.drop(['Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating'], axis = 1)
train_none['MSSubClass'] = train_none['MSSubClass'].astype('category')
for col in cat_hasnull:

    null_idx = train_none[col].isnull()

    train_none.loc[null_idx, col] = 'None'
null_idx_el = train_none['Electrical'].isnull()

train_none.loc[null_idx_el, 'Electrical'] = 'SBrkr'
# for the key (the key is neighborhood in this case), and the group object (group is LotFrontage grouped by Neighborhood) 

# associated with it...

for key,group in gb_neigh_LF:

    # find where we are both simultaneously missing values and where the key exists

    lot_f_nulls_nei = train['LotFrontage'].isnull() & (train['Neighborhood'] == key)

    # fill in those blanks with the median of the key's group object

    train_none.loc[lot_f_nulls_nei,'LotFrontage'] = group.median()
for col in has_rank:

    train_none[col+'_2num'] = train_none[col].map(dic_num)
train_none = pd.get_dummies(train_none)
train_cols = train_none.select_dtypes(include=['number']).columns

train_none = train_none[train_cols]
train_none = train_none.fillna(train_none.median())
xgb_test = XGBRegressor(learning_rate=0.05,n_estimators=500,max_depth=3,colsample_bytree=0.4)

cv_score = cross_val_score(xgb_test, train_none.drop(['SalePrice','Id'], axis = 1), train_none['SalePrice'], cv = 5, n_jobs=-1)
print('CV Score is: '+ str(np.mean(cv_score)))
cols_skew = [col for col in train_none if '_2num' in col or '_' not in col]

train_none[cols_skew].skew()
cols_unskew = train_none[cols_skew].columns[abs(train_none[cols_skew].skew()) > 1]
train_unskew = train_none.copy()
for col in cols_unskew:

    train_unskew[col] = np.log1p(train_none[col])
xgb_test = XGBRegressor(learning_rate=0.05,n_estimators=500,max_depth=3,colsample_bytree=0.4)

cv_score = cross_val_score(xgb_test, train_unskew.drop(['SalePrice','Id'], axis = 1), train_unskew['SalePrice'], cv = 5, n_jobs=-1)
print('CV Score is: '+ str(np.mean(cv_score)))
from sklearn.linear_model import LassoCV

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

LCV = LassoCV()

scale_LCV = Pipeline([('scaler',scaler),('LCV',LCV)])



cv_score = cross_val_score(scale_LCV, train_unskew.drop(['SalePrice','Id'], axis = 1), train_unskew['SalePrice'], cv = 5, n_jobs=-1)
print('CV Score is: '+ str(np.mean(cv_score)))
scale_LCV.fit(train_unskew.drop(['SalePrice','Id'], axis = 1), train_unskew['SalePrice'])
lasso_w = scale_LCV.named_steps['LCV'].coef_

cols= train_unskew.drop(['SalePrice','Id'], axis=1).columns
cols_w = pd.DataFrame()

cols_w['Features'] = cols

cols_w['LassoWeights'] = lasso_w

cols_w['LassoWeightsMag'] = abs(lasso_w)
cols_w[cols_w.LassoWeights==0]
cols_w.sort_values(by='LassoWeightsMag',ascending=False)[:15]
top5_feats = list(cols_w.sort_values(by='LassoWeightsMag',ascending=False)[:5].Features)
def print_scatters(df_in, cols, against):   

    plt.figure(1)

    # sets the number of figure row (ie: for 10 variables, we need 5, for 9 we 

    # need 5 as well)

    rows = math.ceil(len(cols)/2)

    f, axarr = plt.subplots(rows, 2, figsize=(10, rows*3))

    # for each variable you inputted, plot it against the dependant

    for col in cols:

        ind = cols.index(col)

        i = math.floor(ind/2)

        j = 0 if ind % 2 == 0 else 1

        if col != against:

            sns.regplot(data = df_in, x=col, y=against, fit_reg=False, ax=axarr[i,j])

        else:

            sns.distplot(a = df_in[col], ax=axarr[i,j])

        axarr[i, j].set_title(col)

    f.text(-0.01, 0.5, against, va='center', rotation='vertical', fontsize = 12)

    plt.tight_layout()

    plt.show()

    plt.gcf().clear()
print_scatters(df_in=train,cols=['SalePrice']+top5_feats,against='SalePrice')
sns.lmplot(data=train, x='GrLivArea',y='SalePrice')

plt.show()

plt.gcf().clear()
sns.lmplot(data=train[train.GrLivArea < 4500], x='GrLivArea',y='SalePrice')

plt.show()

plt.gcf().clear()
import statsmodels.api as sm
bonf_outliers = [88,462,523,588,632,968,1298,1324]
train_test_raw = train.append(test)
lot_frontage_by_neighborhood_all = train_test_raw["LotFrontage"].groupby(train_test_raw["Neighborhood"])
has_rank = [col for col in train if 'TA' in list(train[col])]
dic_num = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
cat_hasnull = [col for col in train.select_dtypes(['object']) if train[col].isnull().any()]
cat_hasnull.remove('Electrical')
train_c2n = train.copy()
for key,group in lot_frontage_by_neighborhood_all:

    lot_f_nulls_nei = train['LotFrontage'].isnull() & (train['Neighborhood'] == key)

    train_c2n.loc[lot_f_nulls_nei,'LotFrontage'] = group.median()
train_c2n = train_c2n.drop(['Street','Utilities','Condition2','RoofMatl','Heating'], axis=1)
train_c2n['MSSubClass'] = train_c2n['MSSubClass'].astype('category')
for col in cat_hasnull:

    null_idx = train_c2n[col].isnull()

    train_c2n.loc[null_idx, col] = 'None'
null_idx_el = train_c2n['Electrical'].isnull()

train_c2n.loc[null_idx_el, 'Electrical'] = 'SBrkr'
for col in has_rank:

    train_c2n[col+'_2num'] = train_c2n[col].map(dic_num)
train_c2n = pd.get_dummies(train_c2n)
train_cols = train_c2n.select_dtypes(include=['number']).columns

train_c2n = train_c2n[train_cols]
test_c2n = test.copy()
# See Human Analog

test_c2n.loc[666, "GarageQual"] = "TA"

test_c2n.loc[666, "GarageCond"] = "TA"

test_c2n.loc[666, "GarageFinish"] = "Unf"

test_c2n.loc[666, "GarageYrBlt"] = 1980



test_c2n.loc[1116,'GarageType'] = np.nan
for key,group in lot_frontage_by_neighborhood_all:

    lot_f_nulls_nei = test['LotFrontage'].isnull() & (test['Neighborhood'] == key)

    test_c2n.loc[lot_f_nulls_nei,'LotFrontage'] = group.median()
test_c2n = test_c2n.drop(['Street','Utilities','Condition2','RoofMatl','Heating'], axis=1)
test_c2n['MSSubClass'] = test_c2n['MSSubClass'].astype('category')
for col in cat_hasnull:

    null_idx = test_c2n[col].isnull()

    test_c2n.loc[null_idx, col] = 'None'
null_idx_el = test_c2n['Electrical'].isnull()

test_c2n.loc[null_idx_el, 'Electrical'] = 'SBrkr'
for col in has_rank:

    test_c2n[col+'_2num'] = test_c2n[col].map(dic_num)
test_c2n = pd.get_dummies(test_c2n)
test_c2n['SalePrice'] = 0
for col in train_cols:

    if col not in test_c2n:

        train_c2n = train_c2n.drop(col,axis=1)
test_c2n = test_c2n.drop('MSSubClass_150', axis = 1)
final_cols = test_c2n.select_dtypes(include=['number']).columns

test_c2n = test_c2n[final_cols]
test_c2n = test_c2n[train_c2n.columns]
train_test_combo = train_c2n.append(test_c2n)

train_test_raw = train.append(test)
train_test_combo = train_test_combo.fillna(train_test_combo.median())
train_med = train_test_combo[:1460]

test_med = train_test_combo[1460:]
cols = [col for col in train_med if '_2num' in col or '_' not in col]

skew = [abs(stats.skew(train_med[col])) for col in train_med if '_2num' in col or '_' not in col]
skews = pd.DataFrame()

skews['Columns'] = cols

skews['Skew_Magnintudes'] = skew
cols_unskew = skews[skews.Skew_Magnintudes > 1].Columns
train_unskew2 = train_med.copy()

test_unskew2 = test_med.copy()
for col in cols_unskew:

    train_unskew2[col] = np.log1p(train_med[col])

    

for col in cols_unskew:

    test_unskew2[col] = np.log1p(test_med[col])
bonf_outlier = [88,462,523,588,632,968,1298,1324]
train_unskew3 = train_unskew2.drop(bonf_outlier)
drop_cols = ["MSSubClass_160", "MSZoning_C (all)"]
train_unskew3 = train_unskew3.drop(drop_cols, axis = 1)

test_unskew2 = test_unskew2.drop(drop_cols, axis = 1)
X_train = train_unskew3.drop(['Id','SalePrice'],axis = 1)

y_train = train_unskew3['SalePrice']
X_test = test_unskew2.drop(['Id','SalePrice'],axis=1)
scaler = StandardScaler()

LCV = LassoCV()

scale_LCV = Pipeline([('scaler',scaler),('LCV',LCV)])



cv_score = cross_val_score(scale_LCV, X_train, y_train, cv = 5, n_jobs=-1)
print('CV Score is: '+ str(np.mean(cv_score)))
from sklearn.base import BaseEstimator, RegressorMixin
class CustomEnsembleRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, regressors=None):

        self.regressors = regressors



    def fit(self, X, y):

        for regressor in self.regressors:

            regressor.fit(X, y)



    def predict(self, X):

        self.predictions_ = list()

        for regressor in self.regressors:

            self.predictions_.append((regressor.predict(X).ravel()))

        return (np.mean(self.predictions_, axis=0))
xgb1 = XGBRegressor(colsample_bytree=0.2,

                 learning_rate=0.05,

                 max_depth=3,

                 n_estimators=1200

                )



xgb2 = XGBRegressor(colsample_bytree=0.2,

                 learning_rate=0.05,

                 max_depth=3,

                 n_estimators=1200,

                seed = 1234

                )



xgb3 = XGBRegressor(colsample_bytree=0.2,

                 learning_rate=0.05,

                 max_depth=3,

                 n_estimators=1200,

                seed = 1337

                )
xgb_ens = CustomEnsembleRegressor([xgb1,xgb2,xgb3])
cvscore = cross_val_score(cv=5,estimator=xgb1,X = X_train,y = y_train, n_jobs = -1)
print('CV Score is: '+ str(np.mean(cvscore)))
xgb_ens.fit(X_train, y_train);

scale_LCV.fit(X_train,y_train);
preds_x = np.expm1(xgb_ens.predict(X_test));

preds_l = np.expm1(scale_LCV.predict(X_test));

preds = (preds_x+preds_l)/2

out_preds = pd.DataFrame()

out_preds['Id'] = test['Id']

out_preds['SalePrice'] = preds

out_preds.to_csv('output.csv', index=False)