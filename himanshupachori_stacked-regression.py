import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline

from decimal import Decimal
from scipy import stats
from scipy.stats import norm, skew

from sklearn.preprocessing import MinMaxScaler,LabelEncoder,RobustScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.externals import joblib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

import xgboost as xgb
import lightgbm as lgb

from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge 

from mlxtend.regressor import StackingRegressor
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn
file = pd.ExcelFile('../input/Data science assignment.xlsx')
df_input = file.parse('Sheet4')
indata = df_input.copy(deep=False)
indata.head().round(3)
indata.shape
indata.columns
indata.describe()
indata.info()
def listFactorVars(dataFrame):
    for col in dataFrame:
        if (len(indata[col].unique()) < 15):
            print("{} : {}".format(col, indata[col].unique()))
        else:
            None

catCols=listFactorVars(indata)
display(catCols)
indata['Fuel Type'] = indata['Fuel Type'].map(lambda x: x.upper())
indata['Fuel Type'].value_counts()
indata.isnull().sum()
indata.shape
indata[indata.duplicated(keep=False)].sort_values('MMV')
indata.drop_duplicates(inplace=True)
indata.shape
indata['MMV'].nunique()
indata['Model'].nunique()
indata['Color'].nunique()
indata['Variant'].nunique()
indata.groupby('Make')['Kms', 'No of Owner', 'Heath score','price score'].describe().transpose().round(2)
indata.groupby('Make')['Heath score','on road price', 'Current Price', 'Dep'].describe().transpose().round(2)
sns.distplot(indata['on road price'], fit=norm);
sns.distplot(indata['Current Price'], fit=norm);
sns.distplot(indata['Dep'], fit=norm);
sns.distplot(indata['Heath score'], fit=norm);
sns.distplot(indata['price score'], fit=norm);
sns.distplot(indata['Kms'], fit=norm);
indata.plot('Kms','Dep',kind='scatter')
indata.plot('on road price','Dep',kind='scatter')
indata.plot('price score','Dep',kind='scatter')
indata.plot('Heath score','Dep',kind='scatter')
indata.plot('Current Price','Dep', kind='scatter')
sns.boxplot(x='Age',y='Dep',data=indata)
sns.boxplot(x='No of Owner',y='Dep',data=indata)
indata.sort_values('on road price',ascending=False).head()
#indata = indata.drop(indata[indata['on road price'] == 10954664.00].index)
indata.sort_values('Current Price',ascending=False).head()
indata = indata.drop(indata[indata['Kms'] > 800000].index)
indata = indata.drop(indata[indata['Current Price'] > 2800000].index)
indata.loc[(indata['Dep'] < 40) & indata['No of Owner'].isin([3,4])]
indata.loc[(indata['Dep'] < 20) & indata['No of Owner'].isin([2])]
indata.loc[(indata['Dep'] < 20) & indata['Age'].isin([8,9,10,11,12])]
indata.shape
indata.drop([3724,2235,2324,2000,218,2331,25], inplace=True)
indata.shape
df_sub = indata[['Make', 'Model', 'Variant','Transmission', 'Color', 'No of Owner', 'Kms', 'Type', 'Fuel Type', 'on road price','Dep']]
df_sub.head()
indata.head()
train = indata.drop(columns=['MMV','Variant','Model'])
train.head().round(3)
train['Make'] = train['Make'].apply(str)
train['Type'] = train['Type'].apply(str)
train['Fuel Type'] = train['Fuel Type'].apply(str)
train['Transmission'] = train['Transmission'].apply(str)
train['Color'] = train['Color'].apply(str)

cols = ('Make', 'Type', 'Fuel Type', 'Transmission', 'Color')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(train[c].values)) 
    train[c] = lbl.transform(list(train[c].values))
train.head().round(3)  
dep = train['Dep']
train.drop(columns='Dep',inplace=True)
scaler = MinMaxScaler()
scaler.fit(train)
train = pd.DataFrame(data=scaler.transform(train),columns = train.columns,index=train.index)
train.head()
train = pd.get_dummies(train,prefix=['Make', 'Type', 'Fuel Type','Transmission'],
                       columns=['Make', 'Type', 'Fuel Type','Transmission'])
train.columns
train.head()
X_train, X_test, y_train, y_test = train_test_split(train, dep , test_size=0.3, random_state=101)
X_train.shape
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
linear_reg = LinearRegression(normalize=True)
linear_reg.fit(X_train, y_train)
linear_reg_train_pred = linear_reg.predict(X_train)
print(rmsle(y_train, linear_reg_train_pred))
print(mean_absolute_error(y_train, linear_reg_train_pred))
print('R squared: {}'.format(round(linear_reg.score(X_train, y_train),4)))
linear_reg.fit(X_test, y_test)
linear_reg_test_pred = linear_reg.predict(X_test)
print(rmsle(y_test, linear_reg_test_pred))
print(mean_absolute_error(y_test, linear_reg_test_pred))
print('R squared: {}'.format(round(linear_reg.score(X_test, y_test),4)))
dtree = DecisionTreeRegressor(max_depth=8,random_state=51)
dtree.fit(X_train, y_train)
dtree_train_pred = dtree.predict(X_train)
print(rmsle(y_train, dtree_train_pred))
print(mean_absolute_error(y_train, dtree_train_pred))
print('R squared: {}'.format(round(dtree.score(X_train, y_train),4)))
dtree.fit(X_test, y_test)
dtree_test_pred = dtree.predict(X_test)
print(rmsle(y_test, dtree_test_pred))
print(mean_absolute_error(y_test, dtree_test_pred))
print('R squared: {}'.format(round(dtree.score(X_test, y_test),4)))
feat_imp = pd.Series(dtree.feature_importances_, X_train.columns).sort_values(ascending=False)
feat_imp = feat_imp[feat_imp > 0.0]
plt.figure(figsize=(16,6))
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')
random_forest = RandomForestRegressor(max_depth=8, random_state=51)
random_forest.fit(X_train, y_train)
rforest_train_pred = random_forest.predict(X_train)
print(rmsle(y_train, rforest_train_pred))
print(mean_absolute_error(y_train, rforest_train_pred))
print('R squared: {}'.format(round(random_forest.score(X_train, y_train),4)))
random_forest.fit(X_test, y_test)
rforest_test_pred = random_forest.predict(X_test)
print(rmsle(y_test, rforest_test_pred))
print(mean_absolute_error(y_test, rforest_test_pred))
print('R squared: {}'.format(round(random_forest.score(X_test, y_test),4)))
feat_imp = pd.Series(random_forest.feature_importances_, X_train.columns).sort_values(ascending=False)
feat_imp = feat_imp[feat_imp > 0.0]
plt.figure(figsize=(16,6))
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')
ada_boost = AdaBoostRegressor(RandomForestRegressor(max_depth=6), n_estimators=300,loss='linear' , random_state=51)
ada_boost.fit(X_train, y_train)
adaboost_train_pred = ada_boost.predict(X_train)
print(rmsle(y_train, adaboost_train_pred))
print(mean_absolute_error(y_train, adaboost_train_pred))
print('R squared: {}'.format(round(ada_boost.score(X_train, y_train),4)))
ada_boost.fit(X_test, y_test)
adaboost_test_pred = ada_boost.predict(X_test)
print(rmsle(y_test, adaboost_test_pred))
print(mean_absolute_error(y_test, adaboost_test_pred))
print('R squared: {}'.format(round(ada_boost.score(X_test, y_test),4)))
feat_imp = pd.Series(ada_boost.feature_importances_, X_train.columns).sort_values(ascending=False)
feat_imp = feat_imp[feat_imp > 0.0]
plt.figure(figsize=(16,6))
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')
GBoost = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1,
                                   max_depth=6, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =51)
GBoost.fit(X_train, y_train)
GBoost_train_pred = GBoost.predict(X_train)
print(rmsle(y_train, GBoost_train_pred))
print(mean_absolute_error(y_train, GBoost_train_pred))
print('R squared: {}'.format(round(GBoost.score(X_train, y_train),4)))
GBoost.fit(X_test, y_test)
GBoost_test_pred = GBoost.predict(X_test)
print(rmsle(y_test, GBoost_test_pred))
print(mean_absolute_error(y_test, GBoost_test_pred))
print('R squared: {}'.format(round(GBoost.score(X_test, y_test),4)))
feat_imp = pd.Series(GBoost.feature_importances_, X_train.columns).sort_values(ascending=False)
feat_imp = feat_imp[feat_imp > 0.0]
plt.figure(figsize=(16,6))
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')
model_xgb = xgb.XGBRegressor(learning_rate=0.01, max_depth=6,n_estimators=1000,random_state =51, nthread = -1)
model_xgb.fit(X_train, y_train)
xgb_train_pred = model_xgb.predict(X_train)
print(rmsle(y_train, xgb_train_pred))
print(mean_absolute_error(y_train, xgb_train_pred))
print('R squared: {}'.format(round(model_xgb.score(X_train, y_train),4)))
model_xgb.fit(X_test, y_test)
xgb_test_pred = model_xgb.predict(X_test)
print(rmsle(y_test, xgb_test_pred))
print(mean_absolute_error(y_test, xgb_test_pred))
print('R squared: {}'.format(round(model_xgb.score(X_test, y_test),4)))
feat_imp = pd.Series(model_xgb.feature_importances_, X_train.columns).sort_values(ascending=False)
feat_imp = feat_imp[feat_imp > 0.0]
plt.figure(figsize=(16,6))
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.1, n_estimators=1000,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
model_lgb.fit(X_train, y_train)
lgb_train_pred = model_lgb.predict(X_train)
print(rmsle(y_train, lgb_train_pred))
print(mean_absolute_error(y_train, lgb_train_pred))
print('R squared: {}'.format(round(model_lgb.score(X_train, y_train),4)))
model_lgb.fit(X_test, y_test)
lgb_test_pred = model_lgb.predict(X_test)
print(rmsle(y_test, lgb_test_pred))
print(mean_absolute_error(y_test, lgb_test_pred))
print('R squared: {}'.format(round(model_lgb.score(X_test, y_test),4)))
feat_imp = pd.Series(model_lgb.feature_importances_, X_train.columns).sort_values(ascending=False)
feat_imp = feat_imp[feat_imp > 0.0]
plt.figure(figsize=(16,6))
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')
st_regr = StackingRegressor(regressors=[model_lgb, GBoost, ada_boost], meta_regressor=model_xgb)
st_regr.fit(X_train, y_train)
st_regr_train_pred = st_regr.predict(X_train)
print(rmsle(y_train, st_regr_train_pred))
print(mean_absolute_error(y_train, st_regr_train_pred))
print('R squared: {}'.format(round(st_regr.score(X_train, y_train),4)))
st_regr.fit(X_test, y_test)
st_regr_test_pred = st_regr.predict(X_test)
print(rmsle(y_test, st_regr_test_pred))
print(mean_absolute_error(y_test, st_regr_test_pred))
print('R squared: {}'.format(round(st_regr.score(X_test, y_test),4)))
x = pd.DataFrame(data=st_regr_train_pred,index=y_train.index)
y = pd.DataFrame(data=st_regr_test_pred,index=y_test.index)
z = pd.concat([x,y])
z.sort_index(ascending=True).head()
df_sub.head()
df_sub['Predicted_dep'] = pd.DataFrame(z)
df_sub.head(10).round(3)
df_sub.describe()
# joblib.dump(st_regr, '../input/trained_car_regreesor_model.pkl')
# df_sub.to_csv('../Datasets/Final_submission.csv',index=False)
# df_sub.to_excel('../Datasets/Final_submission.xlsx',index=False)