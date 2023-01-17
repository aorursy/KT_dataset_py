PATH = '../input/employee-attrition-rate/'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (8,6)
import seaborn as sns
sns.set(style='darkgrid')
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
test = pd.read_csv(PATH+'Test.csv')
train = pd.read_csv(PATH+'Train.csv')
sample = pd.read_csv(PATH+'sample_submission.csv')

train.shape, test.shape
target = 'Attrition_rate'
catfeat, numfeat = list(train.select_dtypes(exclude=np.number)), list(train.select_dtypes(include=np.number))
numfeat.remove(target)

df = pd.concat([train,test], keys=['train', 'test'])
df
df.info()
## CREATING A TEMPORARY DATAFRAME TO HANDLE MISSING VALUES
temp = pd.concat([df['Age'], df['Time_of_service'], df['Pay_Scale'], df['Work_Life_balance'], df['VAR2'], df['VAR4']], axis=1)
temp.head(20)
sns.heatmap(temp.corr(), annot = True); plt.show()
## PERCENT OF MISSING VALUES
temp.isnull().sum()/len(temp)*100
temp.info()
print('Skewness: '); print(temp.skew())
import missingno as mnso; mnso.heatmap(temp, figsize=(8,6)); plt.show()
df['Age'].fillna(method='ffill', inplace=True)
df['Time_of_service'].fillna(method='ffill', inplace=True)
df['Pay_Scale'].fillna(method='ffill', inplace=True)
df['Work_Life_balance'].fillna(df['Work_Life_balance'].median(), inplace=True)
df['VAR2'].fillna(df['VAR2'].median(), inplace=True)
df['VAR4'].fillna(df['VAR4'].median(), inplace=True)
catfeat
temp = catfeat[1]
f, a = plt.subplots(1,3,figsize=(24,6))
sns.boxplot(x=temp, y=target, data=df,ax=a[0])
a[0].set_title('Bivariate plot')
sns.violinplot(x=temp, y=target, data=df, ax=a[1])
a[1].set_title('Bivariate plot')
sns.countplot(df[temp],ax=a[2])
a[2].set_title('Univariate plot')
plt.show()
temp = catfeat[2]
f, a = plt.subplots(1,3,figsize=(24,6))
sns.boxplot(x=temp, y=target, data=df,ax=a[0])
a[0].set_title('Bivariate plot')
sns.violinplot(x=temp, y=target, data=df, ax=a[1])
a[1].set_title('Bivariate plot')
sns.countplot(df[temp],ax=a[2])
a[2].set_title('Univariate plot')
plt.show()
temp = catfeat[3]
f, a = plt.subplots(1,3,figsize=(24,6))
sns.boxplot(x=temp, y=target, data=df,ax=a[0])
a[0].set_title('Bivariate plot')
sns.violinplot(x=temp, y=target, data=df, ax=a[1])
a[1].set_title('Bivariate plot')
sns.countplot(df[temp],ax=a[2])
a[2].set_title('Univariate plot')
plt.show()
temp = catfeat[4]
f, a = plt.subplots(1,3,figsize=(24,6))
sns.boxplot(x=temp, y=target, data=df,ax=a[0]).set_xticklabels(list(df[temp].unique()), rotation=45, horizontalalignment='right')
a[0].set_title('Bivariate plot')
sns.violinplot(x=temp, y=target, data=df, ax=a[1]).set_xticklabels(list(df[temp].unique()), rotation=45, horizontalalignment='right')
a[1].set_title('Bivariate plot')
sns.countplot(df[temp],ax=a[2]).set_xticklabels(list(df[temp].unique()), rotation=45, horizontalalignment='right')
a[2].set_title('Univariate plot')
plt.show()
temp = catfeat[5]
f, a = plt.subplots(1,3,figsize=(24,6))
sns.boxplot(x=temp, y=target, data=df,ax=a[0])
a[0].set_title('Bivariate plot')
sns.violinplot(x=temp, y=target, data=df, ax=a[1])
a[1].set_title('Bivariate plot')
sns.countplot(df[temp],ax=a[2])
a[2].set_title('Univariate plot')
plt.show()
temp = catfeat[6]
f, a = plt.subplots(1,3,figsize=(24,6))
sns.boxplot(x=temp, y=target, data=df,ax=a[0])
a[0].set_title('Bivariate plot')
sns.violinplot(x=temp, y=target, data=df, ax=a[1])
a[1].set_title('Bivariate plot')
sns.countplot(df[temp],ax=a[2])
a[2].set_title('Univariate plot')
plt.show()
print(numfeat)
temp = numfeat[0]
f, a = plt.subplots(1,2,figsize=(16,6))
sns.scatterplot(x=temp, y=target, data=df,ax=a[0])
a[0].set_title('Bivariate plot')
sns.distplot(df[temp],ax=a[1],fit=stats.norm)
a[1].set_title('Univariate plot, skew: {:.4f}'.format(df[temp].skew()))
plt.show()
temp = numfeat[1]
f, a = plt.subplots(1,2,figsize=(16,6))
sns.scatterplot(x=temp, y=target, data=df,ax=a[0])
a[0].set_title('Bivariate plot')
sns.distplot(df[temp],ax=a[1],fit=stats.norm)
a[1].set_title('Univariate plot, skew: {:.4f}'.format(df[temp].skew()))
plt.show()
temp = numfeat[2]
f, a = plt.subplots(1,2,figsize=(16,6))
sns.scatterplot(x=temp, y=target, data=df,ax=a[0])
a[0].set_title('Bivariate plot')
sns.distplot(df[temp],ax=a[1],fit=stats.norm)
a[1].set_title('Univariate plot, skew: {:.4f}'.format(df[temp].skew()))
plt.show()
trans = stats.boxcox(1+df[temp], stats.boxcox_normmax(1+df[temp]))
sns.distplot(trans, fit = stats.norm)
plt.title('Transformed univariate plot, skew: {:.4f}'.format(pd.Series(trans).skew()))
df[temp] = stats.boxcox(1+df[temp], stats.boxcox_normmax(1+df[temp]))
plt.show()
temp = numfeat[3]
f, a = plt.subplots(1,2,figsize=(16,6))
sns.scatterplot(x=temp, y=target, data=df,ax=a[0])
a[0].set_title('Bivariate plot')
sns.distplot(df[temp],ax=a[1],fit=stats.norm)
a[1].set_title('Univariate plot, skew: {:.4f}'.format(df[temp].skew()))
plt.show()
temp = numfeat[4]
f, a = plt.subplots(1,2,figsize=(16,6))
sns.scatterplot(x=temp, y=target, data=df,ax=a[0])
a[0].set_title('Bivariate plot')
sns.distplot(df[temp],ax=a[1],fit=stats.norm)
a[1].set_title('Univariate plot, skew: {:.4f}'.format(df[temp].skew()))
plt.show()
temp = numfeat[5]
f, a = plt.subplots(1,2,figsize=(16,6))
sns.scatterplot(x=temp, y=target, data=df,ax=a[0])
a[0].set_title('Bivariate plot')
sns.distplot(df[temp],ax=a[1],fit=stats.norm)
a[1].set_title('Univariate plot, skew: {:.4f}'.format(df[temp].skew()))
plt.show()
temp = numfeat[6]
f, a = plt.subplots(1,2,figsize=(16,6))
sns.scatterplot(x=temp, y=target, data=df,ax=a[0])
a[0].set_title('Bivariate plot')
sns.distplot(df[temp],ax=a[1],fit=stats.norm)
a[1].set_title('Univariate plot, skew: {:.4f}'.format(df[temp].skew()))
plt.show()
temp = numfeat[7]
f, a = plt.subplots(1,2,figsize=(16,6))
sns.scatterplot(x=temp, y=target, data=df,ax=a[0])
a[0].set_title('Bivariate plot')
sns.distplot(df[temp],ax=a[1],fit=stats.norm)
a[1].set_title('Univariate plot, skew: {:.4f}'.format(df[temp].skew()))
plt.show()
temp = numfeat[8]
f, a = plt.subplots(1,2,figsize=(16,6))
sns.scatterplot(x=temp, y=target, data=df,ax=a[0])
a[0].set_title('Bivariate plot')
sns.distplot(df[temp],ax=a[1],fit=stats.norm)
a[1].set_title('Univariate plot, skew: {:.4f}'.format(df[temp].skew()))
plt.show()
temp = numfeat[9]
f, a = plt.subplots(1,2,figsize=(16,6))
sns.scatterplot(x=temp, y=target, data=df,ax=a[0])
a[0].set_title('Bivariate plot')
sns.distplot(df[temp],ax=a[1], kde=False)
a[1].set_title('Univariate plot, skew: {:.4f}'.format(df[temp].skew()))
plt.show()
## FEATURE HAS BEEN MISCLASSIFIED AS NUMERIC. CONVERTING TO CATEGORICAL FEATURE
df[numfeat[9]] = df[numfeat[9]].astype('str')
temp = numfeat[10]
f, a = plt.subplots(1,2,figsize=(16,6))
sns.scatterplot(x=temp, y=target, data=df,ax=a[0])
a[0].set_title('Bivariate plot')
sns.distplot(df[temp],ax=a[1],fit=stats.norm)
a[1].set_title('Univariate plot, skew: {:.4f}'.format(df[temp].skew()))
plt.show()
temp = numfeat[11]
f, a = plt.subplots(1,2,figsize=(16,6))
sns.scatterplot(x=temp, y=target, data=df,ax=a[0])
a[0].set_title('Bivariate plot')
sns.distplot(df[temp],ax=a[1],fit=stats.norm)
a[1].set_title('Univariate plot, skew: {:.4f}'.format(df[temp].skew()))
plt.show()
temp = numfeat[12]
f, a = plt.subplots(1,2,figsize=(16,6))
sns.scatterplot(x=temp, y=target, data=df,ax=a[0])
a[0].set_title('Bivariate plot')
sns.distplot(df[temp],ax=a[1],kde=False)
a[1].set_title('Univariate plot, skew: {:.4f}'.format(df[temp].skew()))
plt.show()
temp = numfeat[13]
f, a = plt.subplots(1,2,figsize=(16,6))
sns.scatterplot(x=temp, y=target, data=df,ax=a[0])
a[0].set_title('Bivariate plot')
sns.distplot(df[temp],ax=a[1],fit=stats.norm)
a[1].set_title('Univariate plot, skew: {:.4f}'.format(df[temp].skew()))
plt.show()
## Edited after fitting a baseline model and looking at feature importances
df['Age_Group'] = pd.cut(df['Age'], [18,35,50,70], labels = ['Young', 'Middle_Aged', 'Old'])
df['Pay_Scale'] = df['Pay_Scale'].max() - df['Pay_Scale']
df['growth_rate'] = df['growth_rate'].max() - df['growth_rate']
df['tos_timesincepromo'] = abs(df['Time_of_service']-df['Time_since_promotion'])
df['level'] = abs(df['Post_Level']- df['Education_Level'])
# df['tr_wlb'] = df['Work_Life_balance']/df['Travel_Rate']
df['tos_whole'] = np.fix(df['Time_of_service'])
df['tos_frac'] = df['Time_of_service'] - np.fix(df['Time_of_service'])
df['VAR'] = (df['VAR2']+df['VAR3']+df['VAR4']+df['VAR5']+df['VAR6']+df['VAR7'])/6
df['VAR_sqrt'] = np.sqrt(df['VAR4']**2+df['VAR2']**2+df['VAR3']**2+df['VAR5']**2+df['VAR6']**2+df['VAR7']**2)
df['VAR3'] = df['VAR3'].astype('str')
df['VAR4'] = df['VAR4'].astype('str')
# df['VAR7'] = df['VAR7'].astype('str')
df.describe()
from sklearn import preprocessing as prep

minmax = prep.MinMaxScaler()

catfeat, numfeat = list(train.select_dtypes(exclude=np.number)), list(train.select_dtypes(include=np.number))
numfeat.remove(target)

df[numfeat] = minmax.fit_transform(df[numfeat])
df.drop('Employee_ID', axis=1, inplace=True)
df1 = pd.get_dummies(df)

x, y, X_test = df1.xs('train').drop(target, axis=1), df1.xs('train')[target], df1.xs('test').drop(target,axis=1)
x.shape, y.shape, X_test.shape
from sklearn.decomposition import PCA

pca = PCA(n_components=3)

x_pca = pca.fit_transform(x)
X_test_pca = pca.fit_transform(X_test)
x_pca, X_test_pca= pd.DataFrame(x_pca),pd.DataFrame(X_test_pca) 
x_pca.shape, X_test_pca.shape
## A dictionary to store scores of models
scoresd = dict()
from sklearn.linear_model import Lasso 
from sklearn import model_selection as ms
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold

kf = KFold(n_splits=15,shuffle = True, random_state=100)
parameters= {'alpha':[0.0001,0.0009,0.001,0.01,0.1,1,10],
            'max_iter':[100,500,1000, 50, 20]
             }


lasso = Lasso()
lasso_model = ms.GridSearchCV(lasso, param_grid=parameters, cv=10)
lasso_model.fit(x,y)

print('The best value of Alpha is: ',lasso_model.best_params_)
lss, scores = Lasso(alpha=0.01, max_iter=100), list()

for train, test in kf.split(x,y):
    x_train, y_train = x.loc[train], y.loc[train]
    x_test, y_test = x.loc[test], y.loc[test]

    lss.fit(x_train,y_train)
    y_pred = lss.predict(x_test)

    temp = mean_squared_error(y_pred,y_test)
    scores.append(100*max(0,1-np.sqrt(temp)))
    
scoresd['lasso'] = sum(scores)/len(scores)
print('CV(', kf.get_n_splits(), ') score: ', scoresd['lasso'])
import lightgbm as lgb

lgbmodel, scores= lgb.LGBMRegressor(), list()

for train, test in kf.split(x,y):
    x_train, y_train = x.loc[train], y.loc[train]
    x_test, y_test = x.loc[test], y.loc[test]

    lgbmodel.fit(x_train,y_train, early_stopping_rounds=100, verbose=False, eval_metric='mean_squared_error', eval_set=(x_test,y_test))
    y_pred = lgbmodel.predict(x_test)
    
    temp = mean_squared_error(y_pred,y_test)
    scores.append(100*max(0,1-(np.sqrt(temp))))

scoresd['baseline_lgb'] = sum(scores)/len(scores)
print('CV(', kf.get_n_splits(), ') score: ', scoresd['baseline_lgb'])
imp = pd.concat([pd.Series(lgbmodel.feature_importances_), pd.Series(list(x.columns))], keys = ['Value', 'Name'], axis=1)

plt.figure(figsize=(16,10))
sns.barplot(x='Value', y='Name', data= imp.sort_values(by='Value', ascending=False))
plt.title('Feature Importances for LGBoost')
plt.tight_layout()
from sklearn.model_selection import RandomizedSearchCV

learning_rate = [float(x) for x in np.linspace(0.005, 0.05, 100)]
n_estimators = [int(x) for x in range(500,5000,100)]
max_depth = [int(x) for x in range(2,30,2)]
num_leaves = [int(x) for x in range(1,5000,100)]
bagging_fraction = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

early_stopping_rounds = 50
min_data_in_leaf = 100
lambda_l1 = 0.5

lgbgrid = {'num_leaves': num_leaves,
           'bagging_fraction': bagging_fraction,
           'max_depth': max_depth,
           'n_estimators': n_estimators,
           'learning_rate': learning_rate
           }

lgbm = lgb.LGBMRegressor(min_data_in_leaf=min_data_in_leaf, lambda_l1=lambda_l1)
reg = RandomizedSearchCV(lgbm, lgbgrid, cv=5, n_iter=300, n_jobs=-1)
search = reg.fit(x,y)
search.best_params_
lgbmodelfixed, scores= lgb.LGBMRegressor(num_leaves=3801, n_estimators=600, max_depth=2, learning_rate=0.02363636363636364, bagging_fraction=0.6), list()


kf = KFold(n_splits=15,shuffle = True, random_state=100)

for train, test in kf.split(x_pca,y):
    x_train, y_train = x.loc[train], y.loc[train]
    x_test, y_test = x.loc[test], y.loc[test]

    lgbmodelfixed.fit(x_train,y_train, early_stopping_rounds=100, verbose=False, eval_metric='mean_squared_error', eval_set=(x_test,y_test))
    y_pred = lgbmodelfixed.predict(x_test)
    
    temp = mean_squared_error(y_pred,y_test)
    scores.append(100*max(0,1-(np.sqrt(temp))))

scoresd['pca_tuned_lgb'] = sum(scores)/len(scores)
print('CV(', kf.get_n_splits(), ') score: ', scoresd['pca_tuned_lgb'])
import xgboost as xgb

xgbmodel, scores = xgb.XGBRegressor(), list()

for train, test in kf.split(x,y):
    x_train, y_train = x.loc[train], y.loc[train]
    x_test, y_test = x.loc[test], y.loc[test]

    xgbmodel.fit(x_train,y_train, verbose=False)
    y_pred = xgbmodel.predict(x_test)

    temp = mean_squared_error(y_pred,y_test)
    scores.append(100*max(0,1-np.sqrt(temp)))
    
scoresd['baseline_XGB'] = sum(scores)/len(scores)
print('CV(', kf.get_n_splits(), ') score: ', scoresd['baseline_XGB'])
from catboost import CatBoostRegressor

cbr, scores = CatBoostRegressor(learning_rate=0.01), list()

for train, test in kf.split(x,y):
    x_train, y_train = x.loc[train], y.loc[train]
    x_test, y_test = x.loc[test], y.loc[test]

    cbr.fit(x_train,y_train, early_stopping_rounds=100, verbose=False)
    y_pred = cbr.predict(x_test)
    
    temp = mean_squared_error(y_pred,y_test)
    scores.append(100*max(0,1-(np.sqrt(temp))))

scoresd['catboost'] = sum(scores)/len(scores)
print('CV(', kf.get_n_splits(), ') score: ', scoresd['catboost'])
scoresd
lss.fit(x,y)
lgbmodelfixed.fit(x,y)

lgb_Y_pred = lgbmodelfixed.predict(X_test)
lasso_Y_pred = lss.predict(X_test)

Y_pred = 0.60*lgb_Y_pred + 0.40*lasso_Y_pred
Y_pred.shape
test = pd.read_csv(PATH+'Test.csv')
sample = pd.concat([test['Employee_ID'], pd.Series(np.round(Y_pred, 2))], keys = ('Employee_ID', target), axis=1)
# sample.to_csv(PATH+'blend.csv', index=False)