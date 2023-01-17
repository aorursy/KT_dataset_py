import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#from scipy.stats import norm
#from sklearn.preprocessing import StandardScaler
#from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

d_train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
d_train.head()

d_train['SalePrice'].describe()
sns.distplot(d_train['SalePrice']);
var = 'GrLivArea'
data = pd.concat([d_train['SalePrice'], d_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
data.head()
var = 'OverallQual'
data = pd.concat([d_train['SalePrice'], d_train[var]], axis=1)
data.head()
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
data.head()
#correlation matrix
corrmat = d_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(d_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#missing data
total = d_train.isnull().sum().sort_values(ascending=False)
percent = (d_train.isnull().sum()/d_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
# dealing with missing data
d_train = d_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
d_train = d_train.drop(d_train.loc[d_train['Electrical'].isnull()].index)
d_train.isnull().sum().max()
#histogram and normal probability plot
sns.distplot(d_train['SalePrice'], fit=norm);
fig = plt.figure()

# log transformation
d_train['SalePrice'] = np.log(d_train['SalePrice'])
#histogram and normal probability plot
sns.distplot(d_train['SalePrice'], fit=norm);
fig = plt.figure()

#create column for new  binary variable 
d_train['HasBsmt'] = pd.Series(len(d_train['TotalBsmtSF']), index=d_train.index)
d_train['HasBsmt'] = 0 
d_train.loc[d_train['TotalBsmtSF']>0,'HasBsmt'] = 1
#deleting points
d_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
d_train = d_train.drop(d_train[d_train['Id'] == 1299].index)
d_train = d_train.drop(d_train[d_train['Id'] == 524].index)
#convert categorical variable into dummy
d_train = pd.get_dummies(d_train)
d_train['SalePrice'].describe()
d_train.head()
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import  LassoCV
#Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)
X = d_train.loc[:, ['TotalBsmtSF', 'GrLivArea', 'FullBath', 'OverallQual', 'GarageCars', 'GarageArea', '1stFlrSF' ]]
y = d_train['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
model_lasso = LassoCV(alphas = [2,1,0.5, 0.1, 0.001, 0.0005]).fit(X_train, y_train)
y_predicted = model_lasso.predict(X=X_test)
mean_squared_error(y_test, y_predicted)
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_predicted, s=20)
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)])
plt.tight_layout()
test_subset = test.loc[:, ['TotalBsmtSF', 'GrLivArea', 'FullBath', 'OverallQual', 'GarageCars', 'GarageArea', '1stFlrSF']]
test_subset.fillna(0, inplace = True)
y_pred_test = model_lasso.predict(test_subset)
# Preparing submission
submission = pd.concat([test['Id'], pd.Series(np.exp(y_pred_test))], axis = 1)
submission.columns = ['Id','SalePrice']
submission.to_csv('sample_submission.csv',index=False)
