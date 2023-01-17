import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib　inline
df_train=pd.read_csv('../input/train.csv')
df_test=pd.read_csv('../input/test.csv')
combine = [df_train, df_test]
# check the columns
df_train.columns
# preview the data
df_train.head()
df_test.head()
df_train['SalePrice'].describe()
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)*100
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
#  percentage of missing values for objective variables of non-object type
train_obj=df_train.ix[:, df_train.dtypes != np.object]
total = train_obj.isnull().sum().sort_values(ascending=False)
percent = (train_obj.isnull().sum()/train_obj.isnull().count()).sort_values(ascending=False)*100
missing_data1 = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data1.head()
#  percentage of missing values for objective variables of non-object type
test_obj=df_test.ix[:, df_test.dtypes != np.object]
total = test_obj.isnull().sum().sort_values(ascending=False)
percent = (test_obj.isnull().sum()/test_obj.isnull().count()).sort_values(ascending=False)*100
missing_data2 = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data2.head(12)
df_train = df_train.drop((['LotFrontage','GarageYrBlt']), axis=1)
df_test = df_test.drop((['LotFrontage','GarageYrBlt']), axis=1) 
df_train['MasVnrArea']= df_train['MasVnrArea'].fillna(df_train['MasVnrArea'].mean())
df_test['MasVnrArea']= df_test['MasVnrArea'].fillna(df_test['MasVnrArea'].mean())
df_test['BsmtHalfBath']= df_test['BsmtHalfBath'].fillna(df_test['BsmtHalfBath'].mean())
df_test['BsmtFullBath']= df_test['BsmtFullBath'].fillna(df_test['BsmtFullBath'].mean())
df_test['GarageArea']= df_test['GarageArea'].fillna(df_test['GarageArea'].mean())
df_test['BsmtFinSF1']= df_test['BsmtFinSF1'].fillna(df_test['BsmtFinSF1'].mean())
df_test['BsmtFinSF2']= df_test['BsmtFinSF2'].fillna(df_test['BsmtFinSF2'].mean())
df_test['BsmtUnfSF']= df_test['BsmtUnfSF'].fillna(df_test['BsmtUnfSF'].mean())
df_test['TotalBsmtSF']= df_test['TotalBsmtSF'].fillna(df_test['TotalBsmtSF'].mean())
df_test['GarageCars']= df_test['GarageCars'].fillna(df_test['GarageCars'].mean())
#correlation matrix
df=df_train.drop(['Id'],axis=1).copy()
corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
df_train[['HouseStyle', 'SalePrice']].groupby(['HouseStyle'], as_index=False).mean().sort_values(by='SalePrice', ascending=False)
title_mapping={"2.5Fin":8,"2Story":7,"1Story":6,"SLvl":5,"2.5Unf":4,"1.5Fin":3,"SFoyer":2,"1.5Unf":1}
for dataset in [df_train,df_test]:
    dataset['HouseStyle']=dataset['HouseStyle'].map(title_mapping)
df_train[['HeatingQC', 'SalePrice']].groupby(['HeatingQC'], as_index=False).mean().sort_values(by='SalePrice', ascending=False)

title_mapping={"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1}
for dataset in [df_train,df_test]:
    dataset['HeatingQC']=dataset['HeatingQC'].map(title_mapping)
df_train = df_train.loc[:, df_train.dtypes != 'object']
df_test = df_test.loc[:, df_test.dtypes != 'object']
#saleprice correlation matrix
k = 10  #number of variables for heatmap
corrmat = df_train.corr()
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
f,ax=plt.subplots(figsize=(12,9))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 12}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#scatter plots
sns.set()
sns.pairplot(df_train[cols], size = 3)
plt.show();
#　bivariate analysis saleprice/grlivarea
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
#　bivariate analysis saleprice/grlivarea
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
k = 10  
corrmat = df_train.corr()
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
f,ax=plt.subplots(figsize=(12,9))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 12}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_squared_error

X=pd.DataFrame(df_train,columns=df_train.columns)
y = pd.DataFrame(df_train['SalePrice'])

X=X.loc[:,['OverallQual', 'GrLivArea','TotalBsmtSF','GarageCars','1stFlrSF']]
y= y.values
y_pred = X.values
RMSE = np.sqrt(np.mean((y-y_pred)**2))
RMSE
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import PolynomialFeatures
degree_2 = PolynomialFeatures(degree=2)
degree_3 = PolynomialFeatures(degree=3)
degree_4 = PolynomialFeatures(degree=4)

x_train_2 = degree_2.fit_transform(X_train)
x_train_3 = degree_3.fit_transform(X_train)
x_train_4 = degree_4.fit_transform(X_train)

lin_2d = LinearRegression()
lin_3d = LinearRegression()
lin_4d = LinearRegression()

lin_2d.fit(x_train_2,y_train)
lin_3d.fit(x_train_3,y_train)
lin_4d.fit(x_train_4,y_train)

x_test_2 = degree_2.fit_transform(X_test)
x_test_3 = degree_3.fit_transform(X_test)
x_test_4 = degree_4.fit_transform(X_test)

score_2d = lin_2d.score(x_test_2,y_test)
score_3d = lin_3d.score(x_test_3,y_test)
score_4d = lin_4d.score(x_test_4,y_test)

print("The coefficient of determination in the quadratic equation is %.2f"%(score_2d))
print("The coefficient of determination in the cubic equation is %.2f"%(score_3d))
print("The coefficient of determination in the quartic equation is %.2f"%(score_4d))
X=pd.DataFrame(df_test,columns=df_test.columns)

X_2=X.copy()
X=X.loc[:,['OverallQual', 'GrLivArea','TotalBsmtSF','GarageCars','1stFlrSF']]

from sklearn.preprocessing import PolynomialFeatures
degree_2 = PolynomialFeatures(degree=2)
degree_3 = PolynomialFeatures(degree=3)
degree_4 = PolynomialFeatures(degree=4)

x_test_2 = degree_2.fit_transform(X)
x_test_3 = degree_3.fit_transform(X)
x_test_4 = degree_4.fit_transform(X)

y_pred=lin_2d.predict(x_test_2)
y_pred
submission = pd.DataFrame(lin_2d.predict(x_test_2))
submission.columns=['SalePrice']
submission=pd.concat([X_2['Id'],submission['SalePrice']],axis=1)
#submission.to_csv('../input/submission.csv', index=False)
