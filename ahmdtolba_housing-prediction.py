import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_train
df_train.info()
df_train.describe()
sns.distplot(df_train['SalePrice'])
df_train.corr()
#correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
print(df_train[cols])
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#scatter plot grlivarea/saleprice
var = 'GrLivArea'
df_train.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
var = 'YearBuilt'
df_train.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
var = 'OverallQual'
df_train.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
var = 'GarageCars'
df_train.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

#dealing with missing data
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train.isnull().sum().max() #just checking that there's no missing data missing.
df_train
y = df_train['SalePrice']
df_train = df_train.drop('SalePrice' , axis = 1)
df_train
df_train.drop('Id' , axis = 1)
df_train = pd.get_dummies(df_train)
X = df_train
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X = sc_X.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,  shuffle=True, random_state=42)

# Import Libraries
from sklearn.tree import DecisionTreeRegressor
#----------------------------------------------------

#----------------------------------------------------
#Applying DecisionTreeRegressor Model 


DecisionTreeRegressorModel = DecisionTreeRegressor( max_depth=3,random_state=33)
DecisionTreeRegressorModel.fit(X_train, y_train)

#Calculating Details
print('DecisionTreeRegressor Train Score is : ' , DecisionTreeRegressorModel.score(X_train, y_train))
print('DecisionTreeRegressor Test Score is : ' , DecisionTreeRegressorModel.score(X_test, y_test))
#print('----------------------------------------------------')

#Calculating Prediction
#y_pred = DecisionTreeRegressorModel.predict(X_test)
#print('Predicted Value for DecisionTreeRegressorModel is : ' , y_pred[:10])
#Import Libraries
from sklearn.neighbors import KNeighborsRegressor
#----------------------------------------------------

#----------------------------------------------------
#Applying KNeighborsRegressor Model 



KNeighborsRegressorModel = KNeighborsRegressor(n_neighbors = 5, weights='uniform', #also can be : distance, or defined function 
                                               algorithm = 'auto')    #also can be : ball_tree ,  kd_tree  , brute
KNeighborsRegressorModel.fit(X_train, y_train)

#Calculating Details
print('KNeighborsRegressorModel Train Score is : ' , KNeighborsRegressorModel.score(X_train, y_train))
print('KNeighborsRegressorModel Test Score is : ' , KNeighborsRegressorModel.score(X_test, y_test))
#print('----------------------------------------------------')

#Calculating Prediction
#y_pred = KNeighborsRegressorModel.predict(X_test)
#print('Predicted Value for KNeighborsRegressorModel is : ' , y_pred[:10])
#Import Libraries
from sklearn.ensemble import RandomForestRegressor
#----------------------------------------------------

#----------------------------------------------------
#Applying Random Forest Regressor Model 


RandomForestRegressorModel = RandomForestRegressor(n_estimators=100,max_depth=2, random_state=33)
RandomForestRegressorModel.fit(X_train, y_train)

#Calculating Details
print('Random Forest Regressor Train Score is : ' , RandomForestRegressorModel.score(X_train, y_train))
print('Random Forest Regressor Test Score is : ' , RandomForestRegressorModel.score(X_test, y_test))
#print('Random Forest Regressor No. of features are : ' , RandomForestRegressorModel.n_features_)
#print('----------------------------------------------------')

#Calculating Prediction
#y_pred = RandomForestRegressorModel.predict(X_test)
#print('Predicted Value for Random Forest Regressor is : ' , y_pred[:10])
#Import Libraries
from sklearn.linear_model import Lasso
#----------------------------------------------------

#----------------------------------------------------
#Applying Lasso Regression Model 

'''
sklearn.linear_model.Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=
                           False, copy_X=True, max_iter=1000, tol=0.0001,
                           warm_start=False, positive=False, random_state=None,selection='cyclic')
'''

LassoRegressionModel = Lasso(alpha=1.0,random_state=33,normalize=False)
LassoRegressionModel.fit(X_train, y_train)

#Calculating Details
print('Lasso Regression Train Score is : ' , LassoRegressionModel.score(X_train, y_train))
print('Lasso Regression Test Score is : ' , LassoRegressionModel.score(X_test, y_test))
#print('Lasso Regression Coef is : ' , LassoRegressionModel.coef_)
#print('Lasso Regression intercept is : ' , LassoRegressionModel.intercept_)
#print('----------------------------------------------------')

#Calculating Prediction
#y_pred = LassoRegressionModel.predict(X_test)
#print('Predicted Value for Lasso Regression is : ' , y_pred[:10])
models = pd.DataFrame({
    'Model': [ 'DecisionTreeRegressor',
              'KNeighborsRegressor',
              'RandomForestRegressor',
              'LassoRegressionModel'],
    'Score': [DecisionTreeRegressorModel.score(X_test, y_test) 
              ,KNeighborsRegressorModel.score(X_test, y_test)
              , RandomForestRegressorModel.score(X_test, y_test)
             ,LassoRegressionModel.score(X_test, y_test)]})
models.sort_values(by='Score', ascending=False)