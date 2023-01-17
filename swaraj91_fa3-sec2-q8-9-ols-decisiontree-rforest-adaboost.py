# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import statsmodels.api as sm
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import scale
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
from sklearn import metrics 
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.
#load Dataset
df_res=pd.read_csv("../input/DC_Properties.csv")
#Getting the unknown PRICE into a different DataFrame for prediction
mask = df_res["PRICE"].isnull()
unknown = df_res[mask]
df = df_res[~mask]
#df.corr().style.background_gradient()


#df = df_res.loc[:, pd.notnull(df_res).sum()>len(df_res)*.7]
total = df.isnull().sum().sort_values(ascending=False)
#calculate % missing values in columns
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
#missing_data


#drop columns with more than 20% missing values
cols = missing_data[missing_data['Percent']<0.2].index.tolist()
df = df[cols]
#set '0' in AC column to N in training
df.AC.replace('0', 0, inplace=True)
df.AC.replace('Y', 1, inplace=True)
df.AC.replace('N', 0, inplace = True)
#change SALE date to sale year

#check again for null values in training set
df.isnull().sum()


#filling remaining NA values
df['Y'].fillna(method = 'ffill', inplace=True)
df['X'].fillna(method = 'ffill', inplace=True)
df['QUADRANT'].fillna(df['QUADRANT'].mode()[0], inplace=True)
df['AYB'].fillna(method = 'ffill', inplace=True)
df['SALEDATE'].fillna(method = 'ffill', inplace = True)
#convert last modified date and saledate to datetime
df['GIS_LAST_MOD_DTTM'] = pd.to_datetime(df['GIS_LAST_MOD_DTTM'])
df['SALEDATE'] = pd.to_datetime(df['SALEDATE'])
#Calculating the difference in years between Last Sale Date and Year Built
df['SalevYB']=df['SALEDATE'].dt.year - df['AYB']
#Calculating the difference in years between Last Sale Date and Year Improved
df['SalevYI']=df['SALEDATE'].dt.year - df['EYB']
#Calculating the difference in years between Last Sale Date and Year Last Modified Date
df['SalevLM']=df['SALEDATE'].dt.year - df['GIS_LAST_MOD_DTTM'].dt.year

#drop the columns from which the year differences have been calculated and which are not considered for regression
df.drop(['SALEDATE', 'EYB', 'AYB', 'Unnamed: 0', 'GIS_LAST_MOD_DTTM', 'BLDG_NUM', 'ZIPCODE', 'ASSESSMENT_NBHD'], axis = 1, inplace = True)
#remove rows containing SQUARE value as 'PAR '
df = df.drop(df[df.SQUARE == 'PAR '].index)
pd.to_numeric(df.SQUARE)
df[['SQUARE']] = df[['SQUARE']].apply(pd.to_numeric)

#create base df for reference
Xbase=df.drop(['PRICE'],axis=1)
ybase=df['PRICE']


#Creating dummy variables for categorical columns in Training
df=df.join(pd.get_dummies(df['HEAT'], drop_first=True));
df=df.join(pd.get_dummies(df['QUALIFIED'], drop_first=True));
df=df.join(pd.get_dummies(df['SOURCE'], drop_first=True));
df=df.join(pd.get_dummies(df['WARD'], drop_first=True));
df=df.join(pd.get_dummies(df['QUADRANT'], drop_first=True));
df=df.join(pd.get_dummies(df['AC'], drop_first=True));

#drop dates and replaced dummy columns
#train.drop(['Unnamed: 0', 'GIS_LAST_MOD_DTTM','SALEDATE', 'AC', 'HEAT', 'QUALIFIED', 'SOURCE', 'ASSESSMENT_NBHD', 'WARD', 'QUADRANT'],axis=1,inplace=True)
#drop columns for which dummy variable has been created
df.drop(['HEAT','QUALIFIED','SOURCE','WARD','QUADRANT','AC'], axis = 1, inplace = True)
df.reset_index(inplace = True, drop=True)

#dependent variable transformation
df['PRICE']=df['PRICE']**0.3
print(df.shape)
#remove outliers
df=df[(np.abs(stats.zscore(df)) < 6).all(axis=1)]
print(df.shape)
#Assign IVs to X and DV to y
X=df.drop(['PRICE'],axis=1)
y=df['PRICE']

sns.kdeplot(y)

#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler().fit(X)
#standardizedX = scaler.transform(X)
#Xw = standardizedX


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    print('Model Performance')
    print('Mean_Absolute_Error:', metrics.mean_absolute_error(y_test, predictions))
    print('Mean_Squared_Error:', metrics.mean_squared_error(y_test, predictions))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
#split into Test and Train
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 100)
model = sm.OLS(y_train, X_train.astype(float)).fit()
# Print out the statistics
#evaluate Accuracy
evaluate(model, X_test, y_test)
model.summary()


#Drop insignificant variables and rerun the ols model
X_train_ols = X_train.drop(['Y', 'X', 'Forced Air', 'Hot Water Rad', 'Ht Pump', 'Warm Cool', 'SE'], axis =1)
X_test_ols = X_test.drop(['Y', 'X', 'Forced Air', 'Hot Water Rad', 'Ht Pump', 'Warm Cool', 'SE'], axis = 1)
model = sm.OLS(y_train, X_train_ols.astype(float)).fit()
# Print out the statistics
#evaluate Accuracy
evaluate(model, X_test_ols, y_test)
model.summary()
from sklearn.tree import DecisionTreeRegressor
#regressor = DecisionTreeRegressor(max_depth=15)
param_dist = {'max_depth': [1,20]}
#regressor.fit(X_train, y_train)
#regressor.score(X_train, y_train)
DTR = DecisionTreeRegressor()
DTR_cv = RandomizedSearchCV(DTR, param_dist)
DTR_cv.fit(X_train, y_train)
evaluate(DTR_cv.best_estimator_, X_test, y_test)
from sklearn.ensemble import RandomForestRegressor
from pprint import pprint

# Create the parameter distribution
param_dist = {'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 500, num = 10)],
               'max_features': ['auto', 'sqrt'],
               'max_depth': [1, 20]}

RFR = RandomForestRegressor()
RFR_cv = RandomizedSearchCV(RFR, param_dist)
RFR_cv.fit(X_train, y_train)
evaluate(RFR_cv.best_estimator_, X_test, y_test)
RFR_cv.best_estimator_
#Adaptive Boosting Decision Tree
rng = np.random.RandomState(1)
param_dist = {'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 400, num = 10)]}


ada = AdaBoostRegressor(DecisionTreeRegressor(random_state = rng), random_state = rng)
ada_cv = RandomizedSearchCV(ada, param_dist)
ada_cv.fit(X_train, y_train)

evaluate(ada_cv.best_estimator_, X_test, y_test)
#Obtaining the best estimators from the model
print(ada_cv.best_estimator_)