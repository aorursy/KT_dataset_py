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


#df.select_dtypes(include=['object']).dtypes
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
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler().fit(X)
#standardizedX = scaler.transform(X)
#Xw = standardizedX

from sklearn.neighbors import KNeighborsRegressor


param_dist = {"n_neighbors": [3, 7],
              "weights": ["uniform", "distance"]}

KNR = KNeighborsRegressor()

KNR_cv = RandomizedSearchCV(KNR, param_dist, cv=5)

KNR_cv.fit(X_train, y_train)
#KNR.score(X_train, y_train)

evaluate(KNR_cv.best_estimator_, X_test, y_test)
KNR_cv.best_estimator_
#from sklearn.svm import SVR


#param_dist = {"kernel": ["rbf", "linear", "poly"],
#              "C": [1, 20]}

#svr = SVR()
#svr_cv = RandomizedSearchCV(svr, param_dist, cv=5)
#svr_cv.fit(X_train, y_train)
#svr.fit(X_train, y_train)
#evaluate(svr_cv.best_estimator_, X_test, y_test)
#evaluate(svr, X_test, y_test)
#from sklearn.svm import SVR
#svr = SVR(kernel='linear')
#svr.fit(X_train, y_train)
#accuracy = evaluate(svr, X_test, y_test)
from scipy.stats import uniform
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


param_grid = {'alpha': uniform()}
seed=6
model = Ridge()
kfold = KFold(n_splits=8, random_state=seed, shuffle=True)
rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100, cv=kfold, random_state=seed)
rsearch.fit(X, y)
print(rsearch.best_score_)
print(rsearch.best_estimator_.alpha)

evaluate(rsearch, X_test, y_test)
rsearch.best_estimator_