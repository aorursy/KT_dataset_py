# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import statsmodels.api as sm
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import scale
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.cross_validation import train_test_split
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn import metrics 
print(os.listdir("../input"))
from sklearn.linear_model import Ridge

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
#Question 9 RIDGE REGRESSION
import warnings
warnings.filterwarnings("ignore")
# Split data into training and test sets
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
MSE=[]
xvalues=[]
for i in range (0,500):
    b=random.randint(1, 100000)
    xvalues.append(b*.0001)
    ridge1 = Ridge(alpha =b*.0001, normalize = True)
    ridge1.fit(X_train, y_train)             # Fit a ridge regression on the training data
    pred2 = ridge1.predict(X_test)           # Use this model to predict the test data
    #print(pd.Series(ridge2.coef_, index = X.columns)) # Print coefficients
    # Calculate the test MSE
    #print(b*.0001,mean_squared_error(y_test, pred2),'yes')
    MSE.append(mean_squared_error(y_test, pred2))
# with increase in shrinkage paramter , model is having a high MSE , this implies that ridge regression is not a suitable method

plt.scatter(xvalues,MSE)
plt.xlabel('Alpha for ridge regression')
plt.ylabel('MSE for ridge regression')
plt.title('MSE Vs Alpha RIDGE REGRESSION')
# Split data into training and test sets
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
MSE=[]
xvalues=[]
import warnings
warnings.filterwarnings("ignore")
for i in range (0,100):
    b=random.randint(1,100000)
    xvalues.append(b*.0001)
    lasso1 = Lasso(alpha =b*.0001)
    lasso1.fit(X_train, y_train)             # Fit a ridge regression on the training data
    pred2 = lasso1.predict(X_test)           # Use this model to predict the test data
    #print(pd.Series(ridge2.coef_, index = X.columns)) # Print coefficients
    # Calculate the test MSE
    #print(b*.0001,mean_squared_error(y_test, pred2),'yes')
    MSE.append(mean_squared_error(y_test, pred2))
# with increase in shrinkage paramter , model is having a high MSE , this implies that LASSO regression is not a suitable method

plt.scatter(xvalues,MSE)
plt.xlabel('Alpha for LASSO regression')
plt.ylabel('MSE for LASSO regression')
plt.title('MSE Vs Alpha LASSO  REGRESSION')
