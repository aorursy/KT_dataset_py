# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import (LinearRegression, Ridge, Lasso)
from sklearn.preprocessing import MinMaxScaler
#from sklearn.cross_validation import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/housesalesprediction/kc_house_data.csv")
df.head()
df.shape
#so there are 21 columns and 21613 rows

df.info()
df.date=pd.to_datetime(df.date)
df.date.head()
df.describe()
#to drop any duplicates if any exist
df.drop_duplicates(keep='first',inplace=True)
help(df.drop_duplicates)
df.shape
#if there is any na values drop them there are no na values as seen before
df.dropna(axis=0,inplace=True)
df.shape
#lets have a look at our variables 
df.columns
df.drop(['id','zipcode','lat','long'],inplace=True,axis=1)
import seaborn as sns
sns.pairplot(df)
#to visualise the correlation heatmap is the best method lets see it
plt.figure(figsize=(20,8))
sns.heatmap(df.corr(),annot=True) #here it is necessary to use annot=True to get the correlation values printed on map
plt.show()
df.corr()
#lets add features which have the little high correlation values 
cols=['sqft_living15','sqft_above','sqft_living15','sqft_lot15',]
df_fil=df[cols]
df_fil.head()
#describe function is used to check for any outliers or to observe pattern among values and different percentile values
df_fil.describe()
help(df.describe)
sns.pairplot(x_vars=['bathrooms','sqft_living','grade','sqft_above','sqft_living15'],y_vars='price',data=df_fil)
df_fil.corr()
df_fil.drop('sqft_living',axis=1,inplace=True)
df_fil.columns
df_fil.head()
#lets divide our data into independent variables X and dependent variable 
X=df_fil.copy()
X.shape
X.head()
y=df.price
from sklearn.model_selection import train_test_split
#Let's divide our data into test and train parts
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=0)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train[['bathrooms', 'grade', 'sqft_above', 'sqft_living15']] = scaler.fit_transform(X_train[['bathrooms', 'grade', 'sqft_above', 'sqft_living15']])

X_train.head()
X_train_sm=sm.add_constant(X_train)
lr=sm.OLS(y_train,X_train_sm)
lr=lr.fit()
lr.params
lr.summary()
from sklearn.metrics import r2_score
r2_score(X_train,y_train)
y_train_pred=lr.predict(X_train)
r2_score(y_train,y_train_pred)
y_test_pred=lr.predict(X_test)
y_test_pred.head()
help(r2_score)
lr_new.Summary()
X=df_fil['bathrooms','grade','sqft_above','sqft_living15'].copy()
print(X.columns)
print(X.head(),y.head())
#this function gives us the correlation values
df.corr()
# Inspecting type
print(df.dtypes)
df.columns

df=df.drop(['date', 'id'], axis=1)
df.columns
#to see the columns dropped
with sns.plotting_context("notebook",font_scale=2.5):
    g = sns.pairplot(df[['sqft_lot','sqft_above','price','sqft_living','bedrooms']], 
                 hue='bedrooms', palette='tab20',size=6)
g.set(xticklabels=[]);
str_list = [] # empty list to contain columns with strings (words)
for colname, colvalue in df.iteritems():
    if type(colvalue[1]) == str:
         str_list.append(colname)
# Get to the numeric columns by inversion            
num_list = df.columns.difference(str_list) 
# Create Dataframe containing only numerical features
df_num = df[num_list]
f, ax = plt.subplots(figsize=(16, 12))
plt.title('Pearson Correlation of features')
# Draw the heatmap using seaborn
#sns.heatmap(house_num.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="PuBuGn", linecolor='k', annot=True)
sns.heatmap(df_num.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="cubehelix", linecolor='k', annot=True)
from sklearn.model_selection import train_test_split
X = df.iloc[:,1:].values
y = df.iloc[:,0].values
#splitting dataset into training and testing dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
print(y_pred)