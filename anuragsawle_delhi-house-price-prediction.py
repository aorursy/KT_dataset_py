import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = pd.read_csv('/kaggle/input/delhi-house-price-prediction/MagicBricks.csv')
dataset.head()
dataset.describe()
sns.barplot(x=dataset['BHK'],y=dataset['Price'])
#As we can see the price is dependent on number of BHK in house
sns.barplot(x='Furnishing',y='Price',data=dataset)
#Furnishing doesn't make any big change in price of house
sns.catplot(y='Price',x='Furnishing',data=dataset,hue='BHK')
sns.catplot(y='Price',x='Type',data=dataset)
sns.regplot(y='Area',x='Price',data=dataset)
#In following graph we can se this that price is depend on area 
dataset.isna().sum()
#Our dataset contain lots of null values and we need to clean our data and fill that null values and Per_Sqft contain lots of null values 
#so we can drop this column
le = LabelEncoder()
dataset['Locality'] = le.fit_transform(dataset['Locality'])
sns.regplot(x='Locality',y='Price',data=dataset)
#Locality doesn't make any big change in price so we can drop this column 
X_set = dataset.drop(columns='Per_Sqft',axis=1)
X_set.drop('Locality',inplace=True,axis=1)
X_set.isna().sum()
#now we need to fill null values
X_set['Parking'].fillna(X_set['Parking'].median(),inplace=True)
X_set['Bathroom'].fillna(X_set['Bathroom'].median(),inplace=True)
X_set['Type'].fillna(X_set['Type'].value_counts().idxmax(),inplace=True)
X_set['Furnishing'].fillna(X_set['Furnishing'].value_counts().idxmax(),inplace=True)
X_set.isna().sum()
#now our dataset has zero null value
dataset['Status'].describe()
#as we can see in output that our Status column contain 2 unique value and freq of one value is much higher than other values
#so we can drop this column
X_set.drop(columns='Status',inplace=True)
dummy_X=pd.get_dummies(X_set['Type'])
dummy_Fur=pd.get_dummies(X_set['Furnishing'])
dummy_tra=pd.get_dummies(X_set['Transaction'])
X_set= pd.concat([X_set,dummy_X],axis=1,join='outer')
X_set= pd.concat([X_set,dummy_Fur],axis=1,join='outer')
X_set= pd.concat([X_set,dummy_tra],axis=1,join='outer')
X_set=X_set.drop(['Type'],axis=1)
X_set=X_set.drop(['Furnishing'],axis=1)
X_set=X_set.drop(['Transaction'],axis=1)
#We need to calculate price so we need to drop this price column from our feature dataset
X_set.drop(columns='Price',axis=1,inplace=True)
X_set
#Y_set is our label set
Y_set=dataset['Price']
Y_set
#We our data has variation in diffent columns we need to fit in range for better accuracy
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
X_setn = scaler.fit_transform(X_set)
X_set=pd.DataFrame(X_setn)
X_set
from sklearn.model_selection import train_test_split
X_training,X_test,Y_training,Y_test= train_test_split(X_set,Y_set,test_size=0.25)
score=[]
algo=[]
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
lr_model = LinearRegression()
lr_model.fit(X_training,Y_training)
lr_ypred=lr_model.predict(X_test)
lr_score=metrics.r2_score(Y_test,lr_ypred)
print(lr_score)
score.append(lr_score*100)
algo.append('Linear Reg.')
dtr_model = DecisionTreeRegressor()
dtr_model.fit(X_training,Y_training)
dtr_ypred=dtr_model.predict(X_test)
dtr_score =metrics.r2_score(Y_test,dtr_ypred)
print(dtr_score)
score.append(dtr_score*100)
algo.append('Decision Tree')
rfr_model = RandomForestRegressor()
rfr_model.fit(X_training,Y_training)
rfr_ypred=rfr_model.predict(X_test)
rfr_score =metrics.r2_score(Y_test,rfr_ypred)
print(rfr_score)
score.append(rfr_score*100)
algo.append('Random Forest')
knr_model = KNeighborsRegressor(n_neighbors=3)
knr_model.fit(X_training,Y_training)
knr_ypred = knr_model.predict(X_test)
knr_score =metrics.r2_score(Y_test,knr_ypred)
print(knr_score)
score.append(knr_score*100)
algo.append('KNR')
from sklearn.metrics import mean_squared_error
from math import sqrt
sqrt(mean_squared_error(Y_test,knr_ypred))
error=[]
for k in range(1,10):
    knr=KNeighborsRegressor(n_neighbors=k)
    knr.fit(X_training,Y_training)
    pred=knr.predict(X_test)
    error.append(sqrt(mean_squared_error(Y_test,pred)))
my_plot=pd.DataFrame(error)
%matplotlib inline
my_plot.plot()
sns.pointplot(x=algo,y=score)