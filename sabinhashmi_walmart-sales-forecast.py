# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Loading Packages

import pandas as np

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.impute import SimpleImputer

si=SimpleImputer()

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

from scipy.stats import zscore

from time import time

from sklearn.model_selection import GridSearchCV

#Loading Package for Models

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,BaggingRegressor,ExtraTreesRegressor

from sklearn.metrics import r2_score

lr=LinearRegression()

dt=DecisionTreeRegressor()

svm=SVR()

knn=KNeighborsRegressor()

rf=RandomForestRegressor()

ada=AdaBoostRegressor()

bag=BaggingRegressor()

xtree=ExtraTreesRegressor()
#Calling the files

train=pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip')

features=pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv.zip')

stores=pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv')
#Merging the required tables.

data=pd.merge(train,features,on=['Store','Date','IsHoliday'],how='inner')

data=data.merge(stores)
#To have an outline about the data.

data.head()
#To check the percentage of null values in each attributes.

data.isnull().mean()*100
#Converting all the non-numeric(objects,datetime,etc)

data['Day']=pd.to_datetime(data['Date']).dt.day

data['Month']=pd.to_datetime(data['Date']).dt.month

data['Year']=pd.to_datetime(data['Date']).dt.year

del data['Date']

data['IsHoliday']=le.fit_transform(data['IsHoliday'])

data['Type']=le.fit_transform(data['Type'])
#Applying Simple Imputor Function on Mark-Down Columns and concatinating with the actual dataframe

data=pd.concat([data.drop(['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'],axis=1),

                pd.DataFrame(si.fit_transform(data[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']]),

                             columns=['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'])],axis=1)
#This helps to see the complete change in the new data frame

data.head()
#Outlier Search: This helps you to get some insights about the outliers in the data.

data.plot(kind='box',layout=(3,6),subplots=1,figsize=(20,16))

plt.show()
#Independent and Dependent Variable Segregation

x=data.drop('Weekly_Sales',axis=1)

y=data['Weekly_Sales']
#Scaling the data to a unified nature.

x=x.apply(zscore)
#Splitting the data into four different sections for training and testing of the data

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=5)
start_time=time()

model_list=[lr,dt,knn,rf,ada,bag,xtree]

Score=[]

for i in model_list:

    i.fit(x_train,y_train)

    y_pred=i.predict(x_test)

    score=r2_score(y_test,y_pred)

    Score.append(score)

print(pd.DataFrame(zip(model_list,Score),columns=['Model Used','R2-Score']))

end_time=time()

print(round(end_time-start_time,2),'sec')
#better accuracy for Random Forest.

param={'n_estimators':range(1,10)}

gridsearch=GridSearchCV(rf,param_grid=param,return_train_score=True)

gridsearch.fit(x_train,y_train)
pd.DataFrame(gridsearch.cv_results_).set_index('params')['mean_test_score'].plot.line()

pd.DataFrame(gridsearch.cv_results_).set_index('params')['mean_train_score'].plot.line()

plt.xticks(rotation=45)