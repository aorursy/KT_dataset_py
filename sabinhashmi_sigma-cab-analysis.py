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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import random

import warnings

warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score,classification_report

le=LabelEncoder()

dt=DecisionTreeClassifier()

rf=RandomForestClassifier()

from sklearn.preprocessing import StandardScaler

ss=StandardScaler()
data=pd.read_csv('/kaggle/input/train.csv')
#Understanding Proportion of Null Values

data.isnull().mean()*100
del data['Var1']
data.plot(kind='box', layout=(3,3),subplots=True,figsize=(10,8))

plt.show()
cat_list=list(data.select_dtypes(exclude='object'))

for i in cat_list:

    q1=data[i].quantile(0.25)

    q3=data[i].quantile(0.75)

    iqr=q3-q1

    for j in range(0,len(data[i])):

        if data[i][j]>(q3+(1.5*iqr)) or data[i][j]<(q1-(1.5*iqr)):

            data[i][j]=data[i].mode()[0]
data.plot(kind='box', layout=(3,3),subplots=True,figsize=(10,8))

plt.show()
data.isnull().mean()*100
data['Type_of_Cab']=data['Type_of_Cab'].fillna(data['Type_of_Cab'].mode()[0])

data['Customer_Since_Months']=data['Customer_Since_Months'].fillna(round(data['Customer_Since_Months'].mean()))

data['Life_Style_Index']=data['Life_Style_Index'].fillna(data['Life_Style_Index'].mean())

data['Confidence_Life_Style_Index']=data['Confidence_Life_Style_Index'].fillna(data['Confidence_Life_Style_Index'].mode()[0])
data.plot(kind='box', layout=(3,3),subplots=True,figsize=(10,8))

plt.show()
cat_list=list(data.select_dtypes(exclude='object'))

for i in cat_list:

    q1=data[i].quantile(0.25)

    q3=data[i].quantile(0.75)

    iqr=q3-q1

    for j in range(0,len(data[i])):

        if data[i][j]>(q3+(1.5*iqr)) or data[i][j]<(q1-(1.5*iqr)):

            data[i][j]=data[i].mode()[0]
data.plot(kind='box', layout=(3,3),subplots=True,figsize=(10,8))

plt.show()
data_01=pd.concat([data.select_dtypes(include='object').apply(le.fit_transform),data.drop(data.select_dtypes(include='object'),axis=1)],axis=1)
x=data_01.drop('Surge_Pricing_Type',axis=1)

y=data_01['Surge_Pricing_Type']
x=ss.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=5)
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
accuracy_score(y_test,y_pred)
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
accuracy_score(y_test,y_pred)
params={'n_estimators':range(1,30)} # for Random Forest

gs_rf=GridSearchCV(rf,param_grid=params,return_train_score=True).fit(x_train,y_train)

y_pred=gs_rf.predict(x_test)

print('Accuracy Score is  ',accuracy_score(y_test,y_pred))

df_cv_result=pd.DataFrame(gs_rf.cv_results_)
df_cv_result.set_index('param_n_estimators')['mean_test_score'].plot.line()

df_cv_result.set_index('param_n_estimators')['mean_train_score'].plot.line()

plt.xlim(0,10)
params={'max_depth':range(1,20)} # for Decision Tree

gs_dt=GridSearchCV(dt,param_grid=params,return_train_score=True).fit(x_train,y_train)

y_pred=gs_dt.predict(x_test)

print('Accuracy Score is  ',accuracy_score(y_test,y_pred))

df_cv_result=pd.DataFrame(gs_dt.cv_results_)
df_cv_result.set_index('params')['mean_test_score'].plot.line()

df_cv_result.set_index('params')['mean_train_score'].plot.line()

plt.xlim(0,10)

plt.xticks(rotation=45)

gs_dt.best_estimator_