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

import datetime

import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objects as go

import plotly.graph_objs as go

from plotly.subplots import make_subplots

from sklearn.metrics import r2_score

import warnings

warnings.warn = False
train = pd.read_csv('../input/restaurant-revenue-prediction/train.csv')

test  = pd.read_csv('../input/restaurant-revenue-prediction/test.csv')



print("Training set = ",train.shape)

print("Testing set = ",test.shape)

print("Sum of Missing Values (Train/Test)= ",train.isna().sum().sum(),"(",test.isna().sum().sum(),")")
train.head()
quater_id=[1,1,1,1,2,2,2,3,3,3,4,4,4]



# Working with Training Data

train['year']  = pd.DatetimeIndex(train['Open Date']).year

train['month'] = pd.DatetimeIndex(train['Open Date']).month

train['day']   = pd.DatetimeIndex(train['Open Date']).day

train = train.drop(columns=['Open Date'])



train['quater']=[quater_id[i] for i in train.month]

train['first_week'] = train.day.apply(lambda x: 1 if x<= 7 else 0)

train['last_week']  = train.day.apply(lambda x: 1 if x>=21 else 0)

train['rev_slot'] = pd.qcut(train['revenue'], 20, labels=False)





# Working with Testing Data

test['year']  = pd.DatetimeIndex(test['Open Date']).year

test['month'] = pd.DatetimeIndex(test['Open Date']).month

test['day']   = pd.DatetimeIndex(test['Open Date']).day

test = test.drop(columns=['Open Date'])



test['quater']=[quater_id[i] for i in test.month]

test['first_week'] = test.day.apply(lambda x: 1 if x<= 7 else 0)

test['last_week']  = test.day.apply(lambda x: 1 if x>=21 else 0)





# Display changes in Training Data

train.head()
def define_weekdays(data):

   

  data['day_value'] = pd.DataFrame([datetime.date(data.year[i],data.month[i],data.day[i]).weekday() for i in range(data.shape[0])])

  data['weekdays'] = np.where(data.day_value<5, 1,0)

  data['weekend'] = 1-data['weekdays']

  return data



train = define_weekdays(train) 

test  = define_weekdays(test) 



train.head()
def get_feature_count(data):

  col_names = ['City','City Group','Type','year','quater','month','first_week','last_week','day','weekdays','weekend','rev_slot']

  df_all=pd.DataFrame()

  for i in col_names:

    u = data[i].unique()

    temp=pd.DataFrame()

    for j in u:

      m = (data[i]==j).sum()

      temp = temp.append([[j,m]])

    temp['col_name'] = i    

    df_all = df_all.append(temp)



  df_all.columns = ['X','Y','Feature']

  return df_all



df = get_feature_count(train)



fig=px.bar(data_frame=df, x='X',y='Y',color='Y',facet_col='Feature',facet_col_wrap=4,height=600)

fig.update_xaxes(matches=None)

fig.update_yaxes(matches=None)

fig.show()
def get_feature_count(data):

  col_names = ['City','City Group','Type','year','quater','month','first_week','last_week','day','weekdays','weekend']

  df_all=pd.DataFrame()

  for i in col_names:

    u = data[i].unique()

    temp=pd.DataFrame()

    for j in u:

      m = data.revenue[data[i]==j].mean()

      temp = temp.append([[j,m]])

    temp['col_name'] = i

    df_all = df_all.append(temp)



  df_all.columns = ['X','Y','Feature']

  return df_all



df = get_feature_count(train)





fig=px.bar(data_frame=df, x='X',y='Y',color='Y',facet_col='Feature',facet_col_wrap=4,height=600)

fig.update_xaxes(matches=None)

fig.update_yaxes(matches=None)

fig.show()
# One-Hot Encoding

print("Prior to One-Hot Encoding",train.shape," : ",test.shape)

OHE_cols = ['City','City Group','Type','weekdays']

train = pd.get_dummies(train, columns = OHE_cols, drop_first = True)

test  = pd.get_dummies(test,  columns = OHE_cols, drop_first = True)

print("Post to One-Hot Encoding",train.shape," : ",test.shape)
# Train-Test Dataset Ready

y_train = train.revenue

X_train = train.drop(['Id','revenue','rev_slot'],axis=1)

test    = test.drop(['Id'],axis=1)



print(train.shape,test.shape) 
# Remove columns which exists in Training data but is missing in Testing Data or Vice-Versa



def remove_columns(columns_a,columns_b):    

  col_list = []

  for i in columns_a:

    flag=0

    for j in columns_b:

      if i==j:              

        flag=1

        break

    if flag==0:            

      col_list.append(i)              

  return col_list

    

col_list_1 = remove_columns(X_train.columns,test.columns)

col_list_2 = remove_columns(test.columns,X_train.columns)



X_train = X_train.drop(columns=col_list_1,axis=1)

test = test.drop(columns=col_list_2,axis=1)



print("Post dropping columns from Training Data",X_train.shape)

print("Post dropping columns from Testing Data",test.shape) 
from sklearn.linear_model import LinearRegression

lr = LinearRegression(fit_intercept=True,normalize=False)

lr.fit(X_train,y_train)

y_pred1 = lr.predict(X_train)



print("Training performance from Linear Regression = ",abs(r2_score(y_train,y_pred1)))
# Cannot use Lasso or Elastic Net: Variables are highly correlated, so LASSO may eliminate wrongly



from sklearn.linear_model import Ridge

rd = Ridge(alpha=1.0, max_iter=100, tol=0.0001, random_state=100)

rd.fit(X_train,y_train)

y_pred2 = rd.predict(X_train)



print("Training performance from Ridge Regression = ",abs(r2_score(y_train,y_pred2)))
y_pred = (0.7*y_pred1+0.3*y_pred2)

print("Training performance from Linear and Ridge Regression = ",abs(r2_score(y_train,y_pred)))

y_pred = (0.5*y_pred1+0.5*y_pred2)

print("Training performance from Linear and Ridge Regression = ",abs(r2_score(y_train,y_pred)))

y_pred = (0.3*y_pred1+0.7*y_pred2)

print("Training performance from Linear and Ridge Regression = ",abs(r2_score(y_train,y_pred)))



# Performance will be obtained in-between Linear and Ridge Regression. 

# Thus, one or more models must be considered for taking an average or perform second stage regression.