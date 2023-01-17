# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression



from sklearn import svm

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

    

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score



from sklearn.preprocessing import PolynomialFeatures

from sklearn import preprocessing



import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')

test = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')

sample_submission = pd.read_csv('../input/covid19-global-forecasting-week-5/submission.csv')
train.head()

train.info()
#country vs targetValue



fig = px.pie(train, values='TargetValue', names='Country_Region')



fig.show()
test.head()
test.info()
#Preprocessing

train = train.drop(columns = ['County' , 'Province_State'])

test = test.drop(columns = ['County' , 'Province_State'])
train.info()
test.info()
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

X = train.iloc[:,1].values

train.iloc[:,1] = labelencoder.fit_transform(X.astype(str))



X = train.iloc[:,5].values

train.iloc[:,5] = labelencoder.fit_transform(X)

train.head()

X = test.iloc[:,1].values

test.iloc[:,1] = labelencoder.fit_transform(X)



X = test.iloc[:,5].values

test.iloc[:,5] = labelencoder.fit_transform(X)
train['Date'] = pd.to_datetime(train['Date'])

test['Date'] = pd.to_datetime(test['Date'])

test.head()
train.head()
train.info()
def date_feature(df):

  df['day'] = df['Date'].dt.day

  df['month'] = df['Date'].dt.month

  return df
train = date_feature(train)

test = date_feature(test)

train
# dropping date column



train.drop(['Date'],axis =1, inplace =True)

test.drop(['Date'],axis =1, inplace =True)
train.head()
# Rearranging columns of train



train = train [['Id', 'Country_Region', 'Population','day', 'month','Weight','Target', 'TargetValue']]

# Rearranging columns of test



test = test [['ForecastId','Country_Region', 'Population','day', 'month','Weight','Target']]



train
#Splitting input and output



x=train.drop(columns=['TargetValue'],axis=1)

y=train['TargetValue']

x.pop('Id')

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test =train_test_split(x,y, test_size = 0.2, random_state = 0 )
x.head()
#x_train1, x_val1 ,y_train1, y_val1 = train_test_split(x,y, test_size=0.2, random_state=0)

#model = RandomForestRegressor(n_estimators = 250, random_state=0)

#model.fit(x_train1,y_train1)
#Training performance

#y_train_pred = model.predict(x_train1)

#print("Train r2 score ",r2_score(y_train1,y_train_pred))

#Test performance

#y_valid_pred = model.predict(x_val1)

#print("Valid r2 score ",r2_score(y_val1,y_valid_pred))
model_1 = RandomForestRegressor(n_estimators = 400, random_state=7,n_jobs = -1)



model_1.fit(x,y)
model_1.score(x,y)
#Prediction on test data

X_test = test.iloc[:,1:7]

pred = model_1.predict(X_test)
pred_list = [x for x in pred]

sub = pd.DataFrame({'Id': test.index , 'TargetValue': pred_list})
sub['TargetValue'].value_counts()

p=sub.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index()

q=sub.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index()

r=sub.groupby(['Id'])['TargetValue'].quantile(q=0.95).reset_index()

p.columns = ['Id' , 'q0.05']

q.columns = ['Id' , 'q0.5']

r.columns = ['Id' , 'q0.95']
p = pd.concat([p,q['q0.5'] , r['q0.95']],1)
#p['q0.05']=p['q0.05'].clip(0,10000)

#p['q0.05']=p['q0.5'].clip(0,10000)

#p['q0.05']=p['q0.95'].clip(0,10000)

p
p['Id'] =p['Id']+ 1

p
id_list = []

variable_list = []

value_list = []

for index, row in p.iterrows():

  id_list.append(row['Id'])

  variable_list.append('q0.05')

  value_list.append(row['q0.05'])



  id_list.append(row['Id'])

  variable_list.append('q0.5')

  value_list.append(row['q0.5'])



  id_list.append(row['Id'])

  variable_list.append('q0.95')

  value_list.append(row['q0.95'])



sub = pd.DataFrame({'Id':id_list, 'variable': variable_list, 'value':value_list})

sub
sub = sub.astype({'Id':int})

sub['variable']=sub['variable'].str.replace("q","", regex=False)

sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']

sub['TargetValue']=sub['value']

sub=sub[['ForecastId_Quantile','TargetValue']]

sub.reset_index(drop=True,inplace=True)

sub
sub['TargetValue']= round(sub['TargetValue'],2)

sub
sub.to_csv("submission.csv",index=False)