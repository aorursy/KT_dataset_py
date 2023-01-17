!pip install plotly
import pandas as pd

import numpy as py

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.express as px

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.model_selection import GridSearchCV, KFold

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import datetime





import sklearn.discriminant_analysis

import sklearn.linear_model as skl_lm

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import confusion_matrix, classification_report, precision_score

from sklearn import preprocessing

from sklearn import neighbors

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import PolynomialFeatures

from sklearn import metrics

from datetime import timedelta

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression,Ridge,Lasso

from sklearn.metrics import hamming_loss, accuracy_score 

from pandas import DataFrame

from datetime import datetime

from sklearn import preprocessing

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error

import statsmodels.api as sm

import statsmodels.formula.api as smf
train= pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')

test = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')

submission = pd.read_csv('../input/covid19-global-forecasting-week-5/submission.csv')

print(train.shape)

train.head()


#date=pd.to_datetime(datatrain['Date'])

#datet=pd.to_datetime(datatest['Date'])

#print(date)
#ldate=int(len(date))

#ldatet=int(len(datet))

#m = []

#d = []

##for i in range(0,ldate):

  #  dx = (date[i].strftime("%d"))

   # mx = (date[i].strftime("%m"))

    #m.append(int(mx))

    #d.append(int(dx))



#mt = []

#dt = []

#for i in range(0,ldatet):

 #   dtx = (datet[i].strftime("%d"))

  #  mtx = (datet[i].strftime("%m"))

   # mt.append(int(mtx))

    #dt.append(int(dtx))

#train = datatrain

#test = datatest
#train.insert(6,"Month",m,False)

#train.insert(7,"Day",d,False)

#test.insert(4,"Month",mt,False)

#test.insert(5,"Day",dt,False)

#print("Datatrain")

#traindays = datatrain['Date'].nunique()

#print("Number of Country_Region: ", datatrain['Country_Region'].nunique())

#print("Number of Province_State: ", datatrain['Province_State'].nunique())

#print("Number of Days: ", traindays)



#notrain = datatrain['Id'].nunique()

#print("Number of datapoints in train:", notrain)

#lotrain = int(notrain/traindays)

#print("L Trains:", lotrain)
#print("Datatest")

#testdays = datatest['Date'].nunique()

#print("Number of Days: ", testdays)

#notest = datatest['ForecastId'].nunique()

#print("Number of datapoints in test:", notest)

#lotest = int(notest/testdays)

#print("L Test:", lotest)
#zt = datet[0]

#daycount = []

#for i in range(0,lotrain):

 #   for j in range(1,traindays+1):

  #      daycount.append(j)

#for i in range(traindays):

 #   zx=0

  #  if(zt == date[i]):

   #     

    #    zx = i

     #   print(zx)

        

#daytest = []

#for i in range(0,lotest):

 #   for j in range(1,testdays+1):

  #      jr = zx + j

   #     daytest.append(jr)



#train.insert(8,"DayCount",daycount,False)

#test.insert(6,"DayCount",daytest,False)
#train.head()
#traincount = int(len(train["Date"]))



#testcount = int(len(test["Date"]))
#train.Province_State = train.Province_State.fillna(0)

#empty = 0

#for i in range(0,traincount):

 #   if(train.Province_State[i] == empty):

  #      train.Province_State[i] = train.Country_Region[i]
#test.Province_State = test.Province_State.fillna(0)

#empty = 0

#for i in range(0,testcount):

 #   if(test.Province_State[i] == empty):

  #      test.Province_State[i] = test.Country_Region[i]
#label = preprocessing.LabelEncoder()

#train.Country_Region = label.fit_transform(train.Country_Region)

#train.Province_State = label.fit_transform(train.Province_State)
#test.Country_Region = label.fit_transform(test.Country_Region)

#test.Province_State = label.fit_transform(test.Province_State)
#X = py.c_[train["Province_State"], train["Country_Region"], train["DayCount"], train["Month"], train["Day"]]

#Xt = py.c_[test["Province_State"], test["Country_Region"], test["DayCount"], test["Month"], test["Day"]]

#X
#Y1 = train[train['Target']=="ConfirmedCases"]

#Y2=train[train['Target']=="Fatalities"]

#Y11=Y1.iloc[:,-1]

#Y22=Y2.iloc[:,-1]

#Y11

#Y22

fig = px.pie(train, values='TargetValue', names='Target')

fig.update_traces(textposition='inside')

fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')

fig.show()
ww_me=pd.melt(train,id_vars=['Date'],value_vars=['Target'])

ww_me
train.corr()
train=train.drop(['County','Province_State','Country_Region','Target'],axis=1)

test=test.drop(columns=['County','Province_State','Country_Region','Target'])

train
from sklearn.preprocessing import OrdinalEncoder



def create_features(df):

    df['day'] = df['Date'].dt.day

    df['month'] = df['Date'].dt.month

    df['dayofweek'] = df['Date'].dt.dayofweek

    df['dayofyear'] = df['Date'].dt.dayofyear

    df['quarter'] = df['Date'].dt.quarter

    df['weekofyear'] = df['Date'].dt.weekofyear

    return df
def train_dev_split(df, days):

    #Last days data as dev set

    date = df['Date'].max() - dt.timedelta(days=days)

    return df[df['Date'] <= date], df[df['Date'] > date]
test_date_min = test['Date'].min()

test_date_max = test['Date'].max()

def avoid_data_leakage(df, date=test_date_min):

    return df[df['Date']<date]

def to_integer(dt_time):

    return 10000*dt_time.year + 100*dt_time.month + dt_time.day
train['Date']=pd.to_datetime(train['Date'])

test['Date']=pd.to_datetime(test['Date'])

test['Date']=test['Date'].dt.strftime("%Y%m%d").astype(int)

train['Date']=train['Date'].dt.strftime("%Y%m%d").astype(int)
from sklearn.model_selection import train_test_split



predictors = train.drop(['TargetValue', 'Id'], axis=1)

target = train["TargetValue"]

X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.22, random_state = 0)
test=test.drop(['ForecastId'],axis=1)






model = XGBRegressor(n_estimators = 2500 , alpha=0,gamma=0,learning_rate=0.04,max_depth=23,random_state=42)

model.fit(X_train, y_train)



scores = []



scores.append(model.score(X_test, y_test))

X_test



y_pred2 = model.predict(X_test)

y_pred2

predictions=[]

predictions = model.predict(test)



pred_list = [int(x) for x in predictions]



output = pd.DataFrame({'Id': test.index, 'TargetValue': pred_list})

print(output)
a=output.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index()

b=output.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index()

c=output.groupby(['Id'])['TargetValue'].quantile(q=0.95).reset_index()

a.columns=['Id','q0.05']

b.columns=['Id','q0.5']

c.columns=['Id','q0.95']

a=pd.concat([a,b['q0.5'],c['q0.95']],1)

a['q0.05']=a['q0.05'].clip(0,10000)

a['q0.5']=a['q0.5'].clip(0,10000)

a['q0.95']=a['q0.95'].clip(0,10000)

a
a['Id'] =a['Id']+ 1

a
sub=pd.melt(a, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])

sub['variable']=sub['variable'].str.replace("q","", regex=False)

sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']

sub['TargetValue']=sub['value']

sub=sub[['ForecastId_Quantile','TargetValue']]

sub.reset_index(drop=True,inplace=True)

sub.to_csv("submission.csv",index=False)