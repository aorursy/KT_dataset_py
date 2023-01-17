



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

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current sessio
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline

import plotly.express as pix

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing  import OneHotEncoder,StandardScaler

from sklearn.model_selection import train_test_split,cross_val_score,KFold,RandomizedSearchCV

from sklearn.linear_model import LinearRegression,Ridge,Lasso

import matplotlib

from numpy import mean,std

import sklearn

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

font={'family':'normal',

     'weight':'normal','size':10}

matplotlib.rc('font',**font)
train=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/train.csv")

test=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/test.csv")
train.head()
train.info()
test.info()
def get_nan(data):

    nan_data=data.isna().sum()

    Nan_perctg=nan_data/data.shape[0]

    return Nan_perctg*100
train_nan=get_nan(train)

test_nan=get_nan(test)

Nan_Dataframe=pd.DataFrame({'train':train_nan,'test':test_nan})

Nan_Dataframe
print(f"train_shape:{train.shape},test_shape:{test.shape}")
train['Date']=pd.to_datetime(train['Date'])

test['Date']=pd.to_datetime(test['Date'])

Id=train['Id']

Fid=test['ForecastId']

train=train.drop(columns=['Id'],axis=1)

test=test.drop(columns=['ForecastId'],axis=1)

train=train.rename(columns={'County':'Country'})

test=test.rename(columns={'County':'Country'})
fig,ax=plt.subplots(1,1)

sns.countplot(x='Target',data=train,ax=ax)

ax.set_title("Confirmed_Cases VS Fatalities")

fig.show()
fig,ax=plt.subplots(1,1,figsize=(10,5))

sns.violinplot(y='TargetValue',x='Target',data=train,ax=ax)

fig,ax=plt.subplots(1,1,figsize=(7,5))

sns.barplot(x='Target',y='Population',data=train,ax=ax)

ax.set_title("Population Vs Target")

fig=plt.figure(figsize=(30,32))

fig=pix.pie(train,values='TargetValue',names='Country_Region',hole=.3,color_discrete_sequence=pix.colors.sequential.RdBu)

fig.update_traces(textposition='inside')

fig.update_layout(uniformtext_minsize=12,uniformtext_mode='hide')

fig.show()
train['day'] = train['Date'].dt.day

train['month'] = train['Date'].dt.month

train['dayofweek'] = train['Date'].dt.dayofweek

train['dayofyear'] = train['Date'].dt.dayofyear

train['quarter'] = train['Date'].dt.quarter

train['weekofyear'] = train['Date'].dt.weekofyear

test['day'] = test['Date'].dt.day

test['month'] = test['Date'].dt.month

test['dayofweek'] = test['Date'].dt.dayofweek

test['dayofyear'] = test['Date'].dt.dayofyear

test['quarter'] = test['Date'].dt.quarter

test['weekofyear'] = test['Date'].dt.weekofyear

train=train.drop(columns=['Date','Country','Province_State'],axis=1)

test=test.drop(columns=['Date','Country','Province_State'],axis=1)

Y=train['TargetValue']

X=train.drop(columns=['TargetValue'],axis=1)

x_train,x_val,y_train,y_val=train_test_split(X,Y,test_size=0.1)

x_train_numerical=x_train.select_dtypes(include=['int64','float64']).columns

x_train_object=x_train.select_dtypes(include=['object']).columns
t=[('cat',OneHotEncoder(),x_train_object),('num',StandardScaler(),x_train_numerical)]

column_trans=ColumnTransformer(transformers=t)
dt=DecisionTreeRegressor(random_state=19)

pipe_dt=Pipeline(steps=[('prep',column_trans),('dt',dt)])

pipe_dt.fit(x_train,y_train)

prediction=pipe_dt.predict(x_val)
score=pipe_dt.score(x_val,y_val)
score
test_prediction=pipe_dt.predict(test)

output=pd.DataFrame({'ID':Fid,'TargetValue':test_prediction})

output
a=output.groupby(['ID'])['TargetValue'].quantile(0.05).reset_index()

b=output.groupby(['ID'])['TargetValue'].quantile(0.5).reset_index()

c=output.groupby(['ID'])['TargetValue'].quantile(0.95).reset_index()
a.columns=['Id','q0.05']

b.columns=['Id','q0.5']

c.columns=['Id','q0.95']

a=pd.concat([a,b['q0.5'],c['q0.95']],1)

a['q0.05']=a['q0.05']

a['q0.5']=a['q0.5']

a['q0.95']=a['q0.95']

sub=pd.melt(a, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])

sub['variable']=sub['variable'].str.replace("q","", regex=False)

sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']

sub['TargetValue']=sub['value']

sub=sub[['ForecastId_Quantile','TargetValue']]

sub.reset_index(drop=True,inplace=True)

sub.to_csv("submission.csv",index=False)

sub.head()