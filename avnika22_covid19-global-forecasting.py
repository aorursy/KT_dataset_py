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
import seaborn as sns

import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import plotly.express as px

from datetime import datetime

%matplotlib inline
train=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/train.csv")

test=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/test.csv")

train.head()
test.head()
train.shape
test.shape
train.isnull().sum()
test.isnull().sum()
ID=train['Id']

FID=test['ForecastId']

train=train.drop(columns=['County','Province_State','Id'])

test=test.drop(columns=['County','Province_State','ForecastId'])
sns.pairplot(train)
sns.barplot(y='TargetValue',x='Target',data=train)
sns.barplot(x='Target',y='Population',data=train)
fig=plt.figure(figsize=(45,30))

fig=px.pie(train, values='TargetValue', names='Country_Region',color_discrete_sequence=px.colors.sequential.RdBu,hole=.4)

fig.update_traces(textposition='inside')

fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')

fig.show()
df_grouped=train.groupby(['Country_Region']).sum()

df_grouped.TargetValue
plot=df_grouped.nlargest(5,'TargetValue')

plot
sns.catplot(y="Population", x="TargetValue",height=5,aspect=1,kind="bar", data=plot)

plt.title('Top 5 Target Values',size=20)

plt.show()
plot=df_grouped.nlargest(5,'Population')

plot
fig = px.treemap(train, path=['Country_Region'], values='TargetValue',

                  color='Population', hover_data=['Country_Region'],

                  color_continuous_scale='matter')

fig.show()
da= pd.to_datetime(train['Date'], errors='coerce')

train['Date']= da.dt.strftime("%Y%m%d").astype(int)

da= pd.to_datetime(test['Date'], errors='coerce')

test['Date']= da.dt.strftime("%Y%m%d").astype(int)
plot=train.nlargest(2000,'TargetValue')

plot
fig, ax = plt.subplots(figsize=(10,10))  

h=pd.pivot_table(plot,values='TargetValue',

index=['Country_Region'],

columns='Date')

sns.heatmap(h,cmap="RdYlGn",linewidths=0.05)
plot=train.nlargest(2000,'Population')

plot
fig, ax = plt.subplots(figsize=(20,10))  

h=pd.pivot_table(plot,values='TargetValue',

index=['Country_Region'],

columns='Date')

sns.heatmap(h,cmap="RdYlGn",linewidths=0.005)
train.select_dtypes(include=['object']).columns
test.select_dtypes(include=['object']).columns
from sklearn.preprocessing import LabelEncoder

l = LabelEncoder()

X = train.iloc[:,0].values

train.iloc[:,0] = l.fit_transform(X.astype(str))



X = train.iloc[:,4].values

train.iloc[:,4] = l.fit_transform(X)
from sklearn.preprocessing import LabelEncoder

l = LabelEncoder()

X = test.iloc[:,0].values

test.iloc[:,0] = l.fit_transform(X.astype(str))



X = test.iloc[:,4].values

test.iloc[:,4] = l.fit_transform(X)
train.head()


y_train=train['TargetValue']

x_train=train.drop(['TargetValue'],axis=1)



from sklearn.model_selection import train_test_split 



x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

pip = Pipeline([('scaler2' , StandardScaler()),

                        ('RandomForestRegressor: ', RandomForestRegressor())])

pip.fit(x_train , y_train)

prediction = pip.predict(x_test)
acc=pip.score(x_test,y_test)

acc
predict=pip.predict(test)
output=pd.DataFrame({'id':FID,'TargetValue':predict})

output
a=output.groupby(['id'])['TargetValue'].quantile(q=0.05).reset_index()

b=output.groupby(['id'])['TargetValue'].quantile(q=0.5).reset_index()

c=output.groupby(['id'])['TargetValue'].quantile(q=0.95).reset_index()

    

    
a.columns=['Id','q0.05']

b.columns=['Id','q0.5']

c.columns=['Id','q0.95']

a=pd.concat([a,b['q0.5'],c['q0.95']],1)

a['q0.05']=a['q0.05']

a['q0.5']=a['q0.5']

a['q0.95']=a['q0.95']

a
sub=pd.melt(a, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])

sub['variable']=sub['variable'].str.replace("q","", regex=False)

sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']

sub['TargetValue']=sub['value']

sub=sub[['ForecastId_Quantile','TargetValue']]

sub.reset_index(drop=True,inplace=True)

sub.to_csv("submission.csv",index=False)

sub.head()