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
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.express as px

import seaborn as sns
train = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')

test = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')

submit = pd.read_csv('../input/covid19-global-forecasting-week-5/submission.csv')
train.head()
test.head()
submit.head()
print("length of train", len(train))

print("length of test", len(test))

print("length of submit", len(submit))
train.describe()
train.sort_values(by=['TargetValue']).head()
bydate = train.groupby("Date").sum()

bydate.head()
bydate["TargetValue"].plot()
fig = px.pie(train,

             values="TargetValue",

             names="Target",

             template="seaborn")

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
fig = px.pie(train,

             values="TargetValue",

             names="Country_Region",

             template="seaborn")

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label", textposition='inside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
bycountry = train.groupby(["Country_Region" , "Date"]).sum() #Group by country region and date

bycountry.head()
bycountry.reset_index(inplace=True)
bycountry['Date'] = pd.to_datetime(bycountry['Date']) # change date format
px.line(bycountry, x='Date', y='TargetValue', color='Country_Region', title='COVID19 Total Cases growth by date')
train.isnull().sum() # check null values
test.isnull().sum()
train = train.drop(['County','Province_State','Country_Region','Target'],axis=1) 

test = test.drop(['County','Province_State','Country_Region','Target'],axis=1)

train.head()
test_date_min = test['Date'].min()

test_date_max = test['Date'].max()

train['Date']=pd.to_datetime(train['Date'])

test['Date']=pd.to_datetime(test['Date'])

test['Date']=test['Date'].dt.strftime("%Y%m%d")

train['Date']=train['Date'].dt.strftime("%Y%m%d").astype(int)
train.head()
test.drop(['ForecastId'],axis=1,inplace=True)

test.index.name = 'Id'

test.head()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OrdinalEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, KFold
X = train.drop(['TargetValue', 'Id'], axis=1)

y = train["TargetValue"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 101)
model = RandomForestRegressor(n_jobs=-1)

model.set_params(n_estimators=100)

model.fit(X_train, y_train)

model.score(X_test, y_test)
y_pred2 = model.predict(X_test)

y_pred2
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

sub.head()