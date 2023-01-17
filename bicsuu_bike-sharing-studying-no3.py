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
import seaborn as sns

%matplotlib inline
import matplotlib.pyplot as plt
pwd
df_test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')

df_train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')
df_test.head()
df_train['datetime'] = pd.to_datetime(df_train['datetime'])

df_train['Day'] = df_train['datetime'].dt.day

df_train['Month'] = df_train['datetime'].dt.month

df_train['Year'] = df_train['datetime'].dt.year

df_train['Hour'] = df_train['datetime'].dt.hour

df_train['Minute'] = df_train['datetime'].dt.minute
df_train.head()
eda = df_train.copy()

eda.head()
eda['season'] = eda['season'].map({1:'Spring',2:'Summer',3:'Fall',4:'Winter'})

eda['holiday'] = eda['holiday'].map({0:'Nholiday',1:'Holiday'})

eda['workingday'] = eda['workingday'].map({0:'Off',1:'Workday'})

eda['weather'] = eda['weather'].map({1: 'Clear, Few clouds, Partly cloudy, Partly cloudy',2: 'Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist',3: 'Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds',4: 'Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog' })
eda.head()
cols = ['temp','atemp','humidity','windspeed','casual','registered']

fig, axes = plt.subplots(2,3,figsize = (10,5))

count = 0

for i in range(2):

    for j in range(3):

        s = cols[count+j]

        sns.distplot(eda[s].values, ax = axes[i][j],bins = 30)

        axes[i][j].set_title(s,fontsize=17)

        fig=plt.gcf()

        fig.set_size_inches(15,10)

        plt.tight_layout()

    count = count+j+1    


sns.distplot(eda['count'])
eda['count'] = np.log1p(eda['count'])

sns.distplot(eda['count'])
eda['count'].describe()
sns.boxplot(x = 'season',y = 'count',data = eda)
sns.boxplot(x = 'weather',y = 'count',data = eda)

plt.xticks(rotation = 90)
sns.violinplot(x = 'season',y = 'count',data = eda);
eda.groupby('season')['count'].sum().plot.bar();
eda.groupby('holiday')['count'].sum().plot.barh()
#sns.lineplot(x = 'count',y = 'atemp',hue = 'season',data = eda)

df_train['count'] = np.log(df_train['count']+1)
sns.distplot(df_train['count'],kde = True,bins = 30)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score,mean_squared_error
df_train.columns
X = df_train.drop(['datetime','count','casual','registered'],axis = 1)

y = df_train['count']

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X.head()
lr = LinearRegression().fit(X_train,y_train)

y_pred = lr.predict(X_test)

print('r2_score:',r2_score(y_test,y_pred))

print('rmse:',np.sqrt(mean_squared_error(y_test,y_pred)))
dt = DecisionTreeRegressor().fit(X_train,y_train)

y_pred = dt.predict(X_test)

print('r2_score:',r2_score(y_test,y_pred))

print('rmse:',np.sqrt(mean_squared_error(y_test,y_pred))) 
rf = RandomForestRegressor().fit(X_train,y_train)

y_pred = rf.predict(X_test)

print('r2_score:',r2_score(y_test,y_pred))

print('rmse:',np.sqrt(mean_squared_error(y_test,y_pred))) 
df_test['datetime'] = pd.to_datetime(df_test['datetime'])

df_test['Day'] = df_test['datetime'].dt.day

df_test['Month'] = df_test['datetime'].dt.month

df_test['Year'] = df_test['datetime'].dt.year

df_test['Hour'] = df_test['datetime'].dt.hour

df_test['Minute'] = df_test['datetime'].dt.minute

test = df_test.drop('datetime',axis = 1)

predictions = rf.predict(test)
predictionsanti = np.exp(predictions)-1

plt.plot(predictionsanti)
sample_submission = pd.DataFrame({'datetime':df_test['datetime'],'count':predictionsanti})

sample_submission.head()

sample_submission.to_csv('sampleSubmission.csv',index=False)
