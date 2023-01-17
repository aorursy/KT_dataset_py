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
test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/test.csv")

train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv")
train.Province_State.fillna("None", inplace=True)

display(train.head(5))

display(train.describe())

print("Number of Country_Region: ", train['Country_Region'].nunique())

test.Province_State.fillna("None", inplace=True)

display(test.head(5))

display(test.describe())
confirmed_total_date_india = train[train['Country_Region']=='India'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_india = train[train['Country_Region']=='India'].groupby(['Date']).agg({'Fatalities':['sum']})
confirmed_total_date_india
confirmed_total_date_india
fatalities_total_date_india 
total_date_india = confirmed_total_date_india.join(fatalities_total_date_india)

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

import time

from datetime import datetime

from scipy import integrate, optimize

import warnings

warnings.filterwarnings('ignore')



# ML libraries

import lightgbm as lgb

import xgboost as xgb

from xgboost import plot_importance, plot_tree

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn import linear_model

from sklearn.metrics import mean_squared_error
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))

total_date_india.plot(ax=ax1)

ax1.set_title("confirmed cases in india", size=13)

ax1.set_ylabel("Number of cases", size=13)

ax1.set_xlabel("Date", size=13)

fatalities_total_date_india.plot(ax=ax2, color='red')

ax2.set_title("fatalities in india", size=13)

ax2.set_ylabel("Number of cases", size=13)

ax2.set_xlabel("Date", size=13)
total_date_india
import seaborn as seabornInstance 

from sklearn.model_selection import train_test_split 

import sklearn.linear_model 

from sklearn import metrics

%matplotlib inline
india = [i for i in confirmed_total_date_india.values]

india_76 = india[0:76] 
plt.figure(figsize=(12,6))

plt.plot(india_76)
print(india_76)
india=np.array(india_76)

y=india

print(y)

X=[[i] for i in range(1,77)]

X=np.array(X)

print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.svm import SVR

regressor = SVR(kernel='poly',degree=10)

regressor.fit(X_train,y_train)

regressor.score(X_test,y_test)#accuracy 97.98%
test
t=[]

for j in range(365):

    for i in range(70,106):

        t.append([i])

for j in range(18):

    t.append([j+70])

print(len(t))


test=t

test=np.array(test)

results=regressor.predict(test)

results=np.array(results)

results=results.astype(int)

print(results)
#results = np.argmax(results,axis = 1)



results = pd.Series(results,name="ConfirmedCases")

submission = pd.concat([pd.Series(range(1,13158),name = "ForecastId"),results],axis = 1)



submission.to_csv("svr_datagen.csv",index=False)
import matplotlib

import matplotlib.pyplot as plt

import numpy as np





labels = ["15-4-2020","22-4-2020","29-4-2020","06-5-2020","13-5-2020","20-5-2020","27-5-2020"]

men_means = [13291,29591,62087,123774,236001,432716,766370]

women_means = [364,748,1458,2713,4851,8371,14003]



x = np.arange(len(labels))  # the label locations

width = 0.35  # the width of the bars



fig, ax = plt.subplots( figsize=(10,6))

rects1 = ax.bar(x - width/2, men_means, width, label='confirmed_cases')

rects2 = ax.bar(x + width/2, women_means, width, label='Fatalities')



# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('number of persons')

ax.set_xlabel('on or before date')

ax.set_title('corona_forcast_India_ARKA')

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend()







def autolabel(rects):

    """Attach a text label above each bar in *rects*, displaying its height."""

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 3),  # 3 points vertical offset

                    textcoords="offset points",

                    ha='center', va='bottom')





autolabel(rects1)

autolabel(rects2)



fig.tight_layout()



plt.show()
india = [i for i in fatalities_total_date_india.values]

india_76 = india[0:76] 
plt.figure(figsize=(12,6))

plt.plot(india_76)
india=np.array(india_76)

y=india

print(y)

X=[[i] for i in range(1,77)]

X=np.array(X)

print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.svm import SVR

regressor = SVR(kernel='poly',degree=9)

regressor.fit(X_train,y_train)

regressor.score(X_test,y_test)#accuracy 91.79%
t=[]

for j in range(365):

    for i in range(70,106):

        t.append([i])

for j in range(18):

    t.append([j+70])

print(len(t))
test=t

test=np.array(test)

results1=regressor.predict(test)

results1=np.array(results1)

results1=results1.astype(int)

print(results1)
results1 = pd.Series(results1,name="Fatalities")

results
results1
submission = pd.concat([pd.Series(range(1,13159),name = "ForecastId"),results],axis = 1)
submission1 = pd.concat([submission,results1],axis = 1)

submission1
#submission = pd.concat([pd.Series(range(1,13158),name = "ForecastId"),results],axis = 1)



submission1.to_csv("submission.csv",index=False)
test=[[84],[91],[98],[105],[112],[119],[126]]

test=np.array(test)

predicted_value=regressor.predict(test)

print(predicted_value)