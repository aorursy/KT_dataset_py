# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/sangamiitmadrashackathon/"))



# Any results you write to the current directory are saved as output.



%matplotlib inline

import warnings

warnings.filterwarnings("ignore")





#########*****#########

import pandas_profiling

#########*****#########



import sqlite3

import pandas as pd

import numpy as np

import nltk

import string

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import roc_curve, auc

from nltk.stem.porter import PorterStemmer



import re

import string



from tqdm import tqdm

import os



######################



from xgboost.sklearn import XGBClassifier, XGBRegressor

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier

from sklearn.calibration import CalibratedClassifierCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification

from sklearn.metrics import roc_auc_score



from wordcloud import WordCloud

from IPython.core.display import HTML  

import plotly.offline as offline

import plotly.graph_objs as go

offline.init_notebook_mode()
data_train = pd.read_csv("../input/sangamiitmadrashackathon/Train.csv")

data_test = pd.read_csv("../input/sangamiitmadrashackathon/Test.csv")
# import pandas_profiling

# data_train.profile_report()

dates = data_test['date_time']
data = pd.concat((data_train, data_test))

data.shape
def holiday_Cat(df):

    if df == 'None':

        return 0

    else:

        return 1

def weather_cat(df):

    if df == 'Clouds':

        return 1

    elif df == 'Clear':

        return 2

    elif df == 'Mist':

        return 3

    elif df == 'Rain':

        return 4

    elif df == 'Snow':

        return 5

    elif df == 'Drizzle':

        return 6

    elif df == 'Haze':

        return 7

    elif df == 'Fog':

        return 8

    elif df == 'Thunderstorm':

        return 9

    elif df == 'Smoke':

        return 10

    else:

        return 0



def hour_cat(df):

    if df<7 or df>=23:

        return 4

    elif df>=7 and df<12:

        return 1

    elif df>=12 and df<17:

        return 2

    elif df>=17 and df<23:

        return 3
data['proper_date_time'] = pd.to_datetime(data['date_time'])

data = data.drop(['date_time'], axis=1)

data['year']=data['proper_date_time'].dt.year

data['month']=data['proper_date_time'].dt.month

data['weekday'] =data['proper_date_time'].dt.dayofweek

data['hour'] = data['proper_date_time'].dt.hour



# a = np.array(data['hour'].values.tolist())

# data['hour']=np.where(a>=23, 4, a).tolist()



data['is_holiday'] = data['is_holiday'].map(holiday_Cat) 

data['weather_type'] = data['weather_type'].map(weather_cat) 

data['hour'] = data['hour'].map(hour_cat) 



data = data[data['temperature']>230]

data= data.drop(['weather_description','proper_date_time','visibility_in_miles'], axis=1)


fe = ['air_pollution_index','clouds_all','humidity','snow_p_h','temperature','wind_direction','wind_speed','year','month','weekday', 'hour']



for i in range(len(fe)):

    for j in range(i,len((fe))):

        data[str(fe[i])+'V'+str(fe[j])] = data[fe[i]]*data[fe[j]]

data.head()
data_train = data[data['traffic_volume'].notnull()]

data_test = data[data['traffic_volume'].isnull()]
data_train.head()
# Time Based Splitting

# X_train = data_train[data_train['year']<2017]

X_train = data_train[:24000]

Y_train = X_train['traffic_volume']

X_train = X_train.drop(['traffic_volume'], axis=1)



# X_cv = data_train[data_train['year']>=2017]

X_cv = data_train[24001:]

Y_cv = X_cv['traffic_volume']

X_cv = X_cv.drop(['traffic_volume'], axis=1)



x_train_full = data_train

y_train_full = x_train_full['traffic_volume']

x_train_full = x_train_full.drop(['traffic_volume'], axis=1)





X_test = data_test

X_test = X_test.drop(['traffic_volume'],axis=1)
print("Train: ", X_train.shape)

print("CV: ", X_cv.shape)

print("Test: ", X_test.shape)
df = X_test

for i in df.columns:

    if len(df[df[i].isnull()]):

        df[i].fillna((df[i].mode()[0]), inplace=True)
X_test.head()
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from math import sqrt



estimators = [10,50,100,200,300,400,500,600,800,850,900]

depths=[5,7,10,20,50,80,100,120,160,240,340,500]

iss,js=[],[]

cv_score = []

for i in tqdm(estimators):

    for j in depths:

        rgr = RandomForestRegressor(max_depth=j, n_estimators=i, max_features='sqrt')

        rgr.fit(X_train,Y_train)

        y_cv_pred = rgr.predict(X_cv)

        iss.append(i)

        js.append(j)

        cv_score.append(mean_squared_error(Y_cv,y_cv_pred))



optimal_depth=js[cv_score.index(min(cv_score))]   

optimal_n_estimator=iss[cv_score.index(min(cv_score))]

print('optimal depth : ',optimal_depth)

print('optimal n_estimator : ',optimal_n_estimator)

print('Error : ',sqrt(min(cv_score)))
data = pd.DataFrame({'n_estimators': iss, 'max_depth': js, 'MSE': cv_score})

data_pivoted = data.pivot("n_estimators", "max_depth", "MSE")

ax = sns.heatmap(data_pivoted,annot=True)

plt.title('Heatmap for train data')

plt.show()
rgr = RandomForestRegressor(max_depth=optimal_depth, n_estimators=optimal_n_estimator, max_features='sqrt')

rgr.fit(x_train_full,y_train_full)

y_rf_ans = rgr.predict(X_test)


base_learners = [20,40,60,80,100,120,200,300,400,500]

depths=[1,5,10,50,100,500,1000]



X=[]

Y=[]

Z=[]

for bl in tqdm(base_learners):

    for d in depths:

        gbdt=XGBRegressor(learning_rate=0.05, max_depth=d,n_estimators=bl)

        gbdt.fit(X_train,Y_train)

        pred=gbdt.predict(X_cv)

        X.append(bl)

        Y.append(d)

        Z.append(mean_squared_error(Y_cv,pred))

        

optimal_depth=Y[Z.index(min(Z))]   

optimal_n_estimator=X[Z.index(min(Z))]

print('optimal depth : ',optimal_depth)

print('optimal n_estimator : ',optimal_n_estimator)

        
gbdt=XGBRegressor(learning_rate=0.05, max_depth=optimal_depth, n_estimators=optimal_n_estimator)

gbdt.fit(X_train,Y_train)

y_gbdt_ans = gbdt.predict(X_test)
y = (np.array(y_gbdt_ans) + np.array(y_rf_ans))/2


ans = pd.DataFrame()

ans['date_time'] = dates

ans['traffic_volume'] = y

ans.to_csv('ensembleResult3.csv',header=True,index=False)



ans['traffic_volume'] = y_gbdt_ans

ans.to_csv('ensembleResult4.csv',header=True,index=False)





ans['traffic_volume'] = y_rf_ans

ans.to_csv('ensembleResult5.csv',header=True,index=False)