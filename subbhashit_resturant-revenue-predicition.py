# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/restaurant-revenue-prediction/train.csv.zip")
data.head()
import matplotlib.pyplot as plt
plt.figure(figsize=(15,15))
import seaborn as sns
sns.heatmap(data.corr(),annot=True,cmap="Blues")
data.info()
data.Type.value_counts()
data['City Group'].value_counts()
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
one=OneHotEncoder()
data.City=le.fit_transform(data.City)
a=pd.DataFrame(one.fit_transform(data[['City Group']]).toarray())
b=pd.DataFrame(one.fit_transform(data[['Type']]).toarray())
data=pd.concat([data,a,b],axis=1)
data.drop(['Type','City Group'],axis=1,inplace=True)
data.head()
def year(x):
    o=int(x[6:])
    return o
def month(x):
    o=int(x[3:5])
    return o
def day(x):
    o=int(x[0:2])
    return o
data['year']=data['Open Date'].apply(year)
data['month']=data['Open Date'].apply(month)
data['day']=data['Open Date'].apply(day)
data.drop("Open Date",axis=1,inplace=True)
data.head()
data.columns=[     'Id',    'City',      'P1',      'P2',      'P3',      'P4',
            'P5',      'P6',      'P7',      'P8',      'P9',     'P10',
           'P11',     'P12',     'P13',     'P14',     'P15',     'P16',
           'P17',     'P18',     'P19',     'P20',     'P21',     'P22',
           'P23',     'P24',     'P25',     'P26',     'P27',     'P28',
           'P29',     'P30',     'P31',     'P32',     'P33',     'P34',
           'P35',     'P36',     'P37', 'revenue',         0,         1,
               2,         3,         4,    'year',   'month',     'day']
x=data.drop(["Id","revenue"],axis=1)
y=data["revenue"]
from sklearn.model_selection import train_test_split,KFold,cross_val_score
xr,xt,yr,yt=train_test_split(x,y)
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier,RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBClassifier,XGBRFRegressor,XGBRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression,LinearRegression,SGDRegressor
from sklearn.svm import SVC,SVR
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMRegressor
model=XGBRegressor(n_estimators=1000)
model.fit(x,y)
kfold=KFold(n_splits=10)
res=cross_val_score(model,x,y,cv=kfold)
res.mean()*100
yp=model.predict(xt)
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print(r2_score(yt,yp))
print(mean_absolute_error(yt,yp))
print(mean_squared_error(yt,yp))
test=pd.read_csv("/kaggle/input/restaurant-revenue-prediction/test.csv.zip")
test.head()
test.City=le.fit_transform(test.City)
a=pd.DataFrame(one.fit_transform(test[['City Group']]).toarray())
b=pd.DataFrame(one.fit_transform(test[['Type']]).toarray())
test=pd.concat([test,a,b],axis=1)
test.drop(['Type','City Group'],axis=1,inplace=True)
test['year']=test['Open Date'].apply(year)
test['month']=test['Open Date'].apply(month)
test['day']=test['Open Date'].apply(day)
test.drop("Open Date",axis=1,inplace=True)
test.head()
test.columns=[     'Id',    'City',      'P1',      'P2',      'P3',      'P4',
            'P5',      'P6',      'P7',      'P8',      'P9',     'P10',
           'P11',     'P12',     'P13',     'P14',     'P15',     'P16',
           'P17',     'P18',     'P19',     'P20',     'P21',     'P22',
           'P23',     'P24',     'P25',     'P26',     'P27',     'P28',
           'P29',     'P30',     'P31',     'P32',     'P33',     'P34',
           'P35',     'P36',     'P37', 'revenue',         0,         1,
               2,         3,         4,    'year',   'month',     'day']
pred=test.drop('Id',axis=1)
yp=model.predict(x)
y1=pd.DataFrame(yp)
test=pd.concat([test,y1],axis=1)
test.columns=[     'Id',    'City',      'P1',      'P2',      'P3',      'P4',
            'P5',      'P6',      'P7',      'P8',      'P9',     'P10',
           'P11',     'P12',     'P13',     'P14',     'P15',     'P16',
           'P17',     'P18',     'P19',     'P20',     'P21',     'P22',
           'P23',     'P24',     'P25',     'P26',     'P27',     'P28',
           'P29',     'P30',     'P31',     'P32',     'P33',     'P34',
           'P35',     'P36',     'P37', 'revenue',         0,         1,
               2,         3,         4,    'year',   'month',     'day',
               'Prediction']
test.to_csv('sub.csv',columns=["Id","Prediction"],index=False)
sample=pd.read_csv("/kaggle/input/restaurant-revenue-prediction/sampleSubmission.csv")
sample.head()
sample1=pd.read_csv("sub.csv")
sample1.head()