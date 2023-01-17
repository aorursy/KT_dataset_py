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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df2019 = pd.read_csv('/kaggle/input/flight-delay-prediction/Jan_2019_ontime.csv')

df2020 = pd.read_csv('/kaggle/input/flight-delay-prediction/Jan_2020_ontime.csv')

df2019.head()
df2019.info()
df2020.head()
df2020.info()
df2019.drop('Unnamed: 21',axis=1,inplace=True)

df2020.drop('Unnamed: 21',axis=1,inplace=True)
df2019 = df2019.dropna()

df2019.info()
df2020 = df2020.dropna()

df2020.info()
days = {1:'Monday',2:'Tuesday',3:'Wednesday',4:'Thursday',5:'Friday',6:'Saturday',7:'Sunday'}

dayorder = ('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday')

df2019 = df2019.replace({'DAY_OF_WEEK':days})

df2020 = df2020.replace({'DAY_OF_WEEK':days})
uni_carrier2019 = df2019['OP_UNIQUE_CARRIER'] == df2019['OP_CARRIER']

uni_carrier2019.unique()
col_drop = ['ORIGIN_AIRPORT_ID','ORIGIN_AIRPORT_SEQ_ID','DEST_AIRPORT_ID','DEST_AIRPORT_SEQ_ID','OP_CARRIER_FL_NUM','TAIL_NUM','OP_UNIQUE_CARRIER']

df2019.drop(col_drop,axis=1,inplace=True)

df2020.drop(col_drop,axis=1,inplace=True)

df2019['DEP_TIME_BLK'] = df2019['DEP_TIME_BLK'].str[:4]

df2020['DEP_TIME_BLK'] = df2020['DEP_TIME_BLK'].str[:4]

df2019['DEP_TIME_BLK'] = df2019['DEP_TIME_BLK'].astype(float) 

df2020['DEP_TIME_BLK'] = df2020['DEP_TIME_BLK'].astype(float) 
sns.set_style('whitegrid')
sns.countplot(x='ARR_DEL15',data=df2019)
df2019['ARR_DEL15'].mean()
sns.countplot(x="DAY_OF_WEEK",data=df2019,hue='ARR_DEL15', order=dayorder)
df2019delay = df2019.groupby('DAY_OF_WEEK')['ARR_DEL15'].describe()

df2019delay
sns.countplot(x="DAY_OF_MONTH",data=df2019,hue='ARR_DEL15',)
sns.countplot(x="OP_CARRIER",data=df2019,hue='ARR_DEL15',)
df2019.groupby('OP_CARRIER')['ARR_DEL15'].mean()
df2019['ORIGIN'].nunique()
df2019['DEST'].nunique()
df2019.drop(['ORIGIN','DEST'],axis=1,inplace=True)

df2020.drop(['ORIGIN','DEST'],axis=1,inplace=True)
redays = {'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}

df2019 = df2019.replace({'DAY_OF_WEEK':redays})

df2020 = df2020.replace({'DAY_OF_WEEK':redays})
df2019 = pd.get_dummies(df2019,drop_first=True)

df2020 = pd.get_dummies(df2020,drop_first=True)
df2019.info()
df2020.info()
df2019.drop('OP_CARRIER_AIRLINE_ID',axis=1,inplace=True)

df2020.drop('OP_CARRIER_AIRLINE_ID',axis=1,inplace=True)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

X = df2019.drop('ARR_DEL15',axis=1)

y = df2019['ARR_DEL15']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
X_test2020 = df2020.drop('ARR_DEL15',axis=1)

y_test2020 = df2020['ARR_DEL15']
predictions = logmodel.predict(X_test2020)
print(confusion_matrix(y_test2020,predictions))
print(classification_report(y_test2020,predictions))