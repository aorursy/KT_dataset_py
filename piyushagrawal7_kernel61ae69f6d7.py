# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd#  processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
%matplotlib inline


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_tr = pd.read_csv('../input/flight_delays_train.csv')
df_ts = pd.read_csv('../input/flight_delays_test.csv')
df_tr.head(1)
#df_ts.head(1)

df_tr.info()

df_tr.describe()
plt.figure(figsize=[15,5])
sns.countplot(df_tr['UniqueCarrier'], hue=df_tr['dep_delayed_15min'])


corr= df_tr.corr()
sns.heatmap(data=corr,annot=True)
x = df_tr['Month'].str.split('-')
df_tr['Mon']=x.apply(lambda x:x[1])

y = df_tr['DayofMonth'].str.split('-')
df_tr['DOM']=y.apply(lambda x:x[1])

z = df_tr['DayOfWeek'].str.split('-')
df_tr['DOW']=z.apply(lambda x:x[1])


    
df_tr.info()

df = df_tr.drop(columns=['Month','DayofMonth','DayOfWeek','UniqueCarrier','Origin','Dest'])
df.head()
sns.countplot(x='dep_delayed_15min',data=df)

X_train = df.drop(columns='dep_delayed_15min')
Y_train = df['dep_delayed_15min']
model = LogisticRegression()
model.fit(X_train,Y_train)

df_ts.head()
x = df_ts['Month'].str.split('-')
df_ts['Mon']=x.apply(lambda x:x[1])

y = df_ts['DayofMonth'].str.split('-')
df_ts['DOM']=y.apply(lambda x:x[1])

z = df_ts['DayOfWeek'].str.split('-')
df_ts['DOW']=z.apply(lambda x:x[1])

df_ts.head()
df2 = df_ts.drop(columns=['Month','DayofMonth','DayOfWeek','UniqueCarrier','Origin','Dest'])
predictions = model.predict(df2)
print(predictions)


