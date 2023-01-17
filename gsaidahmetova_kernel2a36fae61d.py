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
%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
df = pd.read_csv('/kaggle/input/gun-violence-data/gun-violence-data_01-2013_03-2018.csv')
df.head()
p_null= (len(df) - df.count())*100.0/len(df)
p_null
train = df[['date','state','city_or_county','address','n_killed','n_injured']]
df.isnull().any() 
plt.figure(figsize=(18,12))
state=df['state'].value_counts()
sns.barplot(state.values,state.index)
plt.xlabel("Number of incidences",fontsize=15)
plt.ylabel("States",fontsize=15)
plt.title("Данные о насилии и оружии в Штатах",fontsize=20)
sns.despine(left=True,right=True)
plt.show()
plt.figure(figsize=(18,12))
state=df['city_or_county'].value_counts()[:20]
sns.barplot(state.values,state.index)
plt.xlabel("Number of incidences",fontsize=15)
plt.ylabel("cities",fontsize=15)
plt.title("Данные о насилии и оружии в городах",fontsize=20)
sns.despine(left=True,right=True)
plt.show()
df['date'] = pd.to_datetime(df['date'])
df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
df["week"] = df["date"].dt.week
df["weekday"] = df["date"].dt.weekday
df["quarter"] = df["date"].dt.quarter
year_wise_total= df[["incident_id"]].groupby(df["year"]).count()
top_year = year_wise_total.sort_values(by='incident_id', ascending=False)
print(top_year)
top_year.plot.barh()
del(top_year)
year_wise = df[["n_killed", "n_injured"]].groupby(df["year"]).sum()
density_plot=sns.kdeplot(year_wise['n_killed'],shade=True,color="red")
density_plot=sns.kdeplot(year_wise['n_injured'],shade=True,color="blue")
print(year_wise['n_killed'])
sns.distplot(year_wise['n_killed'], hist=False, rug=True);
sns.countplot(x='month', data=df)
train = df[['day','month','year','n_killed','n_injured','incident_url_fields_missing']]
df.isnull().any() 
train['incident_url_fields_missing'].fillna('False', inplace = True)
train.isnull().any()
train['incident_url_fields_missing'].replace('True', 1, inplace = True)
train['incident_url_fields_missing'].replace('False', 0, inplace = True)
train.head()
sns.heatmap(train.corr(),cmap='coolwarm',annot=True)
min_max_scaler = preprocessing.MinMaxScaler()
scaled = min_max_scaler.fit_transform(train[['incident_url_fields_missing']])
train[['incident_url_fields_missing']] = pd.DataFrame(scaled)

train.head()
X_train, X_test, y_train, y_test = train_test_split(train[['day','month','year','n_killed','n_injured']], train['incident_url_fields_missing'], test_size = 0.3)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

drugTree = DecisionTreeClassifier(criterion="gini")
drugTree.fit(X_train,y_train)
predTree = drugTree.predict(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)

nbc = GaussianNB()
nbc.fit(X_train,y_train)
y_pred = nbc.predict(X_test)

from sklearn import metrics
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))
print(classification_report(y_test, predTree))
pd.DataFrame(
confusion_matrix(y_test, predTree),
columns=['Predicted No', 'Predicted Yes'],
index=['Actual No', 'Actual Yes']
)   
print("KNN's Accuracy: ", metrics.accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
pd.DataFrame(
confusion_matrix(y_test, pred),
columns=['Predicted No', 'Predicted Yes'],
index=['Actual No', 'Actual Yes']
)  
print("NB's Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
pd.DataFrame(
confusion_matrix(y_test, y_pred),
columns=['Predicted No', 'Predicted Yes'],
index=['Actual No', 'Actual Yes']
)
