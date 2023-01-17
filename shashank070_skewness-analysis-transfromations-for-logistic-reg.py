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
import matplotlib.pyplot as plt
data=pd.read_csv('/kaggle/input/heart-disease/heart.csv')
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
f, ax = plt.subplots(figsize=(15, 6))
plt.xticks(rotation='90')
sns.barplot(x=missing_data.index, y=missing_data['Percent'])
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
data
cols=['trestbps','chol','thalach','oldpeak']
for i in data[cols]:
    print("column",i,"skewvalue",data[i].skew(axis=0))
    plt.figure()
    sns.distplot(data[i])


df=pd.read_csv('/kaggle/input/heart-disease/heart.csv')
df['chol_transformed']=np.log(df['chol']+1)
df['chol_transformed']=df['chol_transformed']/df['chol_transformed'].max()
cols=['chol','chol_transformed']
for i in df[cols]:
    print("column",i,"skewvalue",df[i].skew(axis=0))
    plt.figure()
    sns.distplot(df[i])
# df=pd.read_csv('/kaggle/input/heart-disease/heart.csv')
df['oldpeak_transformed']=np.reciprocal(df['oldpeak']+1)
df['oldpeak_transformed']=df['oldpeak_transformed']/df['oldpeak_transformed'].max()
cols=['oldpeak','oldpeak_transformed']
for i in df[cols]:
    print("column",i,"skewvalue",df[i].skew(axis=0))
    plt.figure()
    sns.distplot(df[i])
# df=pd.read_csv('/kaggle/input/heart-disease/heart.csv')
df['thalach_transformed']=np.power(df['thalach'],3)
df['thalach_transformed']=df['thalach_transformed']/df['thalach_transformed'].max()
cols=['thalach_transformed','thalach']
for i in df[cols]:
    print("column",i,"skewvalue",df[i].skew(axis=0))
    plt.figure()
    sns.distplot(df[i])
# df=pd.read_csv('/kaggle/input/heart-disease/heart.csv')
df['trestbps_transformed']=np.log(df['trestbps']+1)
df['trestbps_transformed']=df['trestbps_transformed']/df['trestbps_transformed'].max()
cols=['trestbps_transformed','trestbps']
for i in df[cols]:
    print("column",i,"skewvalue",df[i].skew(axis=0))
    plt.figure()
    sns.distplot(df[i])
df.columns.tolist()

data
cols=['age','sex','cp','exang','ca','slope','thal','restecg','chol_transformed','oldpeak_transformed','thalach_transformed','trestbps_transformed','target']
dataset=df[cols]
dataset
values=dataset.values
X, y = values[:, :-1], values[:, -1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.4, random_state = 2, stratify = y)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import r2_score

rfc = LogisticRegression(random_state = 42 )
# accuracies = cross_val_score(rfc, X_train, y_train, cv=3)
rfc.fit(X_train,y_train)
# pred = rfc.predict(X_test)
print("Train Score:",np.mean(accuracies))
print("Test Score:",rfc.score(X_test,y_test))
print("The Accuracy Score is:", metrics.accuracy_score(y_test,pred))

