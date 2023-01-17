import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings("ignore")



df = pd.read_csv('../input/heart.csv')
df.shape
df.columns
df.info()
df.head()
plt.figure(figsize=(10,8))

sns.heatmap(df.corr(),annot=True,linewidths=2,)
sns.countplot(x='target',data=df,hue='sex')

plt.legend(['Female','Male'])

plt.xlabel('-ve Heart Diagnosis                   +ve Heart Diagnosis')
sns.pairplot(df,hue='target')
sns.distplot(df['age'],)
df['age'].describe()


df['Age_Category']= pd.cut(df['age'],bins=list(np.arange(25, 85, 5)))
plt.figure(figsize=(20,5))



plt.subplot(121)

df[df['target']==1].groupby('Age_Category')['age'].count().plot(kind='bar')

plt.title('Age Distribution of Patients with +ve Heart Diagonsis')



plt.subplot(122)

df[df['target']==0].groupby('Age_Category')['age'].count().plot(kind='bar')

plt.title('Age Distribution of Patients with -ve Heart Diagonsis')
df.nunique()
plt.figure(figsize=(6,5))

sns.countplot(x='cp',data=df,hue='target')

plt.xlabel('typical angina     atypical angina     non-anginal pain     asymptomatic')

plt.ylabel('Counts of Chest pain Type')
plt.figure(figsize=(7,5))

sns.countplot(x='fbs',data=df,hue='target')

plt.xlabel('fasting blood sugar < 120 mg/dl        fasting blood sugar > 120 mg/dl')

plt.ylabel('Count')
plt.figure(figsize=(5,5))

sns.countplot(x='exang',data=df,hue='target')

plt.xlabel('No                                       Yes')

plt.ylabel('Count')

plt.title('Exercise induced angina')
df = pd.get_dummies(df,columns=['sex','cp','fbs','restecg','exang','slope','ca','thal','Age_Category'])

df.shape
df.columns
df.drop(['age'],axis=1,inplace=True)
df.shape
y = df['target']

X = df.drop(['target'],axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=5)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

from sklearn.tree import DecisionTreeClassifier

trees = DecisionTreeClassifier()
trees.fit(X_train,y_train)

y_pred = trees.predict(X_test)
print(classification_report(y_test,y_pred))

print(accuracy_score(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=750)

rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test) 
print(classification_report(y_test,y_pred))

print(accuracy_score(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))
