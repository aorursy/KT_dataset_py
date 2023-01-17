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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/heart-disease-dataset/heart.csv')
df.head()
df.shape
df.columns
df.describe()
df.isnull().sum()
print(df.info())
plt.figure(figsize=(20,10))

sns.heatmap(df.corr(), annot=True, cmap='terrain')
sns.pairplot(data=df)
df.hist(figsize=(12,12), layout=(5,3));
# box and whiskers plot

df.plot(kind='box', subplots=True, layout=(5,3), figsize=(12,12))

plt.show()
sns.catplot(data=df, x='sex', y='age',  hue='target', palette='husl')
sns.barplot(data=df, x='sex', y='chol', hue='target', palette='spring')
df['sex'].value_counts()
df['target'].value_counts()
df['thal'].value_counts()
sns.countplot(x='sex', data=df, palette='husl', hue='target')
sns.countplot(x='target',palette='BuGn', data=df)
sns.countplot(x='ca',hue='target',data=df)
df['ca'].value_counts()
sns.countplot(x='thal',data=df, hue='target', palette='BuPu' )
sns.countplot(x='thal', hue='sex',data=df, palette='terrain')
df['cp'].value_counts()  # chest pain type
sns.countplot(x='cp' ,hue='target', data=df, palette='rocket')
sns.countplot(x='cp', hue='sex',data=df, palette='BrBG')
sns.boxplot(x='sex', y='chol', hue='target', palette='seismic', data=df)
sns.barplot(x='sex', y='cp', hue='target',data=df, palette='cividis')
sns.barplot(x='sex', y='thal', data=df, hue='target', palette='nipy_spectral')
sns.barplot(x='target', y='ca', hue='sex', data=df, palette='mako')
sns.barplot(x='sex', y='oldpeak', hue='target', palette='rainbow', data=df)
df['fbs'].value_counts()
sns.barplot(x='fbs', y='chol', hue='target', data=df,palette='plasma' )
sns.barplot(x='sex',y='target', hue='fbs',data=df)
gen = pd.crosstab(df['sex'], df['target'])

print(gen)
gen.plot(kind='bar', stacked=True, color=['green','yellow'], grid=False)
temp=pd.crosstab(index=df['sex'],

            columns=[df['thal']], 

            margins=True)

temp
temp.plot(kind="bar",stacked=True)

plt.show()
temp=pd.crosstab(index=df['target'],

            columns=[df['thal']], 

            margins=True)

temp
temp.plot(kind='bar', stacked=True)

plt.show()
chest_pain = pd.crosstab(df['cp'], df['target'])

chest_pain
chest_pain.plot(kind='bar', stacked=True, color=['purple','blue'], grid=False)
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

StandardScaler = StandardScaler()  

columns_to_scale = ['age','trestbps','chol','thalach','oldpeak']

df[columns_to_scale] = StandardScaler.fit_transform(df[columns_to_scale])
df.head()
X= df.drop(['target'], axis=1)

y= df['target']
X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=40)


print('X_train-', X_train.size)

print('X_test-',X_test.size)

print('y_train-', y_train.size)

print('y_test-', y_test.size)
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()



model1=lr.fit(X_train,y_train)

prediction1=model1.predict(X_test)
from sklearn.metrics import confusion_matrix



cm=confusion_matrix(y_test,prediction1)

cm
sns.heatmap(cm, annot=True,cmap='BuPu')
TP=cm[0][0]

TN=cm[1][1]

FN=cm[1][0]

FP=cm[0][1]

print('Testing Accuracy:',(TP+TN)/(TP+TN+FN+FP))
from sklearn.metrics import accuracy_score

accuracy_score(y_test,prediction1)
from sklearn.metrics import classification_report

print(classification_report(y_test, prediction1))
from sklearn.tree import DecisionTreeClassifier



dtc=DecisionTreeClassifier()

model2=dtc.fit(X_train,y_train)

prediction2=model2.predict(X_test)

cm2= confusion_matrix(y_test,prediction2)
cm2
accuracy_score(y_test,prediction2)
print(classification_report(y_test, prediction2))
from sklearn.ensemble import RandomForestClassifier



rfc=RandomForestClassifier()

model3 = rfc.fit(X_train, y_train)

prediction3 = model3.predict(X_test)

confusion_matrix(y_test, prediction3)
accuracy_score(y_test, prediction3)
print(classification_report(y_test, prediction3))
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC



svm=SVC()

model4=svm.fit(X_train,y_train)

prediction4=model4.predict(X_test)

cm4= confusion_matrix(y_test,prediction4)
cm4
accuracy_score(y_test, prediction4)
from sklearn.naive_bayes import GaussianNB



NB = GaussianNB()

model5 = NB.fit(X_train, y_train)

prediction5 = model5.predict(X_test)

cm5= confusion_matrix(y_test, prediction5)
cm5
accuracy_score(y_test, prediction5)
print('cm4', cm4)

print('-----------')

print('cm5',cm5)
from sklearn.neighbors import KNeighborsClassifier



KNN = KNeighborsClassifier()

model6 = KNN.fit(X_train, y_train)

prediction6 = model6.predict(X_test)

cm6= confusion_matrix(y_test, prediction5)

cm6
print('KNN :', accuracy_score(y_test, prediction6))

print('lr :', accuracy_score(y_test, prediction1))

print('dtc :', accuracy_score(y_test, prediction2))

print('rfc :', accuracy_score(y_test, prediction3))

print('NB: ', accuracy_score(y_test, prediction4))

print('SVC :', accuracy_score(y_test, prediction5))