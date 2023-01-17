import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/adult-census-income/adult.csv')
df.head()
df.columns
df.shape
df.info()
df.describe()
df.isnull().values.any()
df.isin(['?']).sum()
df = df.replace('?', np.NaN)
for col in ['workclass', 'occupation', 'native.country']:

    df[col].fillna(df[col].mode()[0], inplace=True)
df.isnull().sum()
df.head()
df['income'].value_counts()
sns.countplot(x='income', data = df)
sns.boxplot(y='age',x='income',data=df)
sns.boxplot(y='hours.per.week',x='income',data=df)
sns.countplot(df['sex'],hue=df['income'])
sns.countplot(df['occupation'],hue=df['income'])

plt.xticks(rotation=90)
df['income']=df['income'].map({'<=50K': 0, '>50K': 1})
sns.barplot(x="education.num",y="income",data=df)
df['workclass'].unique()
sns.barplot(x="workclass",y="income",data=df)

plt.xticks(rotation=90)
df['education'].unique()
sns.barplot(x="education",y="income",data=df)

plt.xticks(rotation=90)
df['marital.status'].unique()
sns.barplot(x="marital.status",y="income",data=df)

plt.xticks(rotation=90)
df['relationship'].unique()
sns.barplot(x="relationship",y="income",data=df)

plt.xticks(rotation=90)
df['native.country'].unique()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in df.columns:

    if df[col].dtypes == 'object':

        df[col] = le.fit_transform(df[col])
df.dtypes
df.head()
corrmat = df.corr()

plt.figure(figsize=(20,12))

sns.heatmap(corrmat, annot=True, cmap='coolwarm')
corrmat['income'].sort_values(ascending = False)
X = df.iloc[:,0:-1]

y = df.iloc[:,-1]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = pd.DataFrame(sc.fit_transform(X_train))

X_test = pd.DataFrame(sc.transform(X_test))
X_train.head()
l=[]
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

model = LogisticRegression()

model.fit(X_train, y_train)



y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print('Logistic Regression:', acc * 100)

l.append(acc)
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

classifier = SVC(kernel = 'rbf', random_state = 42)

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print('SVM:', acc * 100)

l.append(acc)
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print('Knn:',acc * 100)

l.append(acc)
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print('Decision Tree:', acc * 100)

l.append(acc)
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print('Naive Bayes:', acc * 100)

l.append(acc)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix as cm

from sklearn.metrics import classification_report as cr

classifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print('Random Forest:',acc * 100)

l.append(acc)

print(cm(y_test, y_pred))

print(cr(y_test, y_pred))
y_axis=['Logistic Regression',

     'Support Vector Classifier',

        'K-Neighbors Classifier',

      'Decision Tree Classifier',

       'Gaussian Naive Bayes',

      'Random Forest Classifier']

x_axis=l

sns.barplot(x=x_axis,y=y_axis)

plt.xlabel('Accuracy')