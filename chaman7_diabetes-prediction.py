import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.columns
df.head()
df.describe()
df.shape
df.isnull().values.any()
df.hist(figsize=(20,20))

#before preprocessing
df.groupby('Outcome').size()
sns.countplot(x='Outcome', data=df)
df.plot(kind='box', figsize=(20,10))

plt.show()
df = df[df['SkinThickness'] < 80]

df = df[df['Insulin'] <= 600]

df.shape
corrmat = df.corr()

plt.figure(figsize=(20,10))

sns.heatmap(corrmat, annot=True, cmap='coolwarm')
df.corr()
print("total number of rows : {0}".format(len(df)))

print("number of missing pregnancies: {0}".format(len(df.loc[df['Pregnancies'] == 0])))

print("number of missing glucose: {0}".format(len(df.loc[df['Glucose'] == 0])))

print("number of missing bp: {0}".format(len(df.loc[df['BloodPressure'] == 0])))

print("number of missing skinthickness: {0}".format(len(df.loc[df['SkinThickness'] == 0])))

print("number of missing insulin: {0}".format(len(df.loc[df['Insulin'] == 0])))

print("number of missing bmi: {0}".format(len(df.loc[df['BMI'] == 0])))

print("number of missing diabetespedigree: {0}".format(len(df.loc[df['DiabetesPedigreeFunction'] == 0])))

print("number of missing age: {0}".format(len(df.loc[df['Age'] == 0])))
df.loc[df['Insulin'] == 0, 'Insulin'] = df['Insulin'].mean() 

df.loc[df['Glucose'] == 0, 'Glucose'] = df['Glucose'].mean() 

df.loc[df['BMI'] == 0, 'BMI'] = df['BMI'].mean() 

df.loc[df['BloodPressure'] == 0, 'BloodPressure'] = df['BloodPressure'].mean() 

df.loc[df['SkinThickness'] == 0, 'SkinThickness'] = df['SkinThickness'].mean() 
df.head()
sns.pairplot(df, hue='Outcome')
df.hist(figsize=(20,20))

#after preprocessing
df = df/df.max()

df.head()
X = df.iloc[:,0:-1]

y = df.iloc[:,-1]

X.head(10)

#y.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
l=[]
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

classifier = SVC(kernel = 'linear', random_state = 42)

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print('SVM:', acc * 100)

l.append(acc)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, cmap='YlGnBu')

plt.title('Confusion Matrix')

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

model = LogisticRegression()

model.fit(X_train, y_train)



y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print('Logistic Regression:', acc * 100)

l.append(acc)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, cmap='YlGnBu')

plt.title('Confusion Matrix')

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
from sklearn.metrics import classification_report as cr

print(cr(y_test, y_pred))
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

classifier = RandomForestClassifier(n_estimators = 30, criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print('Random Forest:',acc * 100)

l.append(acc)
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print('Knn:',acc * 100)

l.append(acc)
l
y_axis=['Support Vector Classifier',

      'Logistic Regression',

      'Decision Tree Classifier',

       'Gaussian Naive Bayes',

      'Random Forest Classifier',

      'K-Neighbors Classifier']

x_axis=l

sns.barplot(x=x_axis,y=y_axis)

plt.xlabel('Accuracy')