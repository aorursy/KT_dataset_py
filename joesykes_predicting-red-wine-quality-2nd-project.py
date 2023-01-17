import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
df.head()
df.info()
df.groupby('quality').mean()['fixed acidity'].plot(kind='bar')
plt.ylabel('Mean Fixed Acidity')
sns.lmplot(x='volatile acidity',y='quality',data=df)
df.groupby('quality').mean()['volatile acidity'].plot(kind='bar')
plt.ylabel('Mean Volatile Acidity')
df.groupby('quality').mean()['citric acid'].plot(kind='bar')
plt.ylabel('Mean Citric Acid Level')
df.groupby('quality').mean()['residual sugar'].plot(kind='bar')
plt.ylabel('Mean Residual Sugar Level')
df.groupby('quality').mean()['chlorides'].plot(kind='bar')
plt.ylabel('Mean Chloride Level')
df.groupby('quality').mean()['free sulfur dioxide'].plot(kind='bar')
plt.ylabel('Mean Free Sulfur Dioxide')
sns.scatterplot(x='free sulfur dioxide',y='quality',data=df)
df.groupby('quality').mean()['total sulfur dioxide'].plot(kind='bar')
plt.ylabel('Mean Total Sulfur Dioxide Level')
df.groupby('quality').mean()['density'].plot(kind='bar')
plt.ylabel('Mean Density')
plt.ylim(0.99,1)
df.groupby('quality').mean()['pH'].plot(kind='bar')
plt.ylabel('Mean pH Level')
df.groupby('quality').mean()['sulphates'].plot(kind='bar')
plt.ylabel('Mean Sulphate Level')
df.groupby('quality').mean()['alcohol'].plot(kind='bar')
plt.ylabel('Mean Alcohol Level')
series = pd.Series(df.corr()['quality'])
series.drop('quality').plot(kind='bar')
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),annot=True)
sns.scatterplot(x='fixed acidity',y='citric acid',hue='quality',data=df)
df['citric acid percentage'] = df['citric acid'] / (df['citric acid'] + df['fixed acidity'])
df.groupby('quality').mean()['citric acid percentage'].plot(kind='bar')
plt.ylabel('Mean Citric Acid Percentage')
sns.scatterplot(x='fixed acidity',y='density',hue='quality',data=df)
sns.scatterplot(x='fixed acidity',y='pH',hue='quality',data=df)
plt.figure(figsize=(10,6))
sns.scatterplot(x='volatile acidity',y='citric acid',hue='quality',data=df)
sns.scatterplot(x='citric acid',y='pH',hue='quality',data=df)
sns.scatterplot(x='residual sugar',y='density',hue='quality',data=df)
sns.scatterplot(x='chlorides',y='sulphates',hue='quality',data=df)
sns.scatterplot(x='free sulfur dioxide',y='total sulfur dioxide',hue='quality',data=df)
plt.plot(range(0,71),range(0,71))
df['free sulfur dioxide percentage'] = df['free sulfur dioxide'] / (df['free sulfur dioxide'] + df['total sulfur dioxide'])
df.groupby('quality').mean()['free sulfur dioxide percentage'].plot(kind='bar')
plt.ylabel('Mean Free Sulfur Dioxide Percentage')
df.corr()['quality']
sns.scatterplot(x='alcohol',y='density',hue='quality',data=df)
df = df.drop(['free sulfur dioxide','fixed acidity'],axis=1)
df.head()
sns.countplot(x='quality',data=df)
df.groupby('quality').count()['pH'] * 100 /len(df)
df['new_rating'] = df['quality'].apply(lambda x: 1 if x >= 6 else 0)
sns.countplot(x='new_rating',data=df)
df['new_rating'].value_counts() * 100 / len(df)
X = df.drop(['quality','new_rating'],axis=1)
y = df['new_rating']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(max_iter=1500)
logmodel.fit(X_train, y_train)
logmodelpreds = logmodel.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
print("Confusion Matrix: ")
print(confusion_matrix(y_test,logmodelpreds))
print("-" * 50)
print("Classification Report: ")
print(classification_report(y_test,logmodelpreds))
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
treepreds = tree.predict(X_test)

print("Confusion Matrix: ")
print(confusion_matrix(y_test,treepreds))
print("-" * 50)
print("Classification Report: ")
print(classification_report(y_test,treepreds))
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100)
forest.fit(X_train, y_train)
forestpreds = forest.predict(X_test)

print("Confusion Matrix: ")
print(confusion_matrix(y_test,forestpreds))
print("-" * 50)
print("Classification Report: ")
print(classification_report(y_test,forestpreds))
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
svcpreds = svc.predict(X_test)

print("Confusion Matrix: ")
print(confusion_matrix(y_test,svcpreds))
print("-" * 50)
print("Classification Report: ")
print(classification_report(y_test,svcpreds))
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train,y_train)
grid.best_estimator_
gridpreds = grid.predict(X_test)

print("Confusion Matrix: ")
print(confusion_matrix(y_test,gridpreds))
print("-" * 50)
print("Classification Report: ")
print(classification_report(y_test,gridpreds))
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
xgbpreds = xgb.predict(X_test)

print("Confusion Matrix: ")
print(confusion_matrix(y_test,xgbpreds))
print("-" * 50)
print("Classification Report: ")
print(classification_report(y_test,xgbpreds))
df = df.drop('new_rating',axis=1)
df.head()
df = df[['volatile acidity','citric acid','residual sugar','chlorides','total sulfur dioxide','density','pH','sulphates','alcohol','citric acid percentage','free sulfur dioxide percentage','quality']]
df.head(2)
data = df.values
X = data[:, :-1]
y = data[:, -1]
X_columns = df.columns[:-1]
y_columns = df.columns[-1]

from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X, y = oversample.fit_sample(X, y)

X_sampled = pd.DataFrame(X, columns=X_columns)
y_sampled = pd.DataFrame(y, columns=[y_columns])
df = pd.concat([X_sampled,y_sampled],axis=1)
sns.countplot(x='quality',data=df)
df['quality'].value_counts()
X = df.drop('quality',axis=1)
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
logmodel = LogisticRegression(max_iter=5000)
logmodel.fit(X_train, y_train)
logmodelpreds = logmodel.predict(X_test)

print("Confusion Matrix: ")
print(confusion_matrix(y_test,logmodelpreds))
print("-" * 50)
print("Classification Report: ")
print(classification_report(y_test,logmodelpreds))

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
treepreds = tree.predict(X_test)

print("Confusion Matrix: ")
print(confusion_matrix(y_test,treepreds))
print("-" * 50)
print("Classification Report: ")
print(classification_report(y_test,treepreds))
forest = RandomForestClassifier(n_estimators=100)
forest.fit(X_train, y_train)
forestpreds = forest.predict(X_test)

print("Confusion Matrix: ")
print(confusion_matrix(y_test,forestpreds))
print("-" * 50)
print("Classification Report: ")
print(classification_report(y_test,forestpreds))
svc = SVC()
svc.fit(X_train, y_train)
svcpreds = svc.predict(X_test)

print("Confusion Matrix: ")
print(confusion_matrix(y_test,svcpreds))
print("-" * 50)
print("Classification Report: ")
print(classification_report(y_test,svcpreds))
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train,y_train)

grid.best_estimator_
gridpreds = grid.predict(X_test)

print("Confusion Matrix: ")
print(confusion_matrix(y_test,gridpreds))
print("-" * 50)
print("Classification Report: ")
print(classification_report(y_test,gridpreds))
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
xgbpreds = xgb.predict(X_test)

print("Confusion Matrix: ")
print(confusion_matrix(y_test,xgbpreds))
print("-" * 50)
print("Classification Report: ")
print(classification_report(y_test,xgbpreds))
