# Code here

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



# Import Semua Data

df = pd.read_csv('../input/tugas4/titanic.csv')

df_test = pd.read_csv('../input/tugas4/titanic_test.csv')
df.head()
df = pd.get_dummies(df,columns=['Sex','Embarked'])
import seaborn as sns



total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

f, ax = plt.subplots(figsize=(15, 6))

plt.xticks(rotation='90')

sns.barplot(x=missing_data.index, y=missing_data['Percent'])

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)

missing_data.head()
df['Age']=df['Age'].fillna(df['Age'].mean())
# Melihat hubungan antar data

X = df.iloc[:,0:14]  #independent columns

y = df.iloc[:,-1]    #target column i.e price range

#get correlations of each features in dataset

corrmat = df.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(8,8))

#plot heat map

g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
drop_elements = ['PassengerId', 'Name', 'Pclass', 'Ticket', 'Cabin', 'SibSp','Age']

df = df.drop(drop_elements, axis = 1)
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression



classifiers = [

    KNeighborsClassifier(3),

    SVC(probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression()]



log_cols = ["Classifier", "Accuracy"]

log = pd.DataFrame(columns=log_cols)





X = df.iloc[:, 1:]

y = df.iloc[:, 0]



acc_dict = {}



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)



for clf in classifiers:

    name = clf.__class__.__name__

    clf.fit(X_train, y_train)

    train_predictions = clf.predict(X_test)

    acc = accuracy_score(y_test, train_predictions)

    if name in acc_dict:

        acc_dict[name] += acc

    else:

        acc_dict[name] = acc



for clf in acc_dict:

    acc_dict[clf] = acc_dict[clf]

    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)

    log = log.append(log_entry)



plt.xlabel('Accuracy')

plt.title('Classifier Accuracy')



sns.set_color_codes("muted")

sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
from sklearn.model_selection import GridSearchCV

import numpy as np



clf = RandomForestClassifier()



param_grid = {'n_estimators':np.arange(5,60), 'criterion':['gini', 'entropy']}

gscv = GridSearchCV(clf, param_grid=param_grid, cv=5, scoring='roc_auc')



gscv.fit(X_train, y_train)
gscv.best_params_
total = df_test.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

f, ax = plt.subplots(figsize=(15, 6))

plt.xticks(rotation='90')

sns.barplot(x=missing_data.index, y=missing_data['Percent'])

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)

missing_data.head()
clf = RandomForestClassifier(criterion='entropy',n_estimators=16)

clf.fit(X,y)



df_pred = df_test.drop(drop_elements, axis = 1)



df_pred = pd.get_dummies(df_pred,columns=['Sex','Embarked'])

df_pred['Fare']=df_pred['Fare'].fillna(df_pred['Fare'].mean())



y_pred = clf.predict(df_pred)
submission = pd.DataFrame(df_test['PassengerId'])

submission['survived'] = y_pred
submission.to_csv('submissionDTC.csv', index=False)