# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import zscore

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, KFold, RepeatedStratifiedKFold

from sklearn.metrics import classification_report, confusion_matrix, plot_precision_recall_curve, plot_roc_curve, accuracy_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, Lasso, Ridge, ElasticNet

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier

from sklearn.neural_network import MLPClassifier



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_file= pd.read_csv('../input/titanic/train.csv')

train_file.head()
test_file= pd.read_csv('../input/titanic/test.csv')

test_file.head()
train_file.shape
test_file.shape
gender_submiss= pd.read_csv('../input/titanic/gender_submission.csv')

gender_submiss.head()
## Rearranging the column Survived to end of dataframe for easy representation of predictor

train_file= train_file[['PassengerId', 'Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Survived']]

train_file.head()
test_file.info()
train_file.info()
train_file= train_file.drop(["Name", "Ticket", "Cabin"], axis=1)

train_file.head()
train_file["Sex"]= train_file["Sex"].replace({"female":0, "male":1})

train_file.head()
train_file["Sex"]= train_file["Sex"].astype('int64')
train_file.dtypes
train_file.Parch.unique()
train_file.Pclass.unique()
train_file.SibSp.unique()
train_file.Embarked.unique()
train_file.isnull().sum()
sns.distplot(train_file["Age"])
train_file["Age"].describe()
train_file["Age"]= train_file["Age"].fillna(train_file["Age"].mean())

train_file["Age"].isnull().sum()
train_file.isnull().sum()
train_file.dropna(inplace=True)
train_file.isnull().sum()
train_file.shape
train_file.info()
train_file["Survived"]= train_file["Survived"].astype('category')

train_file["Survived"].dtypes
## Checking Data Distribution for Survived (0/1)

f, ax = plt.subplots(1,1, figsize=(6,4))

sns.countplot(x="Survived", data=train_file, hue='Survived')

total = float(len(train_file))

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(100*height/total),

            ha="center") 

plt.show()
plt.figure(figsize=(8,5))

corr= train_file.corr()

sns.heatmap(corr, annot=True, linewidths=1.5, cmap='RdYlGn')
train_file.info()
X= train_file[['Pclass','Sex','Age','SibSp','Parch','Fare']]

y= train_file.Survived.values
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=7)
pipelines = []

pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',LogisticRegression())])))

pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))

pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))

pipelines.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingClassifier())])))

pipelines.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestClassifier())])))

pipelines.append(('ScaledAda', Pipeline([('Scaler', StandardScaler()),('Ada', AdaBoostClassifier())])))

pipelines.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesClassifier())])))

pipelines.append(('ScaledXGB', Pipeline([('Scaler', StandardScaler()),('XGB', XGBClassifier())])))

pipelines.append(('ScaledMLP', Pipeline([('Scaler', StandardScaler()),('MLP', MLPClassifier())])))
results = []

names = []

for name, model in pipelines:

    kfold = KFold(n_splits=10, random_state=21)

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
# Algorithm comparison

fig = plt.figure(figsize=(18,5))

fig.suptitle('Model Selection by comparision')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
gradboost= GradientBoostingClassifier(n_estimators=300, random_state=0).fit(X_train, y_train)

preds= gradboost.predict(X_test)

print('Classification report: \n')

print(classification_report(y_test, preds))
print('Accuracy of model:', accuracy_score(preds,y_test))
print('Training Score:', gradboost.score(X_train, y_train))

print('Test Score:', gradboost.score(X_test, y_test))
test_file.head()
test_data= test_file[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare']]

test_data.head()
test_data["Sex"]= test_data["Sex"].replace({"female":0, "male":1})

test_data.head()
test_data.isnull().sum()
test_data["Age"].describe()
test_data["Age"]= test_data["Age"].fillna(test_data["Age"].mean())

test_data.Age.isnull().sum()
test_data.isnull().sum()
sns.distplot(test_data["Fare"])
#test_data.dropna(inplace=True)

test_data["Fare"]= test_data["Fare"].fillna(test_data["Fare"].median())

test_data.isnull().sum()
test_final = test_data[['Pclass','Sex','Age','SibSp','Parch','Fare']]
test_preds= gradboost.predict(test_final)
passenger= test_data.PassengerId.values
preds_df= pd.DataFrame(passenger, columns=['PassengerId'])

preds_df['Survived']=test_preds
preds_df.head()
preds_df.shape
preds_df.to_csv('/kaggle/working/Titanic_Submission.csv', index=False)