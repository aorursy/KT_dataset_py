# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns





import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC
df_train_original = pd.read_csv("/kaggle/input/titanic/train.csv")

df_test_original = pd.read_csv("/kaggle/input/titanic/test.csv")
df_train = df_train_original.copy(deep = True)

df_test = df_test_original.copy(deep = True)
#Checking lenght of data frame

print(len(df_train))

print(len(df_test))
#checking number of variables

print(len(df_train.columns))

print(len(df_test.columns))
#Getting to know the columns

df_train.columns
# Taking a peek at the data

df_train.head(5)
df_train.describe()
df_test.describe()
Y = df_train.Survived

X = df_train.drop('Survived', axis =1)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25)
print('Variable type: ', x_train.PassengerId.dtype)
#Checking if all PassengerId are Uinque

print('Are passengerId unique for x_train: ', len(np.unique(x_train.PassengerId)) == len(x_train))

print('Are passengerId unique for x_test: ', len(np.unique(x_test.PassengerId)) == len(x_test))

print('Are passengerId unique for df_test: ', len(np.unique(df_test.PassengerId)) == len(df_test))
print("Number of null values in x_train = ",  x_train.PassengerId.isnull().sum())

print("Number of null values in x_test = ",  x_test.PassengerId.isnull().sum())

print("Number of null values in df_test = ",  df_test.PassengerId.isnull().sum())
print("Variable type: ", x_train.Pclass.dtype)
print("Number of Null values in x_train: ", x_train.Pclass.isnull().sum())

print("Number of Null values in x_test: ", x_test.Pclass.isnull().sum())

print("Number of Null values in df_test: ", df_test.Pclass.isnull().sum())
print("Mean of Pclass for x_train: ", x_train.Pclass.mean())

print("Mean of Pclass for x_test: ", x_test.Pclass.mean())

print("Mean of Pclass for df_test: ", df_test.Pclass.mean())
# lets get to know the distribution of the Pclass

sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(24,6)

sns.distplot(x_train.Pclass)
print("Variable type: ", x_train.Sex.dtype)
print("Number of Null values in x_train: ", x_train.Sex.isnull().sum())

print("Number of Null values in x_test: ", x_test.Sex.isnull().sum())

print("Number of Null values in df_test: ", df_test.Sex.isnull().sum())
x_train.Sex.describe()
x_test.Sex.describe()
df_test.Sex.describe()
objects = ('Male', 'Female')

y_pos = np.arange(len(objects))

performance = [((x_train['Sex'] == 'male').sum()), ((x_train['Sex'] == 'female').sum())]



plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.ylabel('Count')

plt.title('Gender Ratio')



plt.show()
print("Variable type: ", x_train.Age.dtype)
print("Number of Null values in x_train: ", x_train.Age.isnull().sum())

print("Number of Null values in x_test: ", x_test.Age.isnull().sum())

print("Number of Null values in df_test: ", df_test.Age.isnull().sum())
x_train.fillna(x_train.mean(skipna = True), inplace=True)

x_test.fillna(x_train.mean(skipna = True), inplace=True)

df_test.fillna(x_train.mean(skipna = True), inplace=True)
print("Checking if all the Ages are in whole number in x_train", ((x_train.Age % 1) != 0).sum())

print("Checking if all the Ages are in whole number in x_test", ((x_test.Age % 1) != 0).sum())

print("Checking if all the Ages are in whole number in df_test", ((df_test.Age % 1) != 0).sum())
x_train.Age = (round(x_train.Age)).astype(int)

x_test.Age = (round(x_test.Age)).astype(int)

df_test.Age = (round(df_test.Age)).astype(int)
print("Mean of Age for x_train: ", x_train.Age.mean())

print("Mean of Age for x_test: ", x_test.Age.mean())

print("Mean of Age for df_test: ", df_test.Age.mean())
# lets get to know the distribution of the Age

sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(24,6)

sns.distplot(x_train.Age)
print("Max Age x_train: ", x_train.Age.max())

print("Max Age x_test: ", x_test.Age.max())

print("Max Age df_test: ", df_test.Age.max())
print("Min Age x_train: ", x_train.Age.min())

print("Min Age x_test: ", x_test.Age.min())

print("Min Age df_test: ", df_test.Age.min())
x_train.Age.plot.box(vert = False)
x_train["Age Bracket"], b = pd.qcut(x_train["Age"], q=5, labels=[1,2,3,4,5], retbins=True)

x_test["Age Bracket"] = pd.cut(x_test["Age"], bins=b, labels=[1,2,3,4,5], include_lowest=True)

df_test["Age Bracket"] = pd.cut(df_test["Age"], bins=b, labels=[1,2,3,4,5], include_lowest=True)
x_train.head()
print("Variable type: ", x_train.SibSp.dtype)
print("Number of Null values in x_train: ", x_train.SibSp.isnull().sum())

print("Number of Null values in x_test: ", x_test.SibSp.isnull().sum())

print("Number of Null values in df_test: ", df_test.SibSp.isnull().sum())
print('Mean of x_train SibSp in x_train:', x_train.SibSp.mean())

print('Mean of x_train SibSp in x_test:', x_test.SibSp.mean())

print('Mean of x_train SibSp in df_test:', df_test.SibSp.mean())
print('Min of x_train SibSp in x_train:', x_train.SibSp.min())

print('Min of x_train SibSp in x_test:', x_test.SibSp.min())

print('Min of x_train SibSp in df_test:', df_test.SibSp.min())
print('Max of x_train SibSp in x_train:', x_train.SibSp.max())

print('Max of x_train SibSp in x_test:', x_test.SibSp.max())

print('Max of x_train SibSp in df_test:', df_test.SibSp.max())
sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(24,6)

sns.distplot(x_train.SibSp)
print("Variable type: ", x_train.Parch.dtype)
print("Number of Null values in x_train: ", x_train.SibSp.isnull().sum())

print("Number of Null values in x_test: ", x_test.SibSp.isnull().sum())

print("Number of Null values in df_test: ", df_test.SibSp.isnull().sum())
print("Mean of x_train: ", x_train.Parch.mean())

print("Mean of x_test: ", x_test.Parch.mean())

print("Mean of df_test: ", df_test.Parch.mean())
print('Min X_train: ', x_train.Parch.min())

print('Min X_test: ', x_test.Parch.min())

print('Min df_test: ', df_test.Parch.min())
print('Max X_train: ', x_train.Parch.max())

print('Max X_test: ', x_test.Parch.max())

print('Max df_test: ', df_test.Parch.max())
sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(24,6)

sns.distplot(x_train.Parch)
print("Variable type: ", x_train.Ticket.dtype)
print("Number of Null values in x_train: ", x_train.Ticket.isnull().sum())

print("Number of Null values in x_test: ", x_test.Ticket.isnull().sum())

print("Number of Null values in df_test: ", df_test.Ticket.isnull().sum())
print("Variable type: ", x_train.Cabin.dtype)
print("Number of Null values in x_train: ", x_train.Cabin.isnull().sum())

print("Number of Null values in x_test: ", x_test.Cabin.isnull().sum())

print("Number of Null values in df_test: ", df_test.Cabin.isnull().sum())
print("Unique:", x_train.Cabin.unique())
x_train['Cabin_Series'] = x_train.Cabin

x_train['Cabin_Series'] = x_train.Cabin_Series.fillna('N')

x_train['Cabin_Series'] = x_train.Cabin_Series.str[0]



x_test['Cabin_Series'] = x_test.Cabin

x_test['Cabin_Series'] = x_test.Cabin_Series.fillna('N')

x_test['Cabin_Series'] = x_test.Cabin_Series.str[0]





df_test['Cabin_Series'] = df_test.Cabin

df_test['Cabin_Series'] = df_test.Cabin_Series.fillna('N')

df_test['Cabin_Series'] = df_test.Cabin_Series.str[0]
((x_train['Cabin_Series'] == 'N').sum())
objects = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'N')

y_pos = np.arange(len(objects))

performance = [((x_train['Cabin_Series'] == 'A').sum()), ((x_train['Cabin_Series'] == 'B').sum()), \

              ((x_train['Cabin_Series'] == 'C').sum()), ((x_train['Cabin_Series'] == 'D').sum()), \

              ((x_train['Cabin_Series'] == 'E').sum()), ((x_train['Cabin_Series'] == 'F').sum()), \

              ((x_train['Cabin_Series'] == 'G').sum()), ((x_train['Cabin_Series'] == 'T').sum()), \

              ((x_train['Cabin_Series'] == 'N').sum())]



plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.ylabel('Count')

plt.title('Cabin Series')



plt.show()
objects = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'T')

y_pos = np.arange(len(objects))

performance = [((x_train['Cabin_Series'] == 'A').sum()), ((x_train['Cabin_Series'] == 'B').sum()), \

              ((x_train['Cabin_Series'] == 'C').sum()), ((x_train['Cabin_Series'] == 'D').sum()), \

              ((x_train['Cabin_Series'] == 'E').sum()), ((x_train['Cabin_Series'] == 'F').sum()), \

              ((x_train['Cabin_Series'] == 'G').sum()), ((x_train['Cabin_Series'] == 'T').sum())]



plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.ylabel('Count')

plt.title('Cabin Series')



plt.show()
print("Variable type: ", x_train.Fare.dtype)
print("Number of Null values in x_train: ", x_train.Fare.isnull().sum())

print("Number of Null values in x_test: ", x_test.Fare.isnull().sum())

print("Number of Null values in df_test: ", df_test.Fare.isnull().sum())
x_train["Fare_Bin"], b = pd.qcut(x_train["Fare"], q = 3, labels=[1,2,3], retbins=True)

x_test["Fare_Bin"] = pd.cut(x_test["Fare"], bins = b, labels=[1,2,3], include_lowest=True)

df_test["Fare_Bin"] = pd.cut(df_test["Fare"], bins= b, labels=[1,2,3], include_lowest=True)
print("Variable type: ", x_train.Embarked.dtype)
print("Number of Null values in x_train: ", x_train.Embarked.isnull().sum())

print("Number of Null values in x_test: ", x_test.Embarked.isnull().sum())

print("Number of Null values in df_test: ", df_test.Embarked.isnull().sum())
x_train.groupby('Embarked').agg("count").reset_index()
x_train.Embarked.fillna("S", inplace = True)

x_test.Embarked.fillna("S", inplace = True)

df_test.Embarked.fillna("S", inplace = True)
objects = ('C', 'Q', 'S')

y_pos = np.arange(len(objects))

performance = [((x_train['Embarked'] == 'C').sum()), ((x_train['Embarked'] == 'Q').sum()), \

              ((x_train['Embarked'] == 'S').sum())]



plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.ylabel('Count')

plt.title('Embarked')



plt.show()
# verifing no missing values are remaining in train and test data

print("No. of null values left in x_train: ", x_train.isnull().sum().sum())

print("No. of null values leftin x_test: ", x_test.isnull().sum().sum())

print("No. of null values left in df_test: ", df_test.isnull().sum().sum())
x_train["Family_size"] = x_train.SibSp + x_train.Parch

x_train["Is_Alone"] = 1

x_train["Is_Alone"][x_train["Family_size"] > 0] = 0

x_train.head()
x_test["Family_size"] = x_test.SibSp + x_test.Parch

x_test["Is_Alone"] = 1

x_test["Is_Alone"][x_test["Family_size"] > 0] = 0



df_test["Family_size"] = df_test.SibSp + df_test.Parch

df_test["Is_Alone"] = 1

df_test["Is_Alone"][df_test["Family_size"] > 0] = 0
import re

# Define function to extract titles from passenger names

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""



# Create a new feature Title, containing the titles of passenger names

x_train['Title'] = x_train['Name'].apply(get_title)

x_test['Title'] = x_test['Name'].apply(get_title)

df_test['Title'] = df_test['Name'].apply(get_title)



full_data = [x_train,x_test, df_test]

# Group all non-common titles into one single grouping "Rare"

for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
x_train.sample(5)
print("Null values in train", x_train.isnull().sum())

print("Null values in test", x_test.isnull().sum())

print("Null values in df_test", df_test.isnull().sum())
# MinMax Scaling

#scaler = MinMaxScaler()

#x_train = scaler.fit_transform(x_train)

#x_test = scaler.transform(x_test)

#df_test = scaler.transform(df_test)

x_train.sample()
df_test["Cabin_Series"].unique()
x_test["Cabin_Series"].unique()
Selcol = ['Pclass', 'Sex', 'Is_Alone', 'Fare_Bin', 'Parch', 'Embarked', 'Age Bracket', 'Cabin_Series', 'Title']
# label encoder

le = LabelEncoder()

obj_columns = [col for col in x_train[Selcol].select_dtypes(include = ['object'])]
# applying label encoder

for col in obj_columns:

    x_train[col] = le.fit_transform(x_train[col])

    x_test[col] = le.transform(x_test[col])

    df_test[col] = le.transform(df_test[col])
grid_values = {'C':[5], 'penalty':['l2']}

log_clf = LogisticRegression()

log_grid = GridSearchCV(log_clf, param_grid = grid_values, scoring = 'roc_auc')

log_grid.fit(x_train[Selcol], y_train)

print('Accurary of Logistic Regression Classifier on train_x: {:.3f}' .format(log_grid.score(x_train[Selcol], y_train)))

print('Accurary of Logistic Regression Classifier on devl_x: {:.3f}' .format(log_grid.score(x_test[Selcol], y_test)))



print('Grid best parameter (max. accuary): ', log_grid.best_params_)

print('Grid best score (accuary):', log_grid.best_score_)
grid_values = {'max_leaf_nodes': [29,30,31], "max_features": [5,6,7,8,9]} #

df_clf = DecisionTreeClassifier(min_samples_split=0.085)

dt_grid = GridSearchCV(df_clf, param_grid = grid_values, scoring = 'roc_auc', cv=10)

dt_grid.fit(x_train[Selcol], y_train)

print('Accurary of Decision Tree Classifier on train_x: {:.3f}' .format(dt_grid.score(x_train[Selcol], y_train)))

print('Accurary of Decision Tree Regression Classifier on devl_x: {:.3f}' .format(dt_grid.score(x_test[Selcol], y_test)))



print('Grid best parameter (max. accuary): ', dt_grid.best_params_)

print('Grid best score (accuary):', dt_grid.best_score_)
df_test['Survived'] = svc_grid.predict(df_test[Selcol])
submit = df_test[['PassengerId','Survived']]

submit.to_csv("../working/submit.csv", index=False)



print('Validation Data Distribution: \n', df_test['Survived'].value_counts(normalize = True))

submit.sample(10)

grid_values = {'gamma': [0.001, 0.03, 1, 10, 300]}

svc_clf = SVC(kernel = 'rbf')

svc_grid = GridSearchCV(svc_clf, param_grid = grid_values, scoring = 'roc_auc')

svc_grid.fit(x_train[Selcol], y_train)

print('Accurary of Support Vector Classifier on train_x: {:.3f}' .format(svc_grid.score(x_train[Selcol], y_train)))

print('Accurary of Support Vector Classifier on devl_x: {:.3f}' .format(svc_grid.score(x_test[Selcol], y_test)))



print('Grid best parameter (max. accuary): ', svc_grid.best_params_)

print('Grid best score (accuary):', svc_grid.best_score_)
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier().fit(x_train[Selcol], y_train)

print('Accurary of Random Forest Classifier on train_x: {:.3f}' .format(roc_auc_score(y_train, rf_clf.predict_proba(x_train[Selcol])[:,1])))

print('Accurary of Random Forest Classifier on devl_x: {:.3f}' .format(roc_auc_score(y_test, rf_clf.predict_proba(x_test[Selcol])[:,1])))
from sklearn.naive_bayes import GaussianNB

nb_clf = GaussianNB().fit(x_train[Selcol], y_train)

print('Accurary of GaussianNB Classifier on train_x: {:.3f}' .format(roc_auc_score(y_train, nb_clf.predict_proba(x_train[Selcol])[:,1])))

print('Accurary of GaussianNB Classifier on devl_x: {:.3f}' .format(roc_auc_score(y_test, nb_clf.predict_proba(x_test[Selcol])[:,1])))
from sklearn.ensemble import GradientBoostingClassifier



grid_values = {'max_features': [9,'log2'], 'min_samples_leaf': [10,20,30], 'max_leaf_nodes': [2,3]}

xgb_clf = GradientBoostingClassifier(n_estimators = 440, random_state=9, max_features =3, learning_rate=0.1, max_depth=3)



xgb_grid = GridSearchCV(xgb_clf, param_grid = grid_values, scoring = 'roc_auc', cv=10)

xgb_grid.fit(x_train[Selcol], y_train)

print('Accurary of Gradient Boosting Classifier on train_x: {:.3f}' .format(xgb_grid.score(x_train[Selcol], y_train)))

print('Accurary of Gradient Boosting Tree Regression Classifier on devl_x: {:.3f}' .format(xgb_grid.score(x_test[Selcol], y_test)))



print('Grid best parameter (max. accuary): ', xgb_grid.best_params_)

print('Grid best score (accuary):', xgb_grid.best_score_)
from sklearn.neural_network import MLPClassifier

#nn_clf = MLPClassifier(hidden_layer_sizes = [20,12,7], solver='lbfgs', random_state = 0).fit(train_x, train_y)

nn_clf = MLPClassifier(solver='adam', activation = 'relu',alpha = 0.3,

                         hidden_layer_sizes = [7,7,7],

                         random_state = 0).fit(x_train[Selcol], y_train)

print('Accurary of NN Classifier on train_x: {:.3f}' .format(roc_auc_score(y_train, nn_clf.predict_proba(x_train[Selcol])[:,1])))

print('Accurary of NN Classifier on devl_x: {:.3f}' .format(roc_auc_score(y_test, nn_clf.predict_proba(x_test[Selcol])[:,1])))