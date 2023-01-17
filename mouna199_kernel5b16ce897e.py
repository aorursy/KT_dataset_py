# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
!pip install impyute
import impyute
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
gender_submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
test_raw = pd.read_csv("/kaggle/input/titanic/test.csv")
train_raw = pd.read_csv("/kaggle/input/titanic/train.csv")
train_raw.head()
train_raw.describe(include=[np.object])
train_raw.describe()
train_raw.isnull().sum()
def preprocess_data(data):
    data.drop(['Ticket','PassengerId'],axis=1,inplace=True)
    data.loc[data['Cabin'].isnull(), 'Cabin'] = 0
    data.loc[data['Cabin']!=0, 'Cabin'] = 1
    # We would like to predict age values using a method called mice, we should first remap categorical data into boolean
    gender_mapper = {'male': 0, 'female': 1}
    data['Sex'].replace(gender_mapper, inplace=True)
    # turn some variables into columns with dummies
    emb_dummies = pd.get_dummies(data['Embarked'], drop_first=True, prefix='Embarked')
    data = pd.concat([data, emb_dummies], axis=1)
    data.drop('Embarked', axis=1, inplace=True)
    # Same for this variable
    emb_dummies = pd.get_dummies(data['Pclass'], drop_first=True, prefix='Pclass')
    data = pd.concat([data, emb_dummies], axis=1)
    data.drop('Pclass', axis=1, inplace=True)
    # title of the person
    data['Title'] = data['Name'].apply(lambda x : x.split(",")[1].strip().split(' ')[0])
    data['Title_boolean']  = [ 0 if x in(['Mr.', 'Mrs.', 'Miss.', 'Mme.','Ms.']) else 1 for x in data['Title']]
    data.drop('Name', axis=1, inplace=True)
    data.drop('Title', axis=1, inplace=True)
    return data
train = preprocess_data(train_raw)
train[train.columns.difference(['Survived'])]
from fancyimpute import IterativeImputer
def fill_missing_values(data):
    MICE_imputer = IterativeImputer()
    data_MICE = data[data.columns.difference(['Survived'])].copy(deep=True)
    data_MICE.iloc[:, :] = MICE_imputer.fit_transform(data_MICE)
    return data_MICE
data_MICE = fill_missing_values(train)
train["Age"].plot(figsize=(16,6), kind="kde")
data_MICE['Age'].plot(figsize=(16,6), kind="kde")

train['Age'] = data_MICE

continuous_columns = ['Age',"SibSp","Parch",'Fare']
categorical_columns = ['Sex','Cabin','Embarked_Q','Embarked_S','Pclass_2','Pclass_3','Title_boolean']
train[continuous_columns].hist(grid=True, rwidth=0.9,
                   color='#607c8e', figsize=(10, 8))

for i, column in enumerate(categorical_columns,1):
    sns.catplot(kind='count',data=train,x=column,height=3, aspect=.7)
    plt.show()
from pandas.plotting import scatter_matrix
scatter_matrix(train[continuous_columns],figsize=(15,10))
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.preprocessing import LabelEncoder
array = train.values
X = array[:,1:]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(array[:,1])
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
test = preprocess_data(test_raw)
test_MICE = fill_missing_values(test)
test['Age']=test_MICE['Age']
test['Fare']=test_MICE['Fare']
X_test = test.values
# train different models
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    # Make predictions on validation dataset
    model.fit(X_train, Y_train)
    pred = model.predict(X_validation)
    predictions = [1 if m > 0.5 else 0 for m in pred]
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))
    test_pred  = model.predict(X_test)
    my_submission = pd.DataFrame(data={"PassengerId":test_raw.PassengerId,"Survived":test_pred})
    my_submission.to_csv('my_submission_model_{}.csv'.format(name),index=False)

