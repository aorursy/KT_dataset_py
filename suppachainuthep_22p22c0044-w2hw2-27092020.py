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

import seaborn as sn

import matplotlib.pyplot as plt

import seaborn as sns
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_gender = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
df_train
df_test
df_gender
df_train.info()
len(df_train['PassengerId'].unique())
df_train['Sex'].unique()
df_train['SibSp'].unique()
df_train['Parch'].unique()
len(df_train['Ticket'].unique())
len(df_train['Cabin'].unique())
df_train['Embarked'].unique()
df_train.isna().sum()
df_train['Sex'].hist()

plt.suptitle("Sex")

plt.show()
df_train['SibSp'].hist()

plt.suptitle("SibSp")

plt.show()
survived = df_train['Survived'].unique()

for s in survived:

  s_data = (df_train['Survived'] == s)

  df_survived = df_train[s_data]

  df_survived['Embarked'].hist(alpha=0.4, bins=50)



plt.legend(survived)

plt.show()
columns_name = ['Sex','SibSp','Parch','Embarked']

fig, ax = plt.subplots(1, 4, figsize=(20, 18))

ax = ax.ravel() 



for index,col in enumerate(columns_name):

    ax[index].hist(np.array(df_train[col][df_train[col].notna()]), alpha=0.5,density=True)

    ax[index].set_title(col)
df_train['Sex'][df_train.Sex == 'female'] = 1

df_train['Sex'][df_train.Sex == 'male'] = 0
df_train['Sex'] = df_train['Sex'].astype(int)

df_train.info()
df_train.head()
sns.pairplot(df_train,vars=['Pclass','Sex','SibSp','Parch','Survived','Embarked'])
df_train.isna().sum()
df_train
df_one_hot = pd.get_dummies(df_train['Embarked'], prefix='Embarked')

df_one_hot
df_data = pd.concat([df_train,df_one_hot], axis=1)

df_data
df_data = df_data.drop(['PassengerId','Name','Ticket','Cabin','Embarked'],axis=1)
df_data
df_data['Age'] = df_data['Age'].fillna(df_data['Age'].mean())

df_data['Fare'] = df_data['Fare'].fillna(df_data['Fare'].mean())
df_data
df_data.isna().sum()
X = np.array(df_data.drop(['Survived'],axis=1))

Y = np.array(df_data['Survived'])
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn import tree

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import  cross_validate

from sklearn.metrics import classification_report, accuracy_score, make_scorer

from sklearn.neural_network import MLPClassifier



K = 5

list_report_class_0 = []

list_report_class_1 = []

def classification_report_with_average_f_measure(y_true, y_pred):

    report = classification_report(y_true, y_pred,output_dict=True)

    list_report_class_0.append(report['0'])

    list_report_class_1.append(report['1'])

    print(f"class 0     {report['0']}")

    print(f"class 1     {report['1']}")

    print()

    return accuracy_score(y_true, y_pred)



X = np.array(df_data.drop(['Survived'],axis=1))

Y = np.array(df_data['Survived'])



# Create Decision Tree classifer object

list_report_class_0 = []

list_report_class_1 = []

clf = DecisionTreeClassifier(criterion="entropy", max_depth=None)

cv_results  = cross_validate(clf, X, Y, cv=K,scoring=make_scorer(classification_report_with_average_f_measure))

df_class0 = pd.DataFrame.from_dict(list_report_class_0)

df_class1 = pd.DataFrame.from_dict(list_report_class_1)

#tree.plot_tree(clf)
print("Decision tree (average class0)")

print(df_class0.mean())
print("Decision tree (average class1)")

print(df_class1.mean())
gnb = GaussianNB()

list_report_class_0 = []

list_report_class_1 = []

cv_results  = cross_validate(gnb, X, Y, cv=K,scoring=make_scorer(classification_report_with_average_f_measure))

df_class0 = pd.DataFrame.from_dict(list_report_class_0)

df_class1 = pd.DataFrame.from_dict(list_report_class_1)
print("Naive bayes (average class0)")

print(df_class0.mean())
print("Naive bayes (average class1)")

print(df_class1.mean())
X.shape
mlp = MLPClassifier(random_state=1,activation='relu', max_iter=500)

list_report_class_0 = []

list_report_class_1 = []

cv_results  = cross_validate(mlp, X, Y, cv=K,scoring=make_scorer(classification_report_with_average_f_measure))

df_class0 = pd.DataFrame.from_dict(list_report_class_0)

df_class1 = pd.DataFrame.from_dict(list_report_class_1)
print("Neural Network (average class0)")

print(df_class0.mean())
print("Neural Network (average class1)")

print(df_class1.mean())