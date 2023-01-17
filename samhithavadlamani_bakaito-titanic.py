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
#Load data

train_data=pd.read_csv('/kaggle/input/titanic/train.csv')

print(train_data.head())

test_data=pd.read_csv('/kaggle/input/titanic/test.csv')

test_data.head()

#list out all features



labels=train_data["Survived"]

train_data=train_data.drop(["Survived"], axis=1)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

import seaborn as sns

import matplotlib.pyplot as plt



num_cols=[c for c in train_data.columns if train_data[c].dtypes in ["int64", "float64"]]

train_data[num_cols].hist(figsize=(20,20), xlabelsize=8, ylabelsize=8)

plt.show()

labels.hist(figsize=(20,20), xlabelsize=8, ylabelsize=8)

plt.show()
sns.heatmap(train_data.corr(), annot=True)


#add a new feature rel=Sibsp+Parch

train_data["rel"]=train_data["SibSp"]+train_data["Parch"]

test_data["rel"]=test_data["SibSp"]+test_data["Parch"]



#feature selection

selected_feat=["Pclass", "Sex", "rel", "Fare"]

X_train=pd.get_dummies(train_data[selected_feat])

X_test=pd.get_dummies(test_data[selected_feat])



#impute na values

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

X_train = my_imputer.fit_transform(X_train)

X_test = my_imputer.transform(X_test)

#Validation: splitting just training data for training and testing

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_train, labels, test_size=0.4) # 70% training and 30% test

clf=RandomForestClassifier(n_estimators=100, random_state=1)

clf.fit(x_train, y_train)

labels_pred=clf.predict(x_test)



#Accuracy

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_test,labels_pred)

accuracy

#Training and predicting total data

clf=RandomForestClassifier(n_estimators=100, random_state=1)

clf.fit(X_train, labels)

labels_pred=clf.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': labels_pred})

output.to_csv('my_submission.csv', index=False)