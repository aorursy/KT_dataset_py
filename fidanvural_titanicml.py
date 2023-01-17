# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # visualization

plt.style.use("seaborn-whitegrid")

import seaborn as sns # visualization

from collections import Counter



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df=pd.read_csv("/kaggle/input/titanic/train.csv")

test_df=pd.read_csv("/kaggle/input/titanic/test.csv")
train_df.head(10)
test_df.head()
train_df.columns
train_df.describe()
train_df.info()
def outlier_detection(df,features):

    outlier_indices=[]

    

    for i in features:

        # 1st quartile

        Q1=np.percentile(df[i],25)

        # 3rd quartile

        Q3=np.percentile(df[i],75)

        # IQR

        IQR=Q3-Q1

        

        min=Q1 - (IQR * 1.5)

        max=Q3 + (IQR * 1.5)

        

        outlier_list=df[(df[i] < min) | (df[i] > max)].index

        outlier_indices.extend(outlier_list)

        

    outlier_indices=Counter(outlier_list) # Counter returns dictionary

    

    return outlier_indices
train_df.loc[outlier_detection(train_df,["Age","SibSp","Parch","Fare"])]
# drop outliers

train_df.drop(outlier_detection(train_df,["Age","SibSp","Parch","Fare"]),inplace=True)
train_df.head()
train_data=train_df.drop(["Name","PassengerId","Ticket","Cabin"],axis=1)

train_data.head()
test_data=test_df.drop(["Name","PassengerId","Ticket","Cabin"],axis=1)

test_data.head()
train_data.isnull().any()
train_data["Age"].fillna(train_data["Age"].mean(),inplace=True)
train_data.isnull().any()
test_data.head()
test_data.isnull().any()
test_data["Fare"].fillna(test_data["Fare"].mean(),inplace=True)
test_data.isnull().any()
test_data["Age"].fillna(test_data["Age"].mean(),inplace=True)
test_data.isnull().any()
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

train_data["Sex"]=le.fit_transform(train_data["Sex"])

train_data["Embarked"]=le.fit_transform(train_data["Embarked"])

test_data["Sex"]=le.fit_transform(test_data["Sex"])

test_data["Embarked"]=le.fit_transform(test_data["Embarked"])

train_data.head()
test_data.head()
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
x_train=train_data.drop("Survived", axis = 1)

y_train=train_data["Survived"]



x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size = 0.33)
lr=LogisticRegression()

lr.fit(x_train,y_train)

acc_lr_train=lr.score(x_train,y_train)

acc_lr_test=lr.score(x_test,y_test)

print("Train accuracy: {}".format(acc_lr_train))

print("Test accuracy: {}".format(acc_lr_test))
svm=SVC(kernel='rbf',gamma='auto')

svm.fit(x_train,y_train)

acc_svm_train=svm.score(x_train,y_train)

acc_svm_test=svm.score(x_test,y_test)

print("Train accuracy: {}".format(acc_svm_train))

print("Test accuracy: {}".format(acc_svm_test))
tree=DecisionTreeClassifier(max_depth=10)

tree.fit(x_train,y_train)

acc_tree_train=tree.score(x_train,y_train)

acc_tree_test=tree.score(x_test,y_test)

print("Train accuracy: {}".format(acc_tree_train))

print("Test accuracy: {}".format(acc_tree_test))
rf=RandomForestClassifier(n_estimators=100)

rf.fit(x_train,y_train)

acc_rf_train=rf.score(x_train,y_train)

acc_rf_test=rf.score(x_test,y_test)

print("Train accuracy: {}".format(acc_rf_train))

print("Test accuracy: {}".format(acc_rf_test))
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=100,max_depth=5)

rf.fit(x_train,y_train)



pred=rf.predict(test_data)



submission = pd.DataFrame({

        "PassengerId": test_df.PassengerId,

        "Survived": pred

    })

submission.to_csv("my_submission",index=False)

print("Your submission was successfully saved!")