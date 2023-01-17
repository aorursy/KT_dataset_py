!pip install datasciencehelper # pip install Data Science Helper
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import  GridSpec
import DataScienceHelper as dsh
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")
train.head(10)
train = train.drop(["Cabin","Name","PassengerId","Ticket"], axis = 1)
train.describe()
train.info()
train = dsh.fill_nan_numeric(train, list(train._get_numeric_data().keys()))
train = dsh.fill_nan_categorical(train, list(train.select_dtypes(include='object').keys()))
train.info()
S = train[train["Survived"] == 1]
SN = train[train["Survived"] == 0]
plt.figure(figsize = (13,8))
sns.kdeplot(S.Age, color = "b" ,shade = True,label ="Survived" )
sns.kdeplot(SN.Age, color = "r", shade = True,label ="Not Survived" )
plt.title("Survived Ages")
plt.ylabel("Frequency")
plt.xlabel("Age")
plt.grid(True, alpha = 0.4)
plt.show()
dsh.find_numericlike_categorical_features(train, list(train._get_numeric_data().keys()), "Fare")
# converting features to categorical
train['Pclass'] = train['Pclass'].apply(str)
train['SibSp'] = train['SibSp'].apply(str)
train['Parch'] = train['Parch'].apply(str)
plt.figure(figsize = (13,8))
sns.distplot(train["Age"])
plt.title("Age Distribution")
plt.show()

plt.figure(figsize = (13,8))
sns.distplot(train["Fare"])
plt.title("Fare Distribution")
plt.show()
fig = plt.figure(figsize = (15,8))
sns.jointplot(x=train["Age"], y=train["Fare"])
plt.show()
y_train = train["Survived"]
test = pd.read_csv("/kaggle/input/titanic/test.csv")
test.head()
test_ID = test['PassengerId'] # save PassengerId (for submission)
test = test.drop(["Cabin","Name","PassengerId","Ticket"], axis = 1)
test.info()
test = dsh.fill_nan_numeric(test, list(test._get_numeric_data().keys()))
test.info()
ntrain = train.shape[0]
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['Survived'], axis=1, inplace=True)
all_data.head()
all_data['Pclass'] = all_data['Pclass'].apply(str)
all_data['SibSp'] = all_data['SibSp'].apply(str)
all_data['Parch'] = all_data['Parch'].apply(str)
all_data['Age'] = all_data['Age'].apply(int)
all_data["Sex"] = [1 if each == "male" else 0 for each in all_data["Sex"]] # male -> 1, female -> 0
def show_countplot(data, features):
    for feature in features:
        plt.figure(figsize = (10,7))
        sns.countplot(data[feature])
        plt.title(feature)
        plt.show()
        
show_countplot(all_data, list(all_data.select_dtypes(include='object').keys()))
all_data = dsh.categorical_features_as_binary(all_data, list(all_data.select_dtypes(include='object').keys()))
list_1 = ["Fare","Age"]
all_data = dsh.numeric_features_as_5_classes(all_data, list_1)
all_data.head()
all_data = pd.get_dummies(all_data)
all_data.head()
x_train = all_data[:ntrain]
test = all_data[ntrain:]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,test_size = 0.2, random_state = 42)
from sklearn.metrics import confusion_matrix
from matplotlib.gridspec import  GridSpec
from sklearn.linear_model import RidgeClassifier

score_list_ridge = []
train_list = []
for i in np.arange(0.0005, 0.003, 0.0005):
    ridge = RidgeClassifier(tol = i, random_state = 42) 
    ridge.fit(x_train,y_train)
    score_list_ridge.append(ridge.score(x_test,y_test))
    train_list.append(ridge.score(x_train,y_train))

best_parameter_value = dsh.show_sklearn_model_results(score_list_ridge, 
                                                      train_list, 
                                                      np.arange(0.0005, 0.003, 0.0005),
                                                      "tol")

test_survived = pd.Series(rid_best.predict(test), name = "Survived").astype(int)
results = pd.concat([test_ID, test_survived],axis = 1)
results.to_csv("titanic.csv", index = False)