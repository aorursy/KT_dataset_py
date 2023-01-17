# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeClassifier
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Converting the dataset to dataframe

train = pd.read_csv("/kaggle/input/weave-titanic/train.csv") 
test = pd.read_csv("/kaggle/input/weave-titanic/test.csv") 
#It shows the first 5 rows of the dataframe
train.head()
#Look that the 'test' dataframe doesn't have a 'Survived' column, it's what we want to predict
test.head()
#It removes the specific columns, like 'Name', 'Ticket', 'Cabin', in this case
#If you don't want to create a new dataframe you need to set the 'inplace' parameter as 'True'
'''
Remove as colunas especificadas, no caso quando desejar remover mais de uma coluna de uma so vez,
deve-se utilizar uma lista como primeiro argumento.
Poderia alterar/remover diretamente no dataset, sem precisar atribuir novamente a um dataframe.
Deveria utilizar entao: train.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)
'''

train = train.drop(["Name", "Ticket", "Cabin"], axis=1)
test = test.drop(["Name", "Ticket", "Cabin"], axis=1)
#Now it shows the new dataframe without the dropped columns
#We need to apply the changes in both 'train' and 'test' datasets
train.head()
test.head()
#Applies the one_hot_encoding for the 'Sex' and 'Embarked' features
new_data_train = pd.get_dummies(train)
new_data_test = pd.get_dummies(test)
#Now we can see that the 'Sex' and 'Embarked' are now numerical columns.
new_data_train.head()
#Checks if there is a NaN value for the training data, 'train' data.
new_data_train.isnull().sum().sort_values(ascending=False).head(10)
#We will use the mean 'Age' of the dataset for the NaN values
new_data_train["Age"].fillna(new_data_train["Age"].mean(), inplace=True)
new_data_test["Age"].fillna(new_data_test["Age"].mean(), inplace=True)
#Checks if there is a NaN value for the testing data, 'test' data
new_data_test.isnull().sum().sort_values(ascending=False).head(10)
#We will use the mean 'Fare' for the NaN values
new_data_test["Fare"].fillna(new_data_test["Fare"].mean(), inplace=True)
#Splitting the 'features' and 'targets' for the model, as X and y
#Separando "features" e "targets" para o modelo, X e y respectivamente
X = new_data_train.drop("Survived", axis=1)
y = new_data_train["Survived"]
#We will use a Decision Tree Model as the Machine Learning Algorithm
#Utilizaremos Decision Tree, como algoritmo de Machine Learning
tree = DecisionTreeClassifier(max_depth = 10, random_state = 0)
tree.fit(X, y)
tree.score(X, y)
#Import the necessary modules for the model
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
#It's already done, without the 'Survived' columns and with all the features prepared
X.head()
#Test Data
Xtest = new_data_test
Xtest.head()
Xtrain, Xvalidation, Ytrain, Yvalidation = train_test_split(X, y, test_size=0.2, random_state=True)
#Model
model = RandomForestClassifier(n_estimators=100,
                               max_leaf_nodes=12,
                               max_depth=12,
                               random_state=0)
model.fit(Xtrain, Ytrain)
# model.score(Xtrain, Ytrain)
#Prediction
from sklearn.metrics import accuracy_score
Yprediction = model.predict(Xvalidation)
accuracy_score(Yvalidation, Yprediction)
#Submission
#We create a new dataframe for the submission
submission = pd.DataFrame()

submission["PassengerId"] = Xtest["PassengerId"]
submission["Survived"] = model.predict(Xtest)

#We save the submission as a '.csv' file
submission.to_csv("submission.csv", index=False)
submission.head()