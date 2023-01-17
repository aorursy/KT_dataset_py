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
train_data = pd.read_csv("../input/titanic/train.csv")

test_data = pd.read_csv("../input/titanic/test.csv")
train_data.columns
test_data.columns
train_data.head()
test_data.head()
train_data.isnull().sum()
test_data.isnull().sum()
train_data.drop(columns = ["Name" ,"Ticket" , "Cabin"], inplace=True)
test_data.drop(columns = ["Name" ,"Ticket" , "Cabin"], inplace=True)

age_mean = train_data.Age.mean()

train_data.Age.fillna(age_mean , inplace= True)

test_data.Age.fillna(age_mean , inplace = True)
train_data.isnull().sum()
train_data.Embarked.dtype
train_data.Embarked.unique()
most_value_Embarked = train_data.Embarked.mode()

most_value_Embarked[0]
train_data.Embarked.fillna(most_value_Embarked[0] , inplace = True)

test_data.Embarked.fillna(most_value_Embarked[0] , inplace = True)
train_data.isnull().sum()
test_data.isnull().sum()
Fare_mean = train_data.Fare.mean()

test_data.Fare.fillna(Fare_mean , inplace = True)

test_data.isnull().sum()
train_data.Sex.replace("male" , 0 , inplace =True)

train_data.Sex.replace("female" , 1 , inplace =True)

train_data.Embarked.replace("C" , 0 , inplace =True)

train_data.Embarked.replace("S" , 1 , inplace =True)

train_data.Embarked.replace("Q" , 2 , inplace =True)
train_data.head()
test_data.Sex.replace("male" , 0 , inplace =True)

test_data.Sex.replace("female" , 1 , inplace =True)

test_data.Embarked.replace("C" , 0 , inplace =True)

test_data.Embarked.replace("S" , 1 , inplace =True)

test_data.Embarked.replace("Q" , 2 , inplace =True)
test_data.head()
X = train_data.drop(columns = ["Survived"])

y = train_data["Survived"]
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y)




from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score,f1_score

def get_acc(max_leaf_nodes, train_X , val_X , train_y , val_y):

    model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes , random_state = 1)

    model.fit(train_X,train_y)

    preds_val = model.predict(val_X)

    accurecy = accuracy_score(y_pred=preds_val , y_true=val_y)

    f1 = f1_score(y_true=val_y, y_pred=preds_val, average='micro')

    return f1

para = list(range(2, 15 , 1))

results= {}

for i in para:

    acc = get_acc(i , train_X , val_X , train_y , val_y)

    results[acc] = i

max_acc = max(results.keys())

best_max_nodes = results[max_acc] 

best_max_nodes
model = DecisionTreeClassifier(max_leaf_nodes=8 , random_state = 1)

model.fit(train_X,train_y)

preds_val = model.predict(test_data)
preds_val.shape
test_data.shape
test_data.PassengerId
test_out = pd.DataFrame({

    'PassengerId': test_data.PassengerId, 

    'Survived': preds_val

})

test_out.to_csv('submission.csv', index=False)