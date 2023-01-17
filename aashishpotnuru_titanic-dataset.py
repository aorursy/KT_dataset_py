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
data = pd.read_csv("/kaggle/input/titanic/train.csv")
data = data.set_index("PassengerId")
Y = data.Survived
data = data.drop(columns= ['Survived','Name', 'Ticket', 'Cabin'])
data.head()
data.Pclass = data.Pclass.astype(str)
features = [ 'Pclass','Sex', 'Embarked']

dums = pd.get_dummies(data[features])

data = data.drop(columns=features)
data = pd.concat([data, dums], axis = 1)
data.head()
data.fillna(data.mean(), inplace=True)
data.isnull().values.any() #verify there's no null values
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

model_features=['Age','SibSp','Parch','Fare','Pclass_1','Pclass_2','Pclass_3','Sex_female','Sex_male','Embarked_C','Embarked_Q','Embarked_S']

X = data[model_features]

X_train, X_dev, y_train, y_dev = train_test_split(X,Y, test_size = 0.15)
 
train_accs=[]
dev_accs = []
for nb_est in range(1,300):
    model = RandomForestClassifier(n_estimators=nb_est, max_depth=5, random_state=1)
    model.fit(X_train,y_train)
    
    pred_train = model.predict(X_train)
    acc_train = accuracy_score(y_train, pred_train)
    
    pred_dev = model.predict(X_dev)
    acc_train = accuracy_score(y_train, pred_train)
    
    pred_dev = model.predict(X_dev)
    acc_dev = accuracy_score(y_dev, pred_dev)
    
    train_accs.append(acc_train)
    dev_accs.append(acc_dev)
plt.plot(train_accs)
plt.plot(dev_accs)

plt.legend(['Training accuracy', 'Dev-set accuracy'], loc='upper left')
plt.show()
model_final = RandomForestClassifier(n_estimators=145, max_depth=5, random_state=1)
model_final.fit(X,Y)
test = pd.read_csv("/kaggle/input/titanic/test.csv")
test = test.set_index("PassengerId")
test = test.drop(columns= ['Name', 'Ticket', 'Cabin'])

test.Pclass = test.Pclass.astype(str)
dums2 = pd.get_dummies(test[features])
test = test.drop(columns=features)

test = pd.concat([test, dums2], axis = 1)

test.fillna(test.mean(), inplace=True)

test.head()
test = test[model_features]
pred = model_final.predict(test)

output = pd.DataFrame({'PassengerId':test.index, 'Survived': pred})
output.to_csv('my_submission.csv', index=False)