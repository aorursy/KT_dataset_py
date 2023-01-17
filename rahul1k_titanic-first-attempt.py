# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression 
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()

train.info()
train.isnull().sum()
# Eliminating attributes not useful
train_1 = train.drop(['Cabin','Ticket','PassengerId','Name'], axis = 1)
test_1 = test.drop(['Cabin','Ticket','PassengerId','Name'], axis = 1)
# Number of Passengers per ticket
train_1['PPT'] = train_1['SibSp'] + train_1['Parch'] + 1
train_1.drop(['SibSp','Parch'], axis = 1,inplace = True)

test_1['PPT'] = test_1['SibSp'] + test_1['Parch'] + 1
test_1.drop(['SibSp','Parch'], axis = 1,inplace = True)

train_1.head()
#train[train['Embarked'].isnull()]
train_1[train_1['Embarked'].isnull()]
# Fill Embarked missing values
# Method 1: Calculate Mode and replace NaN values with most frequent Embark point.
#Value = train_1['Embarked'].mode()
#train_1['Embarked'].fillna('Value',inplace = True)
#train_1[train_1.Embarked.isnull()]
#Embarked_missFill = pd.concat([Embarked_missFill,Embarked], axis = 1)

# Method 2
E_Value = train_1[(train_1['Sex']== 'female') & (train_1['PPT'] == 1) & (train_1['Pclass'] == 1)]
E_Value[['Fare','Embarked']].groupby(['Embarked']).mean()
#sns.countplot(x = 'Pclass',hue = 'Embarked',data = train_1)
#E_Value = train_1[(train_1['Embarked']=='Q') & (train_1['Pclass'] == 3)]
#(E_Value['Age']).hist()

train_1['Embarked'].fillna('S',inplace = True)

train_1[train_1.Age.isnull()]
train_1['Age'].fillna(train_1.Age.mean(), inplace = True)
Embarked_dummies = pd.get_dummies(train_1['Embarked'],drop_first = True)
Sex = pd.get_dummies(train_1['Sex'],drop_first = True)
train_new = pd.concat([train_1,Embarked_dummies,Sex], axis = 1)
train_new.rename(columns={'male':'sex'}, inplace = True)
train_new.drop(['Embarked','Sex'], axis = 1, inplace = True)
train_new.head()
# Test data cleaning
test_1.info()

test_1[test_1.Fare.isnull()]
F_Value = test_1[(test_1['Sex']== 'male') & (test_1['PPT'] == 1) & (test_1['Pclass'] == 3) & (test_1['Embarked'] == 'S')]
v = F_Value['Fare'].mean()
test_1['Fare'].fillna(v,inplace = True)
test_1['Age'].fillna(test_1.Age.mean(),inplace = True)


test_1.isnull().sum()
Embarked_dummies_test = pd.get_dummies(test_1['Embarked'],drop_first = True)
sex = pd.get_dummies(test_1['Sex'],drop_first = True)
test_new = pd.concat([test_1,Embarked_dummies_test,sex], axis = 1)
test_new.rename(columns={'male':'sex'}, inplace = True)
test_new.drop(['Embarked','Sex'], axis = 1, inplace = True)
test_new.head()
X = train_new.drop(['Survived'], axis = 1)
Y = train_new.Survived.copy()


from sklearn.metrics import accuracy_score
kfolds = StratifiedKFold(n_splits=4, random_state = 2)
LR = LogisticRegression()
#svm = SVC()
#sgd = SGDClassifier()
#knn = KNeighborsClassifier()
acc_sum = 0
c= 0
for train_index,test_index in kfolds.split(X,Y):
    x_train = X.loc[train_index]
    y_train = Y.loc[train_index]
    x_test = X.loc[test_index]
    y_test = Y.loc[test_index]
    
    
    LR.fit(x_train,y_train)
    y_predict = LR.predict(x_test)
    acc_sum += accuracy_score(y_test,y_predict)
    c += 1

print(acc_sum/4)

#from sklearn.model_selection import cross_val_score
#cross_val_score(LR,X,Y,cv= 12,scoring = 'accuracy')

output = LR.predict(test_new)
output_1 = pd.DataFrame(output,columns = ['Survived'])

PD = pd.DataFrame(test['PassengerId'])
output_2 = [PD ,output_1]





Final_output = pd.concat(output_2, axis = 1)
Final_output

Final_output.to_csv('csv_to_submit.csv', index = False)
