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
#%%



import numpy as np 

import pandas as pd 

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score





#%%





train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

#Drop features we are not going to use

train = train.drop(['Name','SibSp','Parch', 'Ticket', 'Fare', 'Cabin'],axis=1)

test = test.drop(['Name','SibSp','Parch', 'Ticket', 'Fare', 'Cabin'],axis=1)

test_val = test



for df in [train,test]:

    df['Sex_binary']=df['Sex'].map({'male':1,'female':0})



embark = {'S':0,"C":1,"Q":2}

data = [train,test]

for df in data:

    df['Embarked']=df['Embarked'].map(embark)

train.columns

    

train['Age'] = train['Age'].fillna(0)

test['Age'] = test['Age'].fillna(0)

features = ['Pclass','Age','Sex_binary']

target = 'Survived'



#%%







#%%



mlr = LinearRegression()

mlr.fit(train[features], train[target])

submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':mlr.predict(test[features])})

x_train, x_test, y_train, y_test = train_test_split(train[features], train[target], train_size = 0.8, test_size = 0.2, random_state=6)

print('mlr',mlr.score(x_test, y_test)*100, '%')



#%%



from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(train[features], train[target])



Y_prediction = random_forest.predict(test[features])



random_forest.score(x_test, y_test)

acc_random_forest = round(random_forest.score(x_test, y_test) * 100, 2)

print('random forest',round(acc_random_forest,2,), "%")







from sklearn.svm import SVC

clf = SVC(gamma='auto')

clf.fit(train[features], train[target])

Y_Pred = clf.predict(test[features])

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': Y_Pred})

output.to_csv('submission.csv', index=False)

SVCC = round(clf.score(x_test, y_test) * 100,2)

print('svm', round(SVCC,2,), "%")





from sklearn.tree import DecisionTreeClassifier   

decision_tree = DecisionTreeClassifier()

decision_tree.fit(train[features], train[target])



Y_pred = decision_tree.predict(test[features])



acc_decision_tree = round(decision_tree.score(x_test, y_test) * 100, 2)

print('decision tree',round(acc_decision_tree,2,), "%")






