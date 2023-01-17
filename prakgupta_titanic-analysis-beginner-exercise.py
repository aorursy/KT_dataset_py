# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression

 

training_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')



training_data.info()

#Getting mean age to fill in NA values

training_data["Age"].mean()

#training_data["Age"].fillna(27,inplace=True)

#training_data.tail(50)

 

training_data["Age"].fillna(30,inplace=True)

gender = pd.get_dummies(training_data['Sex'],drop_first=True)

gender.head()

training_data['male'] = gender

#training_data.tail(50)



#Creating dummy variables

training_embarked_dummy = pd.get_dummies(training_data['Embarked'])

training_data['embarked_C'] = training_embarked_dummy['C']

training_data['embarked_Q'] = training_embarked_dummy['Q']

training_data['embarked_S'] = training_embarked_dummy['S']



#training_pclass_dummy = pd.get_dummies(training_data['Pclass'])

#training_data['pclass_1'] = training_pclass_dummy[1]

#training_data['pclass_2'] = training_pclass_dummy[2]

#training_data['pclass_3'] = training_pclass_dummy[3]



#training_data.drop(['PassengerId','Name','Sex','Embarked','Cabin','Fare'])

training_data.drop(['PassengerId','Name','Sex','Embarked','Cabin','Fare','Ticket'],axis=1,inplace=True)

training_data.tail(50)

 

y = training_data['Survived']

training_data.drop('Survived', axis=1,inplace=True)



#training_data.info()





logreg = LogisticRegression()

logreg.fit(training_data, y)

logreg.score(training_data, y)





#Predicting the values using trained model

test_data["Age"].fillna(30,inplace=True)

#training_data.tail(50)

 

gender = pd.get_dummies(test_data['Sex'],drop_first=True)

#gender.head()

test_data['male'] = gender

#training_data.tail(50)





test_embarked_dummy = pd.get_dummies(test_data['Embarked'])

test_data['embarked_C'] = test_embarked_dummy['C']

test_data['embarked_Q'] = test_embarked_dummy['Q']

test_data['embarked_S'] = test_embarked_dummy['S']



#test_pclass_dummy = pd.get_dummies(training_data['Pclass'])

#test_data['pclass_1'] = test_pclass_dummy[1]

#test_data['pclass_2'] = test_pclass_dummy[2]

#test_data['pclass_3'] = test_pclass_dummy[3]





#pd.concat([training_data,xx])

#gender2.head()

 

#training_data.drop(['PassengerId','Name','Sex','Embarked','Cabin','Fare'])

test_passenger_id = test_data['PassengerId']

test_data.drop(['PassengerId','Name','Sex','Embarked','Cabin','Fare','Ticket'],axis=1,inplace=True)

test_data.tail(50)

 

#y = training_data['Survived']

#training_data.drop('Survived', axis=1,inplace=True)

res = logreg.predict(test_data)

res









#test_data.head()

#y_pred = logreg.predict(test_data)

#from sklearn.metrics import confusion_matrix

#confusion_matrix = confusion_matrix(y_test, y_pred)

#print(confusion_matrix)

 

#training_data.append(xx)

#training_data.head()

#training_data.info()

#train_data["Age"].fillna(28, inplace=True)

#training_data.info()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

 

#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))

#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

#Creating output CSV file

test_data.info()

#test_pass_id[0]



final_res = pd.DataFrame({

        "PassengerId": test_passenger_id,

        "Survived": res

    })





final_res.to_csv('titanic_res_4.csv', index=False)



final_res