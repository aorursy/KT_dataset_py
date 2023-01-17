import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from matplotlib import pyplot as plt
train_data=pd.read_csv('../input/train.csv')

print(train_data.head(5))
test_data=pd.read_csv('../input/test.csv')

print(test_data.head(5))
train_data['Gender'] = train_data['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

print(train_data['Gender'].head(5))
if len(train_data.Embarked[train_data.Embarked.isnull()]) > 0:

    train_data.Embarked[train_data.Embarked.isnull()] = train_data.Embarked.dropna().mode().values
Ports = list(enumerate(np.unique(train_data['Embarked']))) 

Ports_dict = { name : i for i, name in Ports }

train_data.Embarked = train_data.Embarked.map( lambda x: Ports_dict[x]).astype(int)
median_age = train_data['Age'].dropna().median()

if len(train_data.Age[ train_data.Age.isnull() ]) > 0:

    train_data.loc[ (train_data.Age.isnull()), 'Age'] = median_age
train_data = train_data.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 
test_data['Gender'] = test_data['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
if len(test_data.Embarked[ test_data.Embarked.isnull() ]) > 0:

    test_data.Embarked[ test_data.Embarked.isnull() ] = test_data.Embarked.dropna().mode().values



test_data.Embarked = test_data.Embarked.map( lambda x: Ports_dict[x]).astype(int)







median_age = test_data['Age'].dropna().median()

if len(test_data.Age[ test_data.Age.isnull() ]) > 0:

    test_data.loc[ (test_data.Age.isnull()), 'Age'] = median_age





if len(test_data.Fare[ test_data.Fare.isnull() ]) > 0:

    median_fare = np.zeros(3)

    for f in range(0,3):                                              # loop 0 to 2

        median_fare[f] = test_data[ test_data.Pclass == f+1 ]['Fare'].dropna().median()

    for f in range(0,3):                                              # loop 0 to 2

        test_data.loc[ (test_data.Fare.isnull()) & (test_data.Pclass == f+1 ), 'Fare'] = median_fare[f]





ids = test_data['PassengerId'].values



test_data = test_data.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 
x_train=train_data[['Pclass','Gender','Age','SibSp','Parch','Fare','Embarked']]

y_train=train_data['Survived']

print(x_train.head(5))
x_test=test_data[['Pclass','Gender','Age','SibSp','Parch','Fare','Embarked']]

x_test.columns
x_train_data_p,x_train_cv_p,y_train_data_p,y_train_cv_p=train_test_split(x_train,y_train,test_size=0.3, random_state=0)
x_train_data=x_train_data_p.values

x_train_cv=x_train_cv_p.values

y_train_data=y_train_data_p.values

y_train_cv=y_train_cv_p.values
forest = RandomForestClassifier(n_estimators=100)

forest = forest.fit(x_train_data,y_train_data)

output = forest.predict(x_train_cv).astype(int)
output.shape,y_train_cv.shape
accuracy=(sum(np.equal(output,y_train_cv))/len(output))*100

print(accuracy)
res=np.arange(58,dtype='float32').reshape(29,2)

count=0

for i in range(5,150,5):

    forest = RandomForestClassifier(n_estimators=i)

    forest = forest.fit(x_train_data,y_train_data)

    output = forest.predict(x_train_cv).astype(int)

    accuracy=(sum(np.equal(output,y_train_cv))/len(output))*100

    res[count][0]=i

    res[count][1]=accuracy

    count=count+1

print(res)
plt.plot(res[:,0], res[:,1], 'b--')

plt.plot(res[:,0], res[:,1], 'ro')
final_forest = RandomForestClassifier(n_estimators=110)

final_forest = final_forest.fit(x_train,y_train)

final_output = final_forest.predict(x_test).astype(int)