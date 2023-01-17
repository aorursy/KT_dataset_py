import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split



from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

from sklearn.metrics import confusion_matrix
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input//test.csv')

test
data = train.append(test , ignore_index = True)

print (train.shape)
train.isnull().sum()
train.shape
data.Age.hist()

plt.show()
corr = data.corr()

fig ,ax = plt.subplots(figsize=(16,12))

fig = sns.heatmap(corr,annot=True)

plt.show()
data.head()

sex = pd.DataFrame()

sex['Sex'] = np.where(data.Sex =='male',1,0)

sex = pd.get_dummies(data.Sex,prefix ='Sex')

embarked = pd.get_dummies(data.Embarked, prefix='Embarked')

pclass = pd.get_dummies(data.Pclass,prefix='Pclass')
sex
embarked
pclass
imputed = pd.DataFrame()

imputed['Age'] = train.Age.fillna(train.Age.mean()).append(test.Age.fillna(train.Age.mean()))

imputed['Fare']  = train.Fare.fillna(train.Fare.mean()).append( test.Fare.fillna(train.Fare.mean()))

imputed
Title_Dictionary = {

                    "Capt":       "Officer",

                    "Col":        "Officer",

                    "Major":      "Officer",

                    "Jonkheer":   "Royalty",

                    "Don":        "Royalty",

                    "Sir" :       "Royalty",

                    "Dr":         "Officer",

                    "Rev":        "Officer",

                    "the Countess":"Royalty",

                    "Dona":       "Royalty",

                    "Mme":        "Mrs",

                    "Mlle":       "Miss",

                    "Ms":         "Mrs",

                    "Mr" :        "Mr",

                    "Mrs" :       "Mrs",

                    "Miss" :      "Miss",

                    "Master" :    "Master",

                    "Lady" :      "Royalty"



                    }



title = pd.DataFrame()

title['Title'] = data['Name'].map(lambda x : x.split(',')[1].split('.')[0].strip())

title['Name'] = title.Title.map(Title_Dictionary)

title = pd.get_dummies(title['Name'])

title
cabin = pd.DataFrame()

cabin['Cabin'] = data['Cabin'].fillna('U')

cabin['Cabin'] = cabin['Cabin'].map(lambda c : c[0])

cabin = pd.get_dummies(cabin['Cabin'],prefix='Cabin')

cabin
family = pd.DataFrame()



family['FamilySize'] = data['Parch'] + data['SibSp'] + 1 



family['Family_Single'] = family['FamilySize'].map(lambda x : 1 if x == 1 else 0)

family['Family_Small'] = family['FamilySize'].map(lambda x : 1 if  2 <= x <= 4 else 0)

family['Family_Large']  = family['FamilySize'].map(lambda x : 1 if 5 <= x else 0)



del family['FamilySize']

family
ticket = pd.DataFrame()

ticket['Ticket']=data['Ticket']

ticket = ticket.replace( '.' , '' )

ticket = ticket.replace( '/' , '' )

ticket
data_x = pd.concat([sex,embarked,cabin,pclass],axis = 1)

train_x = data_x[:891]

train_y = data.Survived[:891]

test = data_x[891:]

(train_xx,test_xx,train_yy,test_yy) = train_test_split(train_x , train_y , test_size = 0.3)

test
model = RandomForestClassifier(n_estimators=500)
model = SVC()
model = GradientBoostingClassifier()
model = KNeighborsClassifier(n_neighbors = 1)
model = GaussianNB()
model = LogisticRegression()
model.fit(train_x,train_y)

print (model.score(test_xx,test_yy))
test_pred = model.predict(test)

passenger_id = data[891:].PassengerId

predicted = pd.DataFrame({'PassengerId': passenger_id,'Survived':test_pred})

predicted.to_csv('randomforest.csv',index=False)