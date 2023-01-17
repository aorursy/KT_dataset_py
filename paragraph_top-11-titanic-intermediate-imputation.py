import os

import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

from xgboost import XGBRegressor, XGBClassifier

from sklearn.model_selection import cross_validate

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, mean_squared_error

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train = pd.read_csv(r'/kaggle/input/titanic/train.csv')

test = pd.read_csv(r'/kaggle/input/titanic/test.csv')

data = pd.concat([train,test],axis=0, sort=True).reset_index(drop=True)
missing = data.isnull().sum().sort_values(ascending=False) / len(data)



plt.figure(figsize=(15,7))

sns.barplot(x=missing.index, y=missing.values)

plt.title('Amount of missing data relative to whole dataset', fontsize=15)

plt.xlabel('Feature',fontsize=11)

plt.ylabel('Amount of missing data',fontsize=11)
data.loc[data.Fare.isnull(),'Fare'] = data[(data.Embarked=='S') & (data.Pclass==3)].Fare.median()

data.loc[data.Embarked.isnull(),'Embarked'] = 'S'
data.Name.head(10)
data['Title'] = data.Name.map(lambda x: x.split(',')[1].split('.')[0])



#Because Mlle is french equivalent of Miss

data.loc[data.Title==' Mlle', 'Title'] = ' Miss'

#They were travelling alone, so I assume that they are unmarried

data.loc[data.Title==' Ms', 'Title'] = ' Miss'

#Mme is french equivalent for Mrs

data.loc[data.Title==' Mme', 'Title'] = ' Mrs'

#Royals and stuff

data.Title = data.Title.replace({' Sir':'Demigod',' Lady':'Demigod',' the Countess':'Demigod',' Don':'Demigod', ' Dona':'Demigod', ' Jonkheer':'Demigod'})

#Military

data.Title = data.Title.replace({' Col':'Military',' Major':'Military',' Capt':'Military'})



data.loc[(data.Title=='Demigod') & (data.Sex=='female'),'Title'] = ' Miss'

data.loc[(data.Title=='Demigod') & (data.Sex=='male'),'Title'] = ' Mr'
data['CabinLetter'] = data.Cabin.astype(str).map(lambda x: str(x[0]))

data['CabinLetter'] = data.CabinLetter.fillna('n')

data =  data.rename(columns={'Sex':'SexFemale'})

data['SexFemale'] = data.SexFemale.astype(str).replace({'male':0,'female':1})

np.random.seed(1)



Y = data.Age[data.Age.isnull()==False]

testY = Y.loc[np.random.choice(Y.index,100, replace=False)]

median = np.median(Y.loc[np.setdiff1d(Y.index,testY.index)])

MaI = np.sqrt(mean_squared_error(testY, np.full((100,1),median)))

print('Holdout set:',MaI)

np.random.seed(1)



testX = pd.DataFrame(data.Title.loc[testY.index])

testX['Predicted'] = np.NaN



for i in data.Title.loc[np.setdiff1d(Y.index,testY.index)].unique():

    testX.loc[testX.Title==i, 'Predicted'] = data[data.Title==i].Age.loc[np.setdiff1d(Y.index,testY.index)].median()



MasI = np.sqrt(mean_squared_error(testY, testX.Predicted))

    

print('Holdout set:',MasI)

data['Family'] = data.SibSp + data.Parch + 1

data['PeopleOnTheSameTicket'] = None



for i in data.Ticket.unique():

    summary = len(data[data.Ticket==i])

    data.loc[data.Ticket==i, 'PeopleOnTheSameTicket'] = summary



data['FarePerPerson'] = data.Fare/data.PeopleOnTheSameTicket

data['Wealth'] = pd.cut(data.FarePerPerson,7,labels=range(1,8))
from sklearn.linear_model import Lasso, Ridge

X = data[data.Age.isnull()==False]

X = X[['Age', 'Embarked','Fare', 'Parch','Pclass','SexFemale','SibSp','Title', 'PeopleOnTheSameTicket','FarePerPerson','Family','Wealth']]

Y = X.Age

X = X.drop('Age',axis=1)

X = pd.get_dummies(X, drop_first=True)

testX = X.loc[testY.index]

X = X.drop(testY.index,axis=0)

Y = Y.drop(testY.index,axis=0)



model = XGBRegressor(learning_rate=0.001,n_estimators=3000, reg_lambda=0.1, objective='reg:squarederror')

model.fit(X, Y)



MA = np.sqrt(mean_squared_error(testY, model.predict(testX)))

print('Holdout set:',MA)
print('Median of age:',MaI,'\nMedian of age in subclasses:',MasI,'\nModel approach:',MA)
#Model age imputation



X = data[data.Age.isnull()==False]

X = X[['Age', 'Embarked','Fare', 'Parch','Pclass','SexFemale','SibSp','Title', 'PeopleOnTheSameTicket','FarePerPerson','Family','Wealth']]

Y = X.Age

X = X.drop('Age',axis=1)

X = pd.get_dummies(X, drop_first=True).drop(['Title_ Rev','Title_Military'],axis=1)



model = XGBRegressor(learning_rate=0.001,n_estimators=3000, reg_lambda=0.1)

model.fit(X,Y)

dummydata = pd.get_dummies(data[data.Age.isnull()][['Embarked','Fare', 'Parch','Pclass','SexFemale','SibSp','Title',

                                                    'PeopleOnTheSameTicket','FarePerPerson','Family','Wealth']], drop_first=True)



data.loc[data.Age.isnull(), 'Age'] = model.predict(dummydata)
data['AgeInterval'] = pd.cut(data.Age, 3, labels=['Young','Mid-aged', 'Old'])

data['FareTAge'] = data.Fare * data.Age
X = data[data.CabinLetter!='n']

X = X[['Age', 'Embarked','Fare', 'Parch','Pclass','SexFemale','SibSp','Title', 'CabinLetter']]

Y = X.CabinLetter

X = X.drop('CabinLetter',axis=1)

X = pd.get_dummies(X)

Y = Y.replace({'C':1,'E':2,'G':3,'D':4,'A':5,'B':6,'F':7,'T':8})



trainX, testX, trainY, testY = train_test_split(X,Y, test_size=0.3)



model = RandomForestClassifier(n_estimators=500, random_state=1)

model.fit(trainX,trainY)



trainX = pd.concat([trainX,trainY], axis=1)

testX = pd.concat([testX, testY], axis=1)



for i in trainX.CabinLetter.unique():

    print('Train result for {} :'.format(i), accuracy_score(trainX.CabinLetter[trainX.CabinLetter==i], model.predict(trainX.drop('CabinLetter',axis=1)[trainX.CabinLetter==i])))

    

for i in testX.CabinLetter.unique():

    print('Holdout result for {} :'.format(i), accuracy_score(testX.CabinLetter[testX.CabinLetter==i], model.predict(testX.drop('CabinLetter',axis=1)[testX.CabinLetter==i])))

    

    

print('Accuracy overall:', accuracy_score(testX.CabinLetter, model.predict(testX.drop('CabinLetter',axis=1))))
#Model CabinLetter inmputation 



X = data[data.CabinLetter!='n']

X = X.drop(X[X.Age.isnull()].index,axis=0)

X = X[['Age', 'Embarked','Fare', 'Parch','Pclass','SexFemale','SibSp','Title', 'CabinLetter']]

Y = X.CabinLetter

X = X.drop('CabinLetter',axis=1)

X = pd.get_dummies(X, drop_first=True)

X['Title_ Rev'] = 0

Y = Y.replace({'C':1,'E':2,'G':3,'D':4,'A':5,'B':6,'F':7,'T':8})



model = RandomForestClassifier(n_estimators=500, random_state=1)

model.fit(X,Y)

dummydata = pd.get_dummies(data[data.CabinLetter=='n'][['Age', 'Embarked','Fare', 'Parch','Pclass','SexFemale','SibSp','Title', 'CabinLetter']], drop_first=True)

data.loc[data.CabinLetter=='n','CabinLetter'] = model.predict(dummydata)



data.CabinLetter = data.CabinLetter.replace({1:'C',2:'E',3:'G',4:'D',5:'A',6:'B',7:'F',8:'T'})
# Because its child is in cabin F and people from 2nd class dont get G cabin

data.CabinLetter.loc[247] = 'F'

# Husband is in E and people from 3rd class dont get C cabin

data.CabinLetter.loc[85] = 'E'

# Rest is in F and people from 2nd class dont get C cabin

data.CabinLetter.loc[665] = 'F'

# There is no F cabin. Changing to A is justified by similar Fare to median of Fare for 1st class in cabin A

data.CabinLetter.loc[339] = 'A'
data = data.drop(['Cabin','Name','Ticket'],axis=1)

data = pd.get_dummies(data, drop_first=True)

X = data[data.Survived.isnull()==False].drop(['Survived', 'PassengerId'],axis=1)

Y = data.Survived[data.Survived.isnull()==False]

trainX, testX, trainY, testY = train_test_split(X,Y,test_size=0.30, random_state=2019)
def ModelCreation(model):

    cv = cross_validate(model,

                        X,

                        Y,

                        cv=10,

                        scoring='accuracy',

                        return_train_score=True)



    print('Training score average:',cv['train_score'].sum()/10)

    print('Holdout score average:',cv['test_score'].sum()/10)
model = RandomForestClassifier(n_estimators=900, max_depth=4)

ModelCreation(model)
model = XGBClassifier(learning_rate=0.001,n_estimators=4000,

                                max_depth=4, min_child_weight=0,

                                gamma=0, subsample=0.7,

                                colsample_bytree=0.7,

                                scale_pos_weight=1, seed=27)



ModelCreation(model)
model = RandomForestClassifier(n_estimators=900, max_depth=4)

model.fit(X, Y)

test = data[data.Survived.isnull()]

test['Survived'] = model.predict(test.drop(['Survived','PassengerId'],axis=1)).astype(int)

test = test.reset_index()

test[['PassengerId','Survived']].to_csv("RFClass.csv",index=False)