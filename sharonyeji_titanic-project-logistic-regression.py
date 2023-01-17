import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import sklearn.model_selection as ms

import sklearn.linear_model as lm

from sklearn.linear_model import LogisticRegression

train = pd.read_csv('../input/titanic-machine-learning-from-disaster/train.csv') #PassengerId 1-891

test = pd.read_csv('../input/titanic-machine-learning-from-disaster/test.csv') #PassengerId 892-1309
train = train.drop('Survived',1)

train.head()
test.head()
titanic_df = train.append(test)

del_cols = ['Name', 'Ticket', 'Cabin']

titanic_df = titanic_df.drop(del_cols,1)

titanic_df.head()
titanic_df.Sex[titanic_df.Sex == 'male'] = 1

titanic_df.Sex[titanic_df.Sex == 'female'] = 0

titanic_df.Embarked[titanic_df.Embarked == 'Q'] = 2

titanic_df.Embarked[titanic_df.Embarked == 'S'] = 1

titanic_df.Embarked[titanic_df.Embarked == 'C'] = 0

titanic_df
titanic_df.loc[titanic_df['Age']<10,'GroupedAge'] = 0

titanic_df.loc[(titanic_df['Age']>=10) & (titanic_df['Age']<20),'GroupedAge'] = 1

titanic_df.loc[(titanic_df['Age']>=20) & (titanic_df['Age']<30),'GroupedAge'] = 2

titanic_df.loc[(titanic_df['Age']>=30) & (titanic_df['Age']<40),'GroupedAge'] = 3

titanic_df.loc[(titanic_df['Age']>=40) & (titanic_df['Age']<50),'GroupedAge'] = 4

titanic_df.loc[(titanic_df['Age']>=50) & (titanic_df['Age']<60),'GroupedAge'] = 5

titanic_df.loc[(titanic_df['Age']>=60) & (titanic_df['Age']<70),'GroupedAge'] = 6

titanic_df.loc[(titanic_df['Age']>=70) & (titanic_df['Age']<80),'GroupedAge'] = 7

titanic_df.loc[titanic_df['Age']>=80,'GroupedAge'] = 8

titanic_df['GroupedAge'] = titanic_df['GroupedAge'].fillna(11)
titanic_df.loc[titanic_df['Fare']<50,'GroupedFare'] = 0

titanic_df.loc[(titanic_df['Fare']>=50) & (titanic_df['Fare']<100),'GroupedFare'] = 1

titanic_df.loc[(titanic_df['Fare']>=100) & (titanic_df['Fare']<150),'GroupedFare'] = 2

titanic_df.loc[(titanic_df['Fare']>=150) & (titanic_df['Fare']<200),'GroupedFare'] = 3

titanic_df.loc[(titanic_df['Fare']>=200) & (titanic_df['Fare']<250),'GroupedFare'] = 4

titanic_df.loc[(titanic_df['Fare']>=250) & (titanic_df['Fare']<300),'GroupedFare'] = 5

titanic_df.loc[(titanic_df['Fare']>=300) & (titanic_df['Fare']<350),'GroupedFare'] = 6

titanic_df.loc[(titanic_df['Fare']>=350) & (titanic_df['Fare']<400),'GroupedFare'] = 7

titanic_df.loc[(titanic_df['Fare']>=400) & (titanic_df['Fare']<450),'GroupedFare'] = 8

titanic_df.loc[(titanic_df['Fare']>=450) & (titanic_df['Fare']<500),'GroupedFare'] = 9

titanic_df.loc[titanic_df['Fare']>=500,'GroupedFare'] = 10

titanic_df['GroupedFare'] = titanic_df['GroupedFare'].fillna(11)

titanic_df['Embarked'] = titanic_df['Embarked'].fillna(11)
titanic_df.info()
del_cols = ['Age', 'Fare']

titanic_df = titanic_df.drop(del_cols,1)
train1 = titanic_df.iloc[0:891]
train = pd.read_csv('../input/titanic-machine-learning-from-disaster/train.csv')

train1 = pd.merge(train1, train[['PassengerId','Survived']], on='PassengerId')
train1.head()
plt.hist2d(train1['Pclass'], train1['Survived'], cmap=plt.cm.jet)

plt.colorbar()

#When Pclass = 3, passenger less likely to survive
train1.groupby(['Pclass','Survived']).size()

372/(372+119)

#75.7% of passengers in Pclass 3 do not survive
plt.hist2d(train1['Sex'], train1['Survived'], cmap=plt.cm.jet)

plt.colorbar()

#When Sex is 1 (male), passenger less likely to survive
train1.groupby(['Sex','Survived']).size()

468/(468+109)

#81.1% of passengers that are Male do not survive
plt.hist2d(train1['GroupedAge'], train1['Survived'], cmap=plt.cm.jet)

plt.colorbar()
train1.groupby(['GroupedAge','Survived']).size()

61/(61+41)

#59.8% of passengers that are age 10-20 do not survive, but it does not seem as significant as sex and pclass
plt.hist2d(train1['GroupedFare'], train1['Survived'], cmap=plt.cm.jet)

plt.colorbar()
train1.groupby(['GroupedFare','Survived']).size()

497/(497+233)

#68% of participants that have fare $0-50 do not survive, but it does not seem as significant as sex and pclass
#Logistic Regression

clf_k_scores = []

for train, test in kf.split(X_train):

    logreg = LogisticRegression()

    logreg.fit(X_train[train], y_train[train])

    clf_k_scores.append(logreg.score(X_train[test], y_train[test]))



scores = np.array(clf_k_scores)

k_scores = scores.mean()

print(k_scores)
submission = logreg.predict(X_test)
Passenger_Id = titanic_df.iloc[891:,0].values
submission_final = pd.DataFrame({'Passenger': Passenger_Id, 'Survived': submission}, columns=['Passenger', 'Survived'])

submission_final
submission_final.to_csv(r'submission_final.csv',index=False)