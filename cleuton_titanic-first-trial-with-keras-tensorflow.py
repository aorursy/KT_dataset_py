%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

rawdata = pd.read_csv("../input/train.csv")

rawdata.head()
rawdata.count()

totalSurvivers = rawdata['Survived'][rawdata.Survived == 1].count()

totalDeaths = 891 - totalSurvivers

percentOfSurvivers = totalSurvivers / 891 * 100

print("Total Survivers:", totalSurvivers)

print("Total Deaths:", totalDeaths)

print("% of Survivers:", percentOfSurvivers)
percentOfWomen = (rawdata['Sex'][rawdata.Sex == 'female'].count() / 891) * 100

print("Percent of Women passengers:", percentOfWomen)

percentOfMen = (rawdata['Sex'][rawdata.Sex == 'male'].count() / 891) * 100

print("Percent of Men passengers:", percentOfMen)
genderSurvivers = rawdata.groupby(['Sex', 'Survived'])['Survived'].count().reset_index(name="count")

genderSurvivers
genderSurvivers.pivot(index='Sex', columns='Survived', values='count').plot(kind='bar')
ageSurvivers = rawdata[rawdata.Survived == 1][['Age']].dropna().hist()

plcassSurvivers = rawdata.groupby(['Pclass', 'Survived'])['Survived'].count().reset_index(name="count")

plcassSurvivers
plcassSurvivers.pivot(index='Pclass', columns='Survived', values='count').plot(kind='bar')
childrenSpec = rawdata[rawdata.Age < 18][rawdata.Parch < 2][['Age','Parch']]

print("Children in special status:", childrenSpec.count())

childrenSpec.head()

chSpecSurvivers = (rawdata[['Parch','Survived']][(rawdata.Age < 18) & (rawdata.Survived == 1)]).groupby('Parch').count()

chSpecSurvivers
adultsVsSibSp = (rawdata[['Parch','Survived']][(rawdata.Age > 18) & (rawdata.Survived == 1)]).groupby('Parch').count()

adultsVsSibSp
adultsVsParch = (rawdata[['Pclass','Parch','Survived','PassengerId']][(rawdata.Age > 18)]).groupby(['Pclass','Parch','Survived']).count()

adultsVsParch
adultsVsSibSp = (rawdata[['SibSp','Survived', 'PassengerId']][(rawdata.Age > 18)]).groupby(['SibSp','Survived']).count()

adultsVsSibSp

rawdata['Party']=rawdata['SibSp'] + rawdata['Parch']

rawdata[['Party', 'Survived']].groupby(['Party', 'Survived'])['Survived'].count().reset_index(name="count")


fareClasses = pd.qcut(rawdata['Fare'],5)

fareClasses.value_counts()
rawdata['FareCategory']=pd.qcut(rawdata['Fare'],6)

rawdata[(rawdata.Survived == 1)].groupby(['FareCategory'])['Survived'].count().to_frame().style.background_gradient(cmap='summer_r')
cabinDf = (rawdata[['Cabin','Survived']][(rawdata.Survived == 1)]).groupby('Cabin').count()

cabinDf
embarkedDf = (rawdata[['Embarked','Survived']][(rawdata.Survived == 1)]).groupby('Embarked').count()

embarkedDf
rawdata_corr = rawdata.corr()

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(rawdata_corr, vmin=-1, vmax=1)

fig.colorbar(cax)

names = rawdata_corr.columns.values.tolist()

print(names)

ticks = np.arange(0,8,1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(names)

ax.set_yticklabels(names)

plt.show()
rawdata['Sex'][(rawdata.Sex != 'male') & (rawdata.Sex != 'female')].count()
rawdata['Sex'].replace(['male','female'],[0,1],inplace=True)
rawdata.info()
np.count_nonzero(np.isnan(rawdata['Age']))
countSurvivorsWithAgeNaN = rawdata[(rawdata.Survived == 1) & (np.isnan(rawdata.Age))]['PassengerId'].count()

countDeathsWithAgeNaN  = rawdata[(rawdata.Survived == 0) & (np.isnan(rawdata.Age))]['PassengerId'].count()

print("Survivors with Age NaN:", countSurvivorsWithAgeNaN)

print("Deaths    with Age Ok :", countDeathsWithAgeNaN)

survAgeAvg = rawdata[(rawdata.Survived == 1)]['Age'].mean()

print(survAgeAvg)

deadAgeAvg = rawdata[(rawdata.Survived == 0)]['Age'].mean()

print(deadAgeAvg)
rawdata[(np.isnan(rawdata.Age))]

rawdata.ix[(np.isnan(rawdata.Age)) & (rawdata.Survived == 1), 'Age']=survAgeAvg

rawdata.ix[(np.isnan(rawdata.Age)) & (rawdata.Survived == 0), 'Age']=deadAgeAvg
rawdata['AgeCategory']=0

rawdata.ix[rawdata.Age < 12, 'AgeCategory'] = 0

rawdata.ix[(rawdata.Age >= 12) & (rawdata.Age < 18), 'AgeCategory'] = 1

rawdata.ix[(rawdata.Age >= 18) & (rawdata.Age < 25), 'AgeCategory'] = 2

rawdata.ix[(rawdata.Age >= 25) & (rawdata.Age < 51), 'AgeCategory'] = 3

rawdata.ix[rawdata.Age > 51, 'AgeCategory'] = 4



rawdata.drop('Age', axis=1)
rawdata.drop('FareCategory', axis=1)

rawdata['FareCategory']=0

rawdata.ix[rawdata.Fare <= 7.775, 'FareCategory'] = 0

rawdata.ix[(rawdata.Fare > 7.775) & (rawdata.Fare <= 8.662), 'FareCategory'] = 1

rawdata.ix[(rawdata.Fare > 8.662) & (rawdata.Fare <= 14.454), 'FareCategory'] = 2

rawdata.ix[(rawdata.Fare > 14.454) & (rawdata.Fare <= 26), 'FareCategory'] = 3

rawdata.ix[(rawdata.Fare > 26) & (rawdata.Fare <= 52.369), 'FareCategory'] = 4

rawdata.ix[(rawdata.Fare > 52.369), 'FareCategory'] = 5

rawdata.drop('Fare', axis=1)
rawdata.info()
rawdata.ix[(rawdata.Embarked.isnull()),'Embarked'] = 'S'

rawdata.info()
rawdata['Embarked'].value_counts()
rawdata['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

rawdata.info()

rawdata.drop(['PassengerId', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin'], axis=1, inplace=True)
rawdata.info()


import keras

from keras.models import Sequential

from keras.layers import Activation, Dense, Dropout, Flatten

from keras.optimizers import Adam

features = rawdata.drop('Survived', axis=1).values

labels = rawdata['Survived'].values

model = Sequential()

model.add(Dense(128, input_dim=6, kernel_initializer='normal', activation='relu'))

model.add(Dense(512, activation='relu'))

model.add(Dense(512, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(features, labels)
testdata = pd.read_csv("../input/test.csv")

testdata.head()
testdata['Party']=testdata['SibSp'] + testdata['Parch']
testdata['Sex'].replace(['male','female'],[0,1],inplace=True)
testdata[(np.isnan(testdata.Age))]
ageMean = testdata['Age'].mean()

testdata.ix[(np.isnan(testdata.Age)) , 'Age']=ageMean
testdata['AgeCategory']=0

testdata.ix[testdata.Age < 12, 'AgeCategory'] = 0

testdata.ix[(testdata.Age >= 12) & (testdata.Age < 18), 'AgeCategory'] = 1

testdata.ix[(testdata.Age >= 18) & (testdata.Age < 25), 'AgeCategory'] = 2

testdata.ix[(testdata.Age >= 25) & (testdata.Age < 51), 'AgeCategory'] = 3

testdata.ix[testdata.Age > 51, 'AgeCategory'] = 4

#testdata.drop('FareCategory', axis=1)

testdata['FareCategory']=0

testdata.ix[testdata.Fare <= 7.775, 'FareCategory'] = 0

testdata.ix[(testdata.Fare > 7.775) & (testdata.Fare <= 8.662), 'FareCategory'] = 1

testdata.ix[(testdata.Fare > 8.662) & (testdata.Fare <= 14.454), 'FareCategory'] = 2

testdata.ix[(testdata.Fare > 14.454) & (testdata.Fare <= 26), 'FareCategory'] = 3

testdata.ix[(testdata.Fare > 26) & (testdata.Fare <= 52.369), 'FareCategory'] = 4

testdata.ix[(testdata.Fare > 52.369), 'FareCategory'] = 5
testdata.info()
testdata['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
testdata.drop(['Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin'], axis=1, inplace=True)
testdata.info()
test = testdata[['Pclass', 'Sex', 'Embarked', 'Party', 'AgeCategory', 'FareCategory']].values

probabilities = model.predict(test, verbose=0)

listOfList = np.round(probabilities).astype(int).tolist()

predictions = [x for obj in listOfList for x in obj]



result = pd.DataFrame( { 'PassengerId': testdata['PassengerId'], 'Survived' : predictions } )

result.shape

result.info()

result



result.to_csv( 'titanic_pred_cleuton.csv' , index = False )
