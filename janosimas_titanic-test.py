# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv as csv

import re

from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#visualization

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline
pd.options.mode.chained_assignment = None
test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")
def refactorDataFrame(dataFrame, titles):

    # female = 0, Male = 1

    dataFrame['Gender'] = dataFrame['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    

    # All missing Embarked -> just make them embark from most common place

    if len(dataFrame.Embarked[ dataFrame.Embarked.isnull() ]) > 0:

        dataFrame.Embarked[ dataFrame.Embarked.isnull() ] = dataFrame.Embarked.dropna().mode().values

        

    dataFrame["EmbarkedInt"] = 0

    Ports = list(enumerate(np.unique(dataFrame['Embarked'])))    # determine all values of Embarked,

    Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index

    dataFrame.EmbarkedInt = dataFrame.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

    

    # All the ages with no data -> make the median of all Ages

    median_age = dataFrame['Age'].dropna().median()

    if len(dataFrame.Age[ dataFrame.Age.isnull() ]) > 0:

        dataFrame.loc[ (dataFrame.Age.isnull()), 'Age'] = median_age

    

    dataFrame["FareGroup"] = 0

    for i in range (10, 121, 15) :

        dataFrame.loc[(dataFrame.Fare < i) & (dataFrame["Fare"] > (i-10)), "FareGroup"] = i

    

    dataFrame["CabinLetter"] = 0

    dataFrame.CabinLetter = dataFrame.Cabin.map( lambda x: x[0:1] if type(x) == type("str") else None )

    

    dataFrame["CabinInt"] = 0

    Cabins = list(enumerate(np.unique(dataFrame['CabinLetter'][dataFrame['CabinLetter'].isnull() == False])))

    Cabins_dict = { name : i for i, name in Cabins }

    dataFrame.CabinInt = dataFrame.CabinLetter.map( lambda x: Cabins_dict[x] if type(x) == type("str") else None )



    dataFrame["Child"] = 0

    dataFrame.Child = dataFrame.Age.map( lambda x: 1 if x < 16 else 0 )

    

    dataFrame["FamilySize"] = 0

    dataFrame.FamilySize = dataFrame["SibSp"] + dataFrame["Parch"]

    

    dataFrame['TitlesInt'] = 0

    dataFrame['Titles'] = 0

    dataFrame.Titles = dataFrame.Name.map( lambda x: re.search('.*(,\ (.*?)\.\ ).*', x).group(2) )

    

    dataFrame.Titles['Mlle'] = 'Miss'

    dataFrame.Titles['Ms'] = 'Miss'

    dataFrame.Titles['Mme'] = 'Mrs'

    

        # Titles with very low cell counts to be combined to "rare" level

    rare_title = ['Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']

    dataFrame.Titles = dataFrame.Titles.map(lambda x: 'rare' if x in rare_title else x)

    

    if not titles:

        titles = list(enumerate(np.unique(dataFrame.Titles)))    # determine all values of Titles,

    

    print(titles)

    titles_dict = { name : i for i, name in titles }

    dataFrame.TitlesInt = dataFrame.Titles.map( lambda x: titles_dict[x]).astype(int)
titles = []

refactorDataFrame(train, titles)

refactorDataFrame(test, titles)



# All the missing Fares -> assume median of their respective class

if len(test.Fare[ test.Fare.isnull() ]) > 0:

    median_fare = np.zeros(3)

    for f in range(0,3):                                              # loop 0 to 2

        median_fare[f] = test[ test.Pclass == f+1 ]['Fare'].dropna().median()

    for f in range(0,3):                                              # loop 0 to 2

        test.loc[ (test.Fare.isnull()) & (test.Pclass == f+1 ), 'Fare'] = median_fare[f]

        
test.info()
train.describe()
train.head()
sns.factorplot(x="Pclass", y="Survived", data=train)
sns.factorplot(x="TitlesInt", y="Survived", data=train)
# plot

sns.factorplot('Embarked','Survived', data=train, aspect=2)



fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(10,5))



# sns.factorplot('Embarked',data=titanic_df,kind='count',order=['S','C','Q'],ax=axis1)

# sns.factorplot('Survived',hue="Embarked",data=titanic_df,kind='count',order=[1,0],ax=axis2)

sns.countplot(x='Embarked', data=train, ax=axis1)

sns.countplot(x='Survived', hue="Embarked", data=train, order=[1,0], ax=axis2)



# group by embarked, and get the mean for survived passengers for each value in Embarked

embark_perc = train[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()

sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)
# convert from float to int

train['FareInt'] = 0

train['FareInt'] = train['Fare'].astype(int)

test_fare    = train['Fare'].astype(int)



# get fare for survived & didn't survive passengers 

fare_not_survived = train['FareInt'][train["Survived"] == 0]

fare_survived     = train['FareInt'][train["Survived"] == 1]



# get average and std for fare of survived/not survived passengers

avgerage_fare = pd.DataFrame([fare_not_survived.mean(), fare_survived.mean()])

std_fare      = pd.DataFrame([fare_not_survived.std(), fare_survived.std()])



# plot

train['FareInt'].plot(kind='hist', bins=100, xlim=(0,50))



avgerage_fare.index.names = std_fare.index.names = ["Survived"]

avgerage_fare.plot(yerr=std_fare,kind='bar',legend=False)
sns.factorplot('FareGroup','Survived', data=train, aspect=2)
sns.factorplot(x="Child", y="Survived", data=train)
train.columns
predictors = ['Pclass', 'Age', 'FareGroup', 'EmbarkedInt', 'Gender', 'SibSp', 'Parch', 'TitlesInt']

forest = RandomForestClassifier(n_estimators=100)

forest = forest.fit( train[predictors], train['Survived'])

forest.score(train[predictors], train['Survived'])
output = forest.predict(test[predictors]).astype(int)
predictions_file = open("titanic_predictions.csv", "w")

open_file_object = csv.writer(predictions_file)

open_file_object.writerow(["PassengerId","Survived"])

open_file_object.writerows(zip(test['PassengerId'].values, output))

predictions_file.close()