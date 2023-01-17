# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



import numpy as np

import pandas as pd

# data visualization

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style



from sklearn.ensemble import RandomForestClassifier



# Ignore Warning

import warnings

warnings.filterwarnings("ignore")

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

train_df.info()
# What is missing and how much missing

total = train_df.isnull().sum().sort_values(ascending=False)

percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent_2],axis=1, keys=['Missing Total', '%'])

missing_data.head()
data = [train_df, test_df]

for dataset in data:

    dataset['relatives'] = dataset['SibSp']+dataset['Parch']

    dataset.loc[dataset['relatives'] > 0, 'not_alone']=0

    dataset.loc[dataset['relatives'] == 0, 'not_alone']=1

    dataset['not_alone'] = dataset['not_alone'].astype(int)



train_df['not_alone'].value_counts()
# Dropping PassengerId column from training data as it is not affecting servival rate

train_df = train_df.drop(['PassengerId'], axis=1)
#CABIN:

import re

deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}

data = [train_df, test_df]

for dataset in data:

    dataset['Cabin'] = dataset['Cabin'].fillna('U0')

    dataset['Deck'] = dataset['Cabin'].map(lambda x : re.compile("([a-zA-Z]+)").search(x).group())

    dataset['Deck'] = dataset['Deck'].map(deck)

    dataset['Deck'] = dataset['Deck'].fillna(0)

    dataset['Deck'] = dataset['Deck'].astype(int)

    

#train_df['Deck']

# we can now drop the cabin feature

train_df = train_df.drop(['Cabin'], axis=1)

test_df = test_df.drop(['Cabin'], axis=1)
#AGE:

#create an array that contains random numbers,

#which are computed based on the mean age value in regards to the standard deviation and is_null.

data = [train_df, test_df]

for dataset in data:

    mean = train_df['Age'].mean()

    std = test_df['Age'].std()

    is_null = dataset["Age"].isnull().sum()

    # compute random numbers between the mean, std and is_null

    rand_age = np.random.randint(mean - std, mean + std, size = is_null)

    # fill NaN values in Age column with random values generated

    age_slice = dataset["Age"].copy()    

    age_slice[np.isnan(age_slice)] = rand_age

    dataset["Age"] = age_slice

    dataset["Age"] = train_df["Age"].astype(int)

    

#train_df["Age"].isnull().sum()
#Embarked:

#Embarked feature has only 2 missing values

#fill these with the most common one

#train_df['Embarked'].describe() #top S

common_value = 'S'

data = [train_df, test_df]

for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)
train_df.info()
#Fare:

#Converting “Fare” from float to int64

data = [train_df, test_df]

for dataset in data:

    dataset['Fare'] = dataset['Fare'].fillna(0)

    dataset['Fare'] = dataset['Fare'].astype(int)
#Name:

#Use the Name feature to extract the Titles from the Name

#ANd build a new feature out of that

data = [train_df, test_df]

titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:

    # extract titles

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    # replace titles with a more common title or as Rare

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',

                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    # convert titles into numbers

    dataset['Title'] = dataset['Title'].map(titles)

    # filling NaN with 0, to get safe

    dataset['Title'] = dataset['Title'].fillna(0)

    

#Drop the Name feature now from both train and test dataframe

train_df = train_df.drop(['Name'], axis=1)

test_df = test_df.drop(['Name'], axis=1)



#train_df.head()
#Sex:

#Convert ‘Sex’ feature into numeric

genders = {"male": 0, "female": 1}

data = [train_df, test_df]

for dataset in data:

    dataset['Sex'] = dataset['Sex'].map(genders)
#Ticket:

#train_df['Ticket'].describe()

#Since the Ticket attribute has 681 unique tickets, 

#it will be a bit tricky to convert them into useful categories. So we will drop it from the dataset.

train_df = train_df.drop(['Ticket'],axis=1)

test_df = test_df.drop(['Ticket'],axis=1)
#Embarked:

#Convert ‘Embarked’ feature into numeric

ports = {"S": 0, "C": 1, "Q": 2}

data = [train_df, test_df]

for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].map(ports)
#Group Age:

data = [train_df, test_df]

for dataset in data:

    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[dataset['Age'] <= 11, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6

    dataset.loc[dataset['Age'] > 66, 'Age'] = 6

# let's see how it's distributed train_df['Age'].value_counts()

train_df['Age'].value_counts()
#Group Fare:

#train_df.head()

data = [train_df, test_df]



for dataset in data:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3

    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4

    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5

    dataset['Fare'] = dataset['Fare'].astype(int)

    

train_df['Fare'].value_counts()
#  two new features to the dataset

# 1:Age times Class

# 2:Fare per Person

data = [train_df, test_df]

for dataset in data:

    dataset['Age_Class']= dataset['Age']* dataset['Pclass']



for dataset in data:

    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)

    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)

    

# Let's take a last look at the training set, before we start training the models.

train_df.head(10)
X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)



Y_prediction = random_forest.predict(X_test)



score = random_forest.score(X_train, Y_train)

acc_random_forest = round(score * 100, 2)

print(acc_random_forest)
output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': Y_prediction})

output.to_csv('RF-Submission.csv', index=False)

print("Your submission was successfully saved!")