# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import matplotlib.pyplot as plt

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import Series, DataFrame

import statsmodels.api as sm

from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Read in data

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')



full_data = pd.concat([train_data, test_data])



full_data.head(20)
#Extract the title from the Name

full_data['Title'] = full_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0])



pd.crosstab(full_data['Sex'], full_data['Title'], margins=True)
#Group the Titles

rare_title = ["Dona", "Don", "Capt", "Col", "Jonkheer", "Lady", "Major", "Rev", "the Countess", "Sir", "Dr"]

def titleCombine(value):

    if "Mlle" in value:

        return 'Miss'

    elif "Ms" in value:

        return 'Miss'

    elif "Mme" in value:

        return 'Mrs'

    elif "Miss" in value:

        return 'Miss'

    elif "Mrs" in value:

        return 'Mrs'

    elif any(x in value for x in rare_title):

        return "Rare Title"

    else:

        return value





full_data['Title'] = full_data['Title'].apply(titleCombine)



pd.crosstab(full_data['Sex'], full_data['Title'], margins=True)
#Drop unused columns.

full_data = full_data.drop(['Name', 'Ticket'], axis=1)
#Create and clean Deck var.

full_data['Deck'] = full_data['Cabin'].astype(str).str[0]

full_data['Deck'] = full_data['Deck'].apply(lambda x: "" if 'n' in x else x)
#Folks without Embarked

#full_data.ix[(full_data['Embarked'] != 'S') & (full_data['Embarked'] != 'Q') & (full_data['Embarked'] != 'C')]



#Location of missing fare

full_data.iloc[1043]



#Take median of matching fair attributes, and drop null row

median_for_missing_fare = full_data.loc[(full_data['Pclass'] == 3) & (full_data['Embarked'] == 'S')]

median_for_missing_fare = median_for_missing_fare[pd.notnull(median_for_missing_fare['Fare'])]

#median_for_missing_fare.Fare.hist()

plt.hist(median_for_missing_fare.Fare, bins=20)

plt.axvline(median_for_missing_fare.Fare.median(), color='r', linestyle='dashed', linewidth = 3)

plt.text (median_for_missing_fare.Fare.median()-3,340, str(median_for_missing_fare.Fare.median()), fontsize=15)



#Replace missing Fare value with Median

full_data["Fare"].fillna(median_for_missing_fare["Fare"].median(), inplace=True)



#Dropping Embarked, because it should have no effect.

full_data = full_data.drop('Embarked',axis=1)
#Let's work on Age

age_avg = full_data.Age.mean()

std_age = full_data.Age.std()

nan_age = full_data.Age.isnull().sum()



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

axis1.set_title("Original Age values")

axis2.set_title("New Age values")

#plot original values

full_data.Age.dropna().astype(int).hist(bins=80, ax=axis1)



#Generate random values within 1 standard deviation of the mean Age. I would prefer to use a method like MICE

#But I can't find a great library for that robust of imputation.

rand = np.random.randint(age_avg - std_age, age_avg + std_age, size = nan_age)

full_data['Age'][np.isnan(full_data['Age'])] = rand

full_data.Age = full_data.Age.astype(int)



full_data.Age.hist(bins=80, ax=axis2)
full_data.drop('Cabin',axis=1,inplace=True)
def assign_child(psngr):

    age, sex = psngr

    return 'child' if age < 16 else sex



full_data['Person'] = full_data[['Age','Sex']].apply(assign_child,axis=1)

full_data.drop("Sex",axis=1)



train = full_data[0:889]

train_y = train['Survived']

train_x = train.drop(['PassengerId', 'Survived', 'Deck'], axis=1)

test = full_data[890:1308]

test_x = test.drop(['PassengerId', 'Survived', 'Deck'], axis=1)



le = preprocessing.LabelEncoder()

train_x.Title = le.fit_transform(train_x.Title)

train_x.Sex = le.fit_transform(train_x.Sex)

train_x.Person = le.fit_transform(train_x.Person)

test_x.Title = le.fit_transform(test_x.Title)

test_x.Person = le.fit_transform(test_x.Person)

test_x.Sex = le.fit_transform(test_x.Sex)
#Predict!

random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(train_x, train_y)



Y_pred = random_forest.predict(test_x)



random_forest.score(train_x, train_y)
#Submit

submission = pd.DataFrame({

    "PassengerId": test["PassengerId"],

    "Survived": Y_pred

})

submission.to_csv('titanic.csv', index=False)