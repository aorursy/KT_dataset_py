# Imports



# pandas

import pandas as pd

from pandas import Series,DataFrame



import re as re

# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



full_data = [train , test]



train.head(3)
test.head(3)
train.info()

print("----------------------------")

test.info()



#to get useful statistics (count, mean, std... per explanatory variables)

#test.describe() 



# get frequencies of Survived in the train dataset

#train["Survived"].value_counts(normalize = True) 



# Frequencies of Survived per Pclass values 

#train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean() 



# Frequencies of Survived for a specic Sex value

#train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)  

#print(pd.crosstab(train['Title'], train['Sex']))
# How many missing values ? 



# Handling missing values. To calculate number of missing values in each column : 

#True values are coerced as 1 and False as 0 and we use sum function to calculate how many missing values are there in each column:

for col_name in train.columns:

    print (col_name,end=": ")

    print (sum(train[col_name].isnull()))
# Age missing values 



# Easier option : replace missing values by median 



#for dataset in full_data:

#    dataset['Age'] = dataset['Age'].fillna(train['Age'].median())  

    

# More sophisticated : Generate random numbers between (mean - std) and (mean + std)



for dataset in full_data: 

    mean_age = dataset['Age'].mean()

    std_age = dataset['Age'].std() 

    missing_age = dataset['Age'].isnull().sum()

    

    dataset['Age2'] = dataset['Age']

    age_null_random_list = np.random.randint(mean_age - std_age, mean_age + std_age, size=missing_age)

    dataset['Age2'][np.isnan(dataset['Age2'])] = age_null_random_list

    dataset['Age2'] = dataset['Age2'].astype(int)
#we kept Age2 for visualisation purpose: 

    

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

axis1.set_title('Original Age values - Titanic')

axis2.set_title('New Age values - Titanic')



# plot original Age values

# NOTE: drop all null values, and convert to int

train['Age'].dropna().astype(int).hist(bins=70, ax=axis1)

# test_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)



# plot new Age Values

train['Age2'].hist(bins=70, ax=axis2)
# Tidy Age / Age2 (keeping Age2 - without missing values) : Notions of drop and rename 



train.drop('Age', axis=1, inplace=True)

train.rename(columns={'Age2': 'Age'}, inplace=True)

 

test.drop('Age', axis=1, inplace=True)

test.rename(columns={'Age2': 'Age'}, inplace=True)

   

train.head(3)
# Other missing values (Fare, Embarked)



for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())  

    

for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S') #most common value 

    
#train["Sex"][train["Sex"] == "male"] = 0

#train["Sex"][train["Sex"] == "female"] = 1
for dataset in full_data:

    # Mapping Sex

    dataset['Sex_cat'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)  
#train["Embarked"][train["Embarked"] == "S"] = 0

#train["Embarked"][train["Embarked"] == "C"] = 1

#train["Embarked"][train["Embarked"] == "Q"] = 2



# Let's create some graphs to see how Embarked 

sns.factorplot('Embarked','Survived', data=train,size=4,aspect=3)



fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))



sns.countplot(x='Embarked', data=train, ax=axis1)

sns.countplot(x='Survived', hue="Embarked", data=train, order=[1,0], ax=axis2)



#group by embarked, and get the mean for survived passengers for each value in Embarked

embark_perc = train[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()

sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)





# Consider Embarked column in predictions,

# and remove "S" dummy variable, 

# and leave "C" & "Q", since they seem to have a good rate for Survival.



embark_dummies_titanic  = pd.get_dummies(train['Embarked'])

embark_dummies_titanic.drop(['S'], axis=1, inplace=True)



embark_dummies_titanic_test  = pd.get_dummies(test['Embarked'])

embark_dummies_titanic_test.drop(['S'], axis=1, inplace=True)



train = train.join(embark_dummies_titanic)

train.drop(['Embarked'], axis=1,inplace=True)



test = test.join(embark_dummies_titanic_test)

test.drop(['Embarked'], axis=1,inplace=True)
train.head(3)
full_data = [train , test]

#train['Child'] = float('NaN')

#train['Child'][train['Age'] > 18] = 0

#train['Child'][train['Age'] < 18] = 1 



# or 



for dataset in full_data:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare']                               = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare']                                  = 3

    dataset['Fare'] = dataset['Fare'].astype(int)
bins = [0, 18, 34, 50, 200] 

group_names = [1, 2, 3 , 4]



for dataset in full_data:

    categories = pd.cut(dataset['Age'], bins, labels=group_names)

    dataset['Age_cat'] = pd.cut(dataset['Age'], bins, labels=group_names)





# note: 

# to automatically cut in 4 bands: train['CategoricalFare'] = pd.cut(train['Fare'], 4)

 

train.head(3)
# Create new feature FamilySize as a combination of SibSp and Parch

for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    

# Create new feature IsAlone from FamilySize

for dataset in full_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Create Title variable 

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""

# Create a new feature Title, containing the titles of passenger names

for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)

# Group all non-common titles into one single grouping "Rare"

for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



print(train['Title'].value_counts())

print("----------------------------")

print (train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
# Create Mother variable 

for dataset in full_data:

    dataset['Mother'] = 0

    dataset.loc[(dataset['Age'] > 18) & (dataset['Parch'] > 1) & (dataset['Title'] != 'Miss'),"Mother"]=1
train.head(3)
# peaks for survived/not survived passengers by their age

facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend()



# average survived passengers by age

fig, axis1 = plt.subplots(1,1,figsize=(18,4))

average_age = train[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()

sns.barplot(x='Age', y='Survived', data=average_age)
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))



sns.countplot(x='Sex', data=train, ax=axis1)



# average of survived for each Person(male, female, or child)

person_perc = train[["Sex", "Survived"]].groupby(['Sex'],as_index=False).mean()

sns.barplot(x='Sex', y='Survived', data=person_perc, ax=axis2, order=['male','female'])

# As we see, children(age < ~16) on aboard seem to have a high chances for Survival.

# So, we can classify passengers as males, females, and child

def get_person(passenger):

    age,sex = passenger

    return 'child' if age < 16 else sex



for dataset in full_data:

    dataset['Person'] = dataset[['Age','Sex']].apply(get_person,axis=1)
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))



sns.countplot(x='Person', data=train, ax=axis1)



# average of survived for each Person(male, female, or child)

person_perc = train[["Person", "Survived"]].groupby(['Person'],as_index=False).mean()

sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male','female','child'])

import seaborn as sns 

import matplotlib.pyplot as plt

%matplotlib inline



#colormap = plt.cm.plasma

#plt.figure(figsize=(8,8))

#plt.title('Pearson Correlation of Features', y=1.05, size=10)

#sns.heatmap(train.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="YlGnBu", linecolor='black')

#sns.heatmap(train.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
#drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin']

#train = train.drop(drop_elements, axis = 1)

#test  = test.drop(drop_elements, axis = 1)



#train = train.values

#test  = test.values