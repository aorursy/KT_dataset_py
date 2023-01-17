# Standard libraries

import pandas as pd

import numpy as np
# Visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv('../input/train.csv')
train.head()
train.info()
train.describe()
train.isnull().head()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='plasma')

# every yellow line means missing data 
# First of all lets plot who survived and who didn't

sns.countplot(x = 'Survived', data = train)

# So about 580 didn't survive 

# And 320 survived
# Lets look at survived with a hue of gender

sns.countplot(x = 'Survived', data = train, hue = 'Sex', palette='RdBu_r')
# Lets look at survived with a hue of Pasenger class

sns.countplot(x = 'Survived', data = train, hue = 'Pclass')
# Lets get an idea about the age of people in the data set

sns.distplot(train['Age'].dropna(), kde= False, bins = 30)
sns.countplot(x = 'SibSp', data = train)

# By looking at this plot, most people on board neither had  siblings / spouses aboard
# Another column which we haven't explored yet is the fare column

train['Fare'].mean()
train['Fare'].hist( bins = 40, figsize = (10,4))

# most of the distribution is between 0 and 100 
plt.figure(figsize = (10,7))

sns.boxplot(x = 'Pclass', y = 'Age', data = train)

# The figure shows that the Passengers in class 1 have older people 

# And younger people in lower Pclass
# Filling in null age values

def substitution(columns):

    Age = columns[0]

    Pclass = columns[1]

    if pd.isnull(Age):

        if Pclass == 1:

            return 36         # approx mean value from blue box

        elif Pclass == 2:

            return 29        # approx mean value from orange box

        else:

            return 23         # approx mean value from green box  

    else:

        return Age           # is not null
train['Age'] = train[['Age', 'Pclass']].apply(substitution, axis = 1)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='plasma')

# no more missing values in Age 
sns.heatmap(train.corr(), annot= True)

# Checking for correlation between columns
train.drop('Cabin',axis=1,inplace=True)

# there are so many missing columns in cabin

# that it seems right to drop it
train.head()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='plasma')

# Final check for null values
pd.get_dummies(train['Sex']).head()

# we need to convert the sex column

# otherwise the machine learning alogorithm won't be able process the data
pd.get_dummies(train['Sex'], drop_first= True).head()

# now you can not feed both these columns as male and female are opposite

# and it will mess up the machine learning algorthim
sex = pd.get_dummies(train['Sex'], drop_first= True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)

# same process with Embarked column
embark.head()
# Since Pclass is also a categorical column

pclass = pd.get_dummies(train['Pclass'],drop_first=True)
train = pd.concat([train, sex, embark, pclass], axis = 1)
train.head()

# now, we don't need sex, embarked, plcass column because we have encoded them.
train.drop(['Sex','Embarked','Name','Ticket', 'Pclass'],axis=1,inplace=True)

# dropping columns which we are not going to use
train.head()

# looks perfect for our machine learning algorithm

# all data is numeric
# Features

X = train.drop('Survived', axis = 1)



# Target variable

y = train['Survived']
from sklearn.ensemble import RandomForestClassifier

# Supervised learning 
model = RandomForestClassifier(n_estimators=2756, max_depth= 5)

model.fit(X,y)
test = pd.read_csv('../input/test.csv')
test.columns
test.info()
test['Age'] = test[['Age', 'Pclass']].apply(substitution, axis = 1)
# Preparing test data according to the model

sex = pd.get_dummies(test['Sex'], drop_first= True)

embark = pd.get_dummies(test['Embarked'],drop_first=True)

pclass = pd.get_dummies(test['Pclass'],drop_first=True)



test = pd.concat([test, sex, embark, pclass], axis = 1)



test.drop(['Sex','Embarked','Name','Ticket', 'Pclass', 'Cabin'],axis=1,inplace=True)
test.columns
# Checking for null values in test

sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='plasma')
test.info()
test.fillna(value=test['Fare'].mean(), inplace= True)
predictions = model.predict(test)
d = {'PassengerId': test['PassengerId'], 'Survived': predictions}

result = pd.DataFrame(d)
result.to_csv('submission.csv', index= False)
result.head()