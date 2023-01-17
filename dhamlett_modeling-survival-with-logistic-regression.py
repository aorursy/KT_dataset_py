import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import collections as cs

import plotly

import plotly.plotly as py

import plotly.graph_objs as go

import statsmodels.api as sm

from scipy import stats

from sklearn.linear_model import LogisticRegression



## Set default Seaborn plot style

sns.set(style="darkgrid")



## Import Titanic test data - confirm all required variables available

test_df = pd.read_csv("../input/test.csv")



## Preview results

test_df.head(5)



## Results confirm data set does not contain all required variables for analysis
## Import Titanic training data - Test data set does not include survival as a variable

train_df = pd.read_csv("../input/train.csv")



## Preview results

train_df.head(5)



## Results confirm data set does contain all required variables for analysis (inclusive of Survived as a dependent variable)
## Analyze the data

train_df.info()



## Results confirm that there are 891 records total

## Null values appear in the data set (Age - 177; Cabin - 687; Embarked - 2)

## Do we need all of these variables in the model?
## Let's assess cabin's, and assign a grouping by cabin type.

train_df['Cabin Cleansed'] = train_df['Cabin'].astype(str).str[0]

train_df['Cabin_Cleansed'] = train_df['Cabin Cleansed'].replace('n','Unassigned')

#train_df['Cabin_Cleansed'] = train_df['Cabin_Cleansed'].replace('n','Unassigned')



## Let's build a chart to further analyze the data

plt.figure(figsize=(15,7))

ax = sns.countplot(y="Cabin_Cleansed", data=train_df, order = train_df['Cabin_Cleansed'].value_counts().index)

ax.set(xlabel='Passengers', ylabel='Cabin Group')



## How can we further interpret the results?

cabin_null = train_df['Cabin'].isnull().sum()

total_pass = len(train_df)

print("{0:.0f}%".format(round((cabin_null/total_pass),2) * 100) + " or (" + str(cabin_null) + ") of passengers are not assigned to a cabin.")
## Let's look at how being assigned to a cabin affected survival . . . maybe we don't have to replace the NULL values.

train_df['Assigned Cabin'] = train_df['Cabin'].isnull()



## Time to plot the data

plt.figure(figsize=(15,7))

bx = sns.countplot(x='Assigned Cabin', hue='Survived', data=train_df)

bx.set(ylabel='Passengers')



## It appears as though passengers who were assigned a cabin were less likely to survive

## We may want to include this variable in our model

train_df['Assigned_Cabin'] = train_df['Assigned Cabin']*1
## Now let's take a look at the Age variable, which also had several NULL values

plt.figure(figsize=(15,7))

sns.distplot(train_df['Age'].dropna(), kde=False, bins = 30)



## NULL values needed to be replaced to build this chart, let's uncover a strategy to fill in these values.
## Could we potentially predict age, based off of the fare paid?

plt.figure(figsize=(15,7))

sns.regplot(x="Age", y="Fare", data=train_df)



# No visible correlation between the variables, is there a better way?
# Let's observe the spread of values between the Pclass and Age/Sex variables.

plt.figure(figsize=(15,7))

sns.boxplot(x="Pclass", y="Age", hue="Sex", data=train_df)



# Not to much spread between males/females, but there is certainly a spread between Pclass and Age.
## Let's fill in the NULL age values, based off of the Pclass

train_df.Age = train_df.groupby(['Pclass'])['Age'].apply(lambda x: x.fillna(x.mean()))

train_df.Age = train_df.Age.fillna(train_df.Age.mean())



## Let's update our distribvution chart (without having to drop NULL values)

plt.figure(figsize=(15,7))

sns.distplot(train_df['Age'], kde=False, bins = 30)



## We are now closer to a normal distribution for this variable
## Now let's take a look at the fare distribution

plt.figure(figsize=(15,7))

sns.distplot(train_df['Fare'], kde=False, bins = 30)



## This data is heavily right skewed.
## Let's apply a log transformation and check the distribution.

train_df['Log_Fare'] = train_df.apply(lambda row: 0 if row['Fare'] in (0,1) else np.log10(row.Fare), axis=1)



## Time to plot the data.

plt.figure(figsize=(15,7))

sns.distplot(train_df['Log_Fare'], kde=False, bins = 30)



## While the data is still skewed, it is reflecting a more normal distribution.
## Let's check the last variable where there are NULL values

plt.figure(figsize=(15,7))

cx = sns.countplot(x='Embarked',data=train_df)

cx.set(ylabel='Passengers')

print("There are " + str(train_df['Embarked'].isnull().sum()) + " passengers with no defined origin.")



## Given that there are only a few folks who are missing an Embarked value, we can assume they originated from the most popular origin.
## Let's find out who these passengers are, and what additional details we can infer

train_df[train_df['Embarked'].isnull()]



## Given there is no clear details to help derive the point of embarkment, we will leverage the most popular embarkement value 'S'.
## Let's replace both NULL values with the most common embarkment point

train_df.loc[train_df['Embarked'].isnull(), 'Embarked'] = 'S'
## Analyze the data again, to confirm critical NULL values have been updated

train_df.info()
## Let's look at another view variables, to get an understanding for this data

plt.figure(figsize=(15,7))

dx = sns.countplot(x='Sex', hue='Survived', data=train_df)

dx.set(ylabel='Passengers')



## Looks like quite a few more female passengers survived
## Now let's take a look at the SibSp (Sibling/Spouse) variable

plt.figure(figsize=(15,7))

ex = sns.countplot(x='SibSp', hue='Survived', data=train_df)

ex.set(ylabel='Passengers')



## The ratio of passengers who survived vs. perished is pretty evenly split for all sibling/spouse groupings, except for single passengers where the survival rate is significantly smaller.
## Now let's take a look at the Parch (Parents/Children) variable

plt.figure(figsize=(15,7))

fx = sns.countplot(x='Parch', hue='Survived', data=train_df)

fx.set(ylabel='Passengers')



## Similar results as seen with the SibSp variable, where the survival rate of single passengers is significantly smaller.
## As single passengers have drastically higher popultations, let's simply these variables

train_df['Family'] = train_df['SibSp'] + train_df['Parch'] + 1

train_df['Alone'] = 0

train_df.loc[train_df['Family']==1,'Alone'] = 1
## Let's review all of the variables that we have in play

train_df.head(5)
## Now we need to create dummy variables for our categorical data

Class = pd.get_dummies(train_df['Pclass'], prefix='Class', drop_first=True)

Sex = pd.get_dummies(train_df['Sex'], prefix='Sex', drop_first=True)

Embarked = pd.get_dummies(train_df['Embarked'], prefix='Embarked', drop_first=True)



## Let's merge our existing data set w/ the new variables

train_df = pd.concat([train_df, Class, Sex, Embarked], axis=1)



## Let's review results - quite a few columns to be removed

train_df.head(5)
## We need to start removing variables, as we prepare for our final model

train_df.drop(['PassengerId'], axis=1, inplace=True) #High cardinality

train_df.drop(['Pclass'], axis=1, inplace=True) #Dummy variable created

train_df.drop(['Name'], axis=1, inplace=True) #High cardinality

train_df.drop(['Sex'], axis=1, inplace=True) #Dummy variable created

train_df.drop(['SibSp'], axis=1, inplace=True) #New variable created

train_df.drop(['Parch'], axis=1, inplace=True) #New variable created

train_df.drop(['Ticket'], axis=1, inplace=True) #High cardinality

train_df.drop(['Cabin'], axis=1, inplace=True) #New variable created

train_df.drop(['Embarked'], axis=1, inplace=True) #Dummy variable created

train_df.drop(['Cabin Cleansed'], axis=1, inplace=True) #New variable created

train_df.drop(['Cabin_Cleansed'], axis=1, inplace=True) #New variable created

train_df.drop(['Assigned Cabin'], axis=1, inplace=True) #New variable created

train_df.drop(['Family'], axis=1, inplace=True) #New variable created
## Let's review the results, and our final data frame

train_df.head(5)
## Let's select the columns we want to leverage in our model

features = ["Age", "Fare", "Log_Fare", "Assigned_Cabin", "Alone", "Class_2", "Class_3", "Sex_male", "Embarked_Q", "Embarked_S"]

X = train_df[features]

Y = train_df['Survived']
## Time to execute our prediction

stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

logit_model = sm.Logit(Y,X)

result = logit_model.fit()

print(result.summary())



## A few of the variables are reflecting P-values greater than an alpha of .05 . . . let's strip those out.
## Let's select the columns we want to leverage in our model

features_2 = ["Fare", "Alone", "Class_2", "Sex_male",]

X_2 = train_df[features_2]



## Our original Log transformation on the Fare variable did not prove to be too valuable in this model
## Time to execute our NEW prediction

stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

logit_model = sm.Logit(Y,X_2)

result = logit_model.fit()

print(result.summary())



## These variables look good
## Model 1 results

logreg = LogisticRegression()

logreg.fit(X, Y)

logreg.score(X, Y)
## Model 2 Results

logreg.fit(X_2, Y)

logreg.score(X_2, Y)



## Given that our test data set did not contain a survived variable (reason we used the primary training data set all the way through), when applying this model to the entire data set, and after refining our variable selection, we received a logistic regression score of 78% . . . this is pretty darn good!