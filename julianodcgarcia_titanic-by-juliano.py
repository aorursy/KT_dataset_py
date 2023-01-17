# Importing data wrangling and visualization libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# importing the train dataset

df = pd.read_csv('../input/train.csv')



# importing the test dataset

df_pred = pd.read_csv('../input/test.csv')



# And let's check how the data looks like!

df.sample(10)
# Let's check the datatypes

df.info()
# Copy dataframe

df1 = df.iloc[:, :]

df_pred1 = df_pred.iloc[:, :]



# Create a list of both dataframes

both = [df1, df_pred1]
# For the Train Set

df1.isnull().sum()
# and the Test Set

df_pred1.isnull().sum()
for dataset in both:

    dataset['Age'].fillna(dataset['Age'].median(), inplace=True)

    dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)



# Let's delete the cabin attribute and also PassengerId and Ticket, already stated above

df1.drop(['Cabin', 'PassengerId', 'Ticket'], axis=1, inplace=True)
# Let's check the null values again

print(df1.isnull().sum())

print('-'*20)

print(df_pred1.isnull().sum())
for dataset in both:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    dataset['IsAlone'] = dataset['FamilySize'] == 1

    dataset['IsAlone'] = dataset['IsAlone'].astype('int')

    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

    dataset['FareBin'] = pd.cut(dataset['Fare'], 4)

    dataset['AgeBin'] = pd.cut(dataset['Age'].astype('int'), 5)
df1.info()
df1.sample(10)
df1['Title'].value_counts()
# Selects the titles to delete in both datasets

title_del = (pd.concat([df1, df_pred1], sort=False)['Title'].value_counts() < 10)



# Replace them with "Other"

for dataset in both:

    dataset['Title'] = dataset['Title'].apply(lambda x: 'Other' if title_del.loc[x] == True else x)



# Let's see how it looks like

df1['Title'].value_counts()
# Import the library

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# Crete the encoder object

encoder = LabelEncoder()



to_encode = [('Sex', 'Sex_Coded'), ('Embarked', 'Embarked_Coded'), ('Title', 'Title_Coded'), 

             ('FareBin', 'Fare_Coded'), ('AgeBin', 'Age_Coded')]



# Fit and transform using the Train set and transform the test set

for dataset in both:

    for a in to_encode:

        dataset[a[1]] = encoder.fit_transform(dataset[a[0]])
df1.sample(5)
df1_dummy = pd.get_dummies(df1[['Sex', 'Embarked', 'Title']])

df_pred1_dummy = pd.get_dummies(df_pred1[['Sex', 'Embarked', 'Title']])

df1_dummy.head()
# List the features to be used on the analysis

features = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone']



# Set the target

target = ['Survived']
# Let's check the Survived mean for each non continuous attribute

for feature in features:

    if df1[feature].dtype != 'float64' :

        print(df1[[feature, target[0]]].groupby(feature, as_index=False).mean())

        print('-'*20, '\n')
# Let's get visual for the continuous attributes

plt.figure(figsize=[16,12])



# Plot Fare as boxplot to identify outliers

plt.subplot(231)

plt.boxplot(x=df1['Fare'])

plt.title('Fare')

plt.ylabel('Fare ($)')



# Also let's plot Age in a boxplot

plt.subplot(232)

plt.boxplot(df1['Age'])

plt.title('Age')

plt.ylabel('Age (Years)')



# How about family size?

plt.subplot(233)

plt.boxplot(df1['FamilySize'])

plt.title('Family Size')

plt.ylabel('Family Size (#)')



# Now how would Fare affect survivability?

plt.subplot(234)

plt.hist(x = [df1[df1['Survived']==1]['Fare'], df1[df1['Survived']==0]['Fare']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Fare by Survival')

plt.xlabel('Fare ($)')

plt.ylabel('# of Passengers')

plt.legend()



# Age and survivability

plt.subplot(235)

plt.hist(x = [df1[df1['Survived']==1]['Age'], df1[df1['Survived']==0]['Age']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Age by Survival')

plt.xlabel('Age (Years)')

plt.ylabel('# of Passengers')

plt.legend()



# Family size and survivability

plt.subplot(236)

plt.hist(x = [df1[df1['Survived']==1]['FamilySize'], df1[df1['Survived']==0]['FamilySize']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Family Size by Survival')

plt.xlabel('Family Size (#)')

plt.ylabel('# of Passengers')

plt.legend()
#Now let's visualize the 

fig, saxis = plt.subplots(2, 3,figsize=(16,12))



sns.barplot(x = 'Embarked', y = 'Survived', data=df1, ax = saxis[0,0])

sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=df1, ax = saxis[0,1])

sns.barplot(x = 'IsAlone', y = 'Survived', order=[1,0], data=df1, ax = saxis[0,2])



sns.pointplot(x = 'FareBin', y = 'Survived',  data=df1, ax = saxis[1,0])

sns.pointplot(x = 'AgeBin', y = 'Survived',  data=df1, ax = saxis[1,1])

sns.pointplot(x = 'FamilySize', y = 'Survived', data=df1, ax = saxis[1,2])
#graph distribution of qualitative data: Pclass

#we know class mattered in survival, now let's compare class and a 2nd feature

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(14,12))



sns.boxplot(x = 'Pclass', y = 'Fare', hue = 'Survived', data = df1, ax = axis1)

axis1.set_title('Pclass vs Fare Survival Comparison')



sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = df1, split = True, ax = axis2)

axis2.set_title('Pclass vs Age Survival Comparison')



sns.boxplot(x = 'Pclass', y ='FamilySize', hue = 'Survived', data = df1, ax = axis3)

axis3.set_title('Pclass vs Family Size Survival Comparison')
#graph distribution of qualitative data: Sex

#we know sex mattered in survival, now let's compare sex and a 2nd feature

fig, qaxis = plt.subplots(1,3,figsize=(14,12))



sns.barplot(x = 'Sex', y = 'Survived', hue = 'Embarked', data=df1, ax = qaxis[0])

axis1.set_title('Sex vs Embarked Survival Comparison')



sns.barplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data=df1, ax  = qaxis[1])

axis1.set_title('Sex vs Pclass Survival Comparison')



sns.barplot(x = 'Sex', y = 'Survived', hue = 'IsAlone', data=df1, ax  = qaxis[2])

axis1.set_title('Sex vs IsAlone Survival Comparison')
#more side-by-side comparisons

fig, (maxis1, maxis2) = plt.subplots(1, 2,figsize=(14,12))



#how does family size factor with sex & survival compare

sns.pointplot(x="FamilySize", y="Survived", hue="Sex", data=df1,

              palette={"male": "blue", "female": "pink"},

              markers=["*", "o"], linestyles=["-", "--"], ax = maxis1)



#how does class factor with sex & survival compare

sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=df1,

              palette={"male": "blue", "female": "pink"},

              markers=["*", "o"], linestyles=["-", "--"], ax = maxis2)
#how does embark port factor with class, sex, and survival compare

#facetgrid: https://seaborn.pydata.org/generated/seaborn.FacetGrid.html

e = sns.FacetGrid(df1, col = 'Embarked')

e.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', ci=95.0, palette = 'deep')

e.add_legend()
#histogram comparison of sex, class, and age by survival

h = sns.FacetGrid(df1, row = 'Sex', col = 'Pclass', hue = 'Survived')

h.map(plt.hist, 'Age', alpha = .75)

h.add_legend()
df1.head()
pred_atributes = ['Pclass', 'Embarked_Coded', 'IsAlone', 'Sex_Coded', 'Title_Coded', 'Age_Coded']

X = df1[pred_atributes]

Y = df1['Survived']

X_pred = df_pred1[pred_atributes]
# first let us split the dataset into Train and Test

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
# Importing the library

from sklearn import ensemble
# Using random forest model anf fit with Train Set

forest = ensemble.RandomForestClassifier(n_estimators=20)

forest.fit(X_train, Y_train)
# Now let's predict the Test set

Yhat_test = forest.predict(X_test)

Yhat_test[:6]
# Now let us compare

compare = Y_test == Yhat_test

compare.mean()
# Now let's fit the whole dataset

forest.fit(X, Y)

Yhat = forest.predict(X)



final = Y == Yhat

final.mean()
# Now let us predict the Predict Dataset

Y_pred = forest.predict(X_pred)



# Create the output dataframe

output = pd.concat([df_pred[['PassengerId']], pd.Series(Y_pred)], axis=1)

output.rename(columns={0 : 'Survived'}, inplace=True)

output.head()
# Let's export the CSV

output.to_csv('output.csv', index=None)