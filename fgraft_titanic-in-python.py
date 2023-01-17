# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Data Visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Machine Learning

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score

from sklearn.impute import SimpleImputer

# from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Read the data to a pandas data frame

X_train = pd.read_csv('../input/train.csv', index_col='PassengerId')

X_test = pd.read_csv('../input/test.csv', index_col='PassengerId')

print("Data loaded:", X_train.shape, X_test.shape)
# A glimpse on a few rows.

X_train.head()

#X_test.head()
# Column names, types and amount of Nulls

#X_train.info()

X_test.info()
# Summary of numerical features

X_train.describe()

#X_test.describe()
# Summary of non numerical features

#X_train.describe(include=['O'])

X_test.describe(include=['O'])
#X_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#X_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#X_train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

X_train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#g = sns.FacetGrid(X_train, col='Pclass', row='Embarked')

#g = sns.FacetGrid(X_train, col='Survived', row='Pclass')

g = sns.FacetGrid(X_train, col='Survived', row='Embarked')

#g = sns.FacetGrid(X_train, col='Sex', row='Embarked')

#g = sns.FacetGrid(X_train, col='Pclass', row='Embarked')

#g = sns.FacetGrid(X_train, col='Survived', row='Parch')

#g = sns.FacetGrid(X_train, col='Survived', row='SibSp')



g.map(plt.hist, 'Fare', bins=20)

#g.map(plt.hist, 'Pclass')
#ax1 = X_train.plot.scatter(x="Age", y="Fare", c="Survived", colormap="viridis")

#ax1 = X_train.plot.scatter(x="Age", y="Pclass", c="Survived", colormap="viridis")

ax1 = X_train.plot.scatter(x="Age", y="SibSp", c="Survived", colormap="viridis")
X_train['Sex'] = X_train['Sex'].map({'female':1, 'male':0}).astype(int)

X_test['Sex'] = X_test['Sex'].map({'female':1, 'male':0}).astype(int)



X_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
X_train[X_train.Embarked.isnull()]
X_train[(X_train.Survived > 0) & (X_train.Sex > 0) & (X_train.Pclass < 2) & (X_train.SibSp < 1) & (X_train.Parch < 1) & (X_train.Fare > 50)]
# Determining the most frequent port

freq_port = X_train.Embarked.dropna().mode()[0]

freq_port
# Filling the empty values

X_train['Embarked'] = X_train['Embarked'].fillna(freq_port)

X_test['Embarked'] = X_test['Embarked'].fillna(freq_port) #no empty value
# Converting categorical to numerical

X_train['Embarked'] = X_train['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)

X_test['Embarked'] = X_test['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)
X_train = X_train.drop(['Ticket', 'Cabin'], axis=1)

X_test = X_test.drop(['Ticket', 'Cabin'], axis=1)



print("Features dropped: ", X_train.shape, X_test.shape)
#pd.crosstab(X_train.Name.str.extract(' ([A-Za-z]+)\.', expand=False), X_train['Survived'])

#pd.crosstab(X_train.Name.str.extract(' ([A-Za-z]+)\.', expand=False), X_train['Sex'])

pd.crosstab(X_test.Name.str.extract(' ([A-Za-z]+)\.', expand=False), X_test['Sex'])
#Create a new feature called Title

X_train['Title'] = X_train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

X_test['Title'] = X_test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



#Replace the Title with a more common name

X_train['Title'] = X_train['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

X_test['Title'] = X_test['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

X_train['Title'] = X_train['Title'].replace(['Mlle'], 'Miss')

X_train['Title'] = X_train['Title'].replace(['Ms'], 'Miss')

X_train['Title'] = X_train['Title'].replace(['Mme'], 'Mrs')

X_test['Title'] = X_test['Title'].replace(['Mlle'], 'Miss')

X_test['Title'] = X_test['Title'].replace(['Ms'], 'Miss')

X_test['Title'] = X_test['Title'].replace(['Mme'], 'Mrs')



# Survival rate per title

X_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
# Mapping the titles and replacing

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

X_train['Title'] = X_train['Title'].map(title_mapping)

X_train['Title'] = X_train['Title'].fillna(0)

X_test['Title'] = X_test['Title'].map(title_mapping)

X_test['Title'] = X_test['Title'].fillna(0)



# Survival rate per title

X_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
X_train = X_train.drop(['Name'], axis=1)

X_test = X_test.drop(['Name'], axis=1)



print("Features dropped: ", X_train.shape, X_test.shape)

print("New columns:\nX_train: ", X_train.columns, "\nX_test:", X_test.columns)
# Create an empty array

guess_ages = np.zeros((2,3))



for i in range(0, 2):

    for j in range(0, 3):

        guess_df = X_train[(X_train['Sex'] == i) & \

                            (X_train['Pclass'] == j+1)]['Age'].dropna()

        age_guess = guess_df.median()



        # Convert random age float to nearest .5 age

        guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

for i in range(0, 2):

    for j in range(0, 3):

        X_train.loc[ (X_train.Age.isnull()) & (X_train.Sex == i) & (X_train.Pclass == j+1),\

                'Age'] = guess_ages[i,j]



X_train['Age'] = X_train['Age'].astype(int)



for i in range(0, 2):

    for j in range(0, 3):

        guess_df = X_test[(X_test['Sex'] == i) & \

                            (X_test['Pclass'] == j+1)]['Age'].dropna()

        age_guess = guess_df.median()



        # Convert random age float to nearest .5 age

        guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

for i in range(0, 2):

    for j in range(0, 3):

        X_test.loc[ (X_test.Age.isnull()) & (X_test.Sex == i) & (X_test.Pclass == j+1),\

                'Age'] = guess_ages[i,j]



X_test['Age'] = X_test['Age'].astype(int)



# Check amount of nulls in each df.

X_train.info()

print('-'*20)

X_test.info()
X_train['Fare'].fillna(X_train['Fare'].dropna().median(), inplace=True)

X_train.info()
# Separate target from predictors

y_train = X_train.Survived              

X_train = X_train.drop(['Survived'], axis=1)

print("Survival separated:", X_train.shape, X_test.shape, y_train.shape)
# Select categorical columns with relatively low cardinality

low_cardinality_cols = [cname for cname in X_train.columns if X_train[cname].nunique() < 10 and 

                        X_train[cname].dtype == "object"]



# Select numeric columns

numeric_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = low_cardinality_cols + numeric_cols

X_train = X_train[my_cols].copy()

X_test = X_test[my_cols].copy()



# One-hot encode the data with pandas

X_train = pd.get_dummies(X_train)

X_test = pd.get_dummies(X_test)

X_train, X_test = X_train.align(X_test, join='left', axis=1)

X_train.shape, X_test.shape, y_train.shape
def get_score(n_estimators):

    my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()), #the imputer is obsolete if you ran all the data wrangling code

                                    ('model', XGBRegressor(n_estimators=n_estimators,

                                                              learning_rate=0.39

                                                          ))

                             ])

    scores = -1 * cross_val_score(my_pipeline, X_train, y_train,

                                  cv=10,

                                  scoring='neg_mean_absolute_error')

    return scores.mean()



results = {}

for i in range(40,55):

    results[i*1] = get_score(i*1)



plt.plot(results.keys(), results.values())

plt.show()



n_estimators_best = min(results, key=results.get)

print(get_score(n_estimators_best), "from:", n_estimators_best)
# Bundle preprocessing and modeling code in a pipeline

my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()), #the imputer is obsolete if you ran all the data wrangling code

                              ('model', XGBRegressor(n_estimators=48,

                                                              learning_rate=0.39))

                             ])



# Preprocessing of training data, fit model 

my_pipeline.fit(X_train, y_train)



# Preprocessing of test data, fit model

preds_test = my_pipeline.predict(X_test)



# Save test predictions to file

output = pd.DataFrame({'PassengerId': X_test.index,

                       'Survived': preds_test.round(decimals=0, out=None).astype(int)})

output.to_csv('submission.csv', index=False)