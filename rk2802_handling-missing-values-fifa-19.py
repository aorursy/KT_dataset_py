import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import missingno as msno

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LinearRegression

from sklearn.impute import KNNImputer
pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)



data = pd.read_csv('../input/fifa19/data.csv')

data.shape
data.head()
# Required Features....

working_col = ['Name', 'Age', 'Nationality', 'Overall', 'Potential', 'Club', 'Value',

                'Wage', 'Special', 'Preferred Foot', 'International Reputation', 'Weak Foot',

                'Skill Moves', 'Work Rate', 'Body Type', 'Position', 'Height', 'Weight','Crossing',

                'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',

                'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',

                'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',

                'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',

                'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

                'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',

                'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause']
# Working DataFrame....

df = data[working_col]

df.head()
# Cleaning the Value column

def clean_money(column):

    values = []

    for value in data[column]:

        if value[-1]=='M':

            money = 1000000

            money *= float(value[1:-1])

        elif value[-1]=='K':

            money = 1000

            money *= float(value[1:-1])

        else: 

            money = 0

        values.append(money/1000000)

    return values



# Cleaning Weight column

def clean_weight():

    weights = []

    for weight in data['Weight'].fillna(''):

        if weight != '':

            weights.append(int(weight[:-3]))

        else:

            weights.append(np.nan)

    return weights



# Cleaning Height Column

def clean_height():

    heights = []

    for height in data['Height'].fillna(''):

        if height != '':

            height =int(height[0])*12 + int(height[2])

            heights.append(height)

        else:

            heights.append(np.nan)

    return heights



# # Cleaning Release Clause

def clean_release_clause():

    release_clause = []

    for clause in data['Release Clause'].fillna(''):

        if clause == '':

            money=0.0

        elif clause[-1]=='M':

            money = 1000000

            money *= float(clause[1:-1])

        elif clause[-1]=='K':

            money = 1000

            money *= float(clause[1:-1])

        else: 

            money = 0

        release_clause.append(money/1000000)

    return release_clause
df['Value'] =  clean_money('Value')

df['Wage'] = clean_money('Wage')

df['Weight'] = clean_weight()

df['Height'] = clean_height()

df['Release Clause'] = clean_release_clause()
df.isna().sum()
numerical_features =['Age', 'Overall', 'Potential', 'Value', 'Wage', 'Special', 'Height',

                   'Weight', 'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing',

                   'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing',

                   'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions',

                   'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',

                   'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',

                   'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',

                   'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes','Release Clause']



categorical_features = ['Name','Nationality', 'Club', 'Preferred Foot', 'Work Rate','Body Type', 

                        'Position','International Reputation', 'Weak Foot', 'Skill Moves']
df[numerical_features].describe().T
fig = plt.figure(figsize=(15,7))

sns.heatmap(df.isna(), yticklabels=False, cmap='YlGnBu')
fig = plt.figure(figsize=(100,100))

fig.subplots_adjust(hspace=0.4, wspace=0.1)



ax = fig.add_subplot(7, 7, 1)

sns.heatmap(df[categorical_features].isna(), yticklabels=False, cmap='YlGnBu')



ax = fig.add_subplot(7, 7, 2)

sns.heatmap(df[numerical_features].isna(), yticklabels=False, cmap='YlGnBu')
fig = plt.figure(figsize=(20,30))

fig.subplots_adjust(hspace=0.4, wspace=0.4)

count=1

for feature in numerical_features:

    ax = fig.add_subplot(len(numerical_features)//4+1, 4, count)

    sns.boxplot(x=df[feature])

    count +=1
corr_ = df[numerical_features].corr()



f,ax = plt.subplots(figsize=(25, 10))

sns.heatmap(corr_,annot=True, linewidths=0.5, cmap="YlGnBu", fmt= '.1f',ax=ax)

plt.show()
features =  ['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing','Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing','BallControl', 

            'Acceleration', 'SprintSpeed', 'Agility', 'Reactions','Balance', 'ShotPower','Stamina','LongShots','Aggression', 'Interceptions', 

            'Positioning', 'Vision', 'Penalties','Composure', 'Marking','GKDiving','GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']



fig = plt.figure(figsize=(20,30))

fig.subplots_adjust(hspace=0.4, wspace=0.4)

count=1

for feature in features:

    ax = fig.add_subplot(len(features)//4+1, 4, count)

    sns.scatterplot(x=df['Special'], y=df[feature])

    count +=1
sns.scatterplot(df['Balance'], df['Weight'])
fig = plt.figure(figsize=(15,5))

fig.subplots_adjust(hspace=0.4, wspace=0.4)



ax = fig.add_subplot(1, 3, 1)

sns.scatterplot(x=df['Marking'], y=df['StandingTackle'])



ax = fig.add_subplot(1, 3, 2)

sns.scatterplot(x=df['Marking'], y=df['SlidingTackle'])



ax = fig.add_subplot(1, 3, 3)

sns.scatterplot(x=df['StandingTackle'], y=df['SlidingTackle'])
fig = plt.figure(figsize=(5,5))

sns.distplot(df['Height'])
train_df = df[['Special']+features[:-5]].dropna()

test_df = df[df[['Special']+features[:-5]].isnull().any(axis=1)]
for feature in features[:-5]:

    

    polyreg=make_pipeline(PolynomialFeatures(2),LinearRegression())

    polyreg.fit(X = train_df[['Special']], y = train_df[feature])

    

    predicted_output = polyreg.predict(test_df[['Special']])

    test_df[feature] = np.round(predicted_output)

    df[feature].fillna(test_df[feature], inplace=True)
fig = plt.figure(figsize=(20,30))

fig.subplots_adjust(hspace=0.4, wspace=0.4)

count=1

for feature in features[:-5]:

    ax = fig.add_subplot(len(features[:-5])//4+1, 4, count)

    sns.scatterplot(x=train_df[feature], y=train_df['Special'])

    sns.scatterplot(x=test_df[feature], y=test_df['Special'])

    count +=1
train_df = df[['Special']+features[-5:]].dropna()

test_df = df[df[features[-5:]].isnull().any(axis=1)][['Special']+features[-5:]]

imputer = KNNImputer(n_neighbors=1)

imputer.fit(train_df)

predicted_df = pd.DataFrame(np.round(imputer.transform(test_df)), columns=test_df.columns,index=test_df.index)
for col in test_df.columns[1:]:

    df[col].fillna(predicted_df[col], inplace=True)
fig = plt.figure(figsize=(20, 10))

fig.subplots_adjust(hspace=0.4, wspace=0.4)

count=1

for feature in features[-5:]:

    ax = fig.add_subplot(2, 3, count)

    sns.scatterplot(x=train_df[feature], y=train_df['Special'])

    sns.scatterplot(x=predicted_df[feature], y=predicted_df['Special'])

    count +=1
train_df = df[['Weight', 'Balance']].dropna()

test_df = df[df['Weight'].isna()][['Weight', 'Balance']]
polyreg=make_pipeline(PolynomialFeatures(2),LinearRegression())

polyreg.fit(X = train_df[['Balance']], y = train_df['Weight'])



test_df['Weight'] = np.round(polyreg.predict(test_df[['Balance']]))
df['Weight'].fillna(test_df['Weight'], inplace=True)
sns.scatterplot(train_df['Weight'], train_df['Balance'])

sns.scatterplot(test_df['Weight'], test_df['Balance'])
train_df = df[['StandingTackle', 'SlidingTackle', 'Marking']].dropna()

test_df = df[df['SlidingTackle'].isna()][['StandingTackle', 'SlidingTackle', 'Marking']]
for feature in ['StandingTackle', 'SlidingTackle']:

    polyreg=make_pipeline(PolynomialFeatures(2),LinearRegression())

    polyreg.fit(X = train_df[['Marking']], y = train_df[feature])



    test_df[feature] = np.round(polyreg.predict(test_df[['Marking']]))

    df[feature].fillna(test_df[feature], inplace=True)
fig = plt.figure(figsize=(20, 5))

fig.subplots_adjust(hspace=0.4, wspace=0.4)

count=1

for feature in ['StandingTackle', 'SlidingTackle']:

    ax = fig.add_subplot(1, 2, count)

    sns.scatterplot(train_df['Marking'], train_df[feature])

    sns.scatterplot(test_df['Marking'], test_df[feature])

    count +=1



df['Height'].fillna(61.0, inplace=True)

df['Jumping'].fillna(df['Jumping'].mean(), inplace=True)

df['Strength'].fillna(df['Strength'].mean(), inplace=True)
df[categorical_features].isna().sum()
sns.countplot(df['International Reputation'])
sns.countplot(df['Skill Moves'])
sns.countplot(df['Preferred Foot'])
sns.countplot(df['Weak Foot'])
fig = plt.figure(figsize=(15,5))

sns.countplot(df['Position'])
fig = plt.figure(figsize=(15,5))

sns.countplot(df['Body Type'])
fig = plt.figure(figsize=(15, 5))

sns.countplot(df['Work Rate'])
df['Club'].fillna('No Club', inplace=True)

df['Preferred Foot'].fillna('Right', inplace=True)

df['Weak Foot'].fillna(3.0, inplace=True)

df['International Reputation'].fillna(1.0, inplace=True)

df['Body Type'].fillna('Normal',inplace=True)

df['Work Rate'].fillna('Medium/Medium', inplace=True)

df['Position'].fillna('NA', inplace=True)

df['Skill Moves'].fillna(2.0, inplace=True)
fig = plt.figure(figsize=(15,7))

sns.heatmap(df.isna(), yticklabels=False, cmap='YlGnBu')