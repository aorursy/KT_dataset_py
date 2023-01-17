import pandas as pd

import numpy as np

import seaborn as sns

sns.set()

%matplotlib inline

import matplotlib.pyplot as plt

from IPython.display import display

from sklearn.preprocessing import StandardScaler



import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import iplot, init_notebook_mode

import cufflinks

cufflinks.go_offline(connected=True)

init_notebook_mode(connected=True)



import warnings

warnings.filterwarnings('ignore')



def draw_missing_data_table(df):

    total = df.isnull().sum().sort_values(ascending=False)

    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return missing_data
# Path of datasets

titanic_df = pd.read_csv('../input/train.csv')

titanic_df.head()
missing_values = draw_missing_data_table(titanic_df)

display(missing_values)

missing_values[['Percent']].iplot(kind='bar', xTitle='Features', yTitle='Percent of missing values', title='Percent missing data by feature')
figure, axes = plt.subplots(1,1,figsize=(20, 8))

plot = sns.catplot(x="Embarked", y="Fare", hue="Sex", data=titanic_df, palette=('nipy_spectral'), kind="bar", ax=axes)

plt.close(plot.fig)

plt.show()

display(titanic_df[titanic_df['Embarked'].isnull()])
titanic_df['Embarked'].fillna('C', inplace=True)
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)
titanic_df['Cabin'].fillna('U', inplace=True)
draw_missing_data_table(titanic_df[['Cabin', 'Age', 'Embarked']])
# Deck column from letter contained in cabin

titanic_df['Deck'] = titanic_df['Cabin'].str[:1]

titanic_df['Deck'] = titanic_df['Cabin'].map({cabin: p for p, cabin in enumerate(set(cab for cab in titanic_df['Cabin']))})



# Title column from title contained in name

titanic_df['Title'] = pd.Series((name.split('.')[0].split(',')[1].strip() for name in titanic_df['Name']), index=titanic_df.index)

titanic_df['Title'] = titanic_df['Title'].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

titanic_df['Title'] = titanic_df['Title'].replace(['Mlle', 'Ms'], 'Miss')

titanic_df['Title'] = titanic_df['Title'].replace('Mme', 'Mrs')



# Famillysize columns obtained by adding number of sibling and parch

titanic_df['FamillySize'] = titanic_df['SibSp'] + titanic_df['Parch'] + 1

titanic_df['FamillySize'][titanic_df['FamillySize'].between(1, 5, inclusive=False)] = 2

titanic_df['FamillySize'][titanic_df['FamillySize']>5] = 3

titanic_df['FamillySize'] = titanic_df['FamillySize'].map({1: 'Alone', 2: 'Medium', 3: 'Large'})



# IsAlone and IsChild column, quite explicit

titanic_df['IsAlone'] = np.where(titanic_df['FamillySize']!=1, 0, 1)

titanic_df['IsChild'] = titanic_df['Age'] < 18

titanic_df['IsChild'] = titanic_df['IsChild'].astype(int)    
titanic_df = titanic_df.drop(['Name', 'Ticket', 'PassengerId', 'Cabin'], 1)    

titanic_df.head()
titanic_df = pd.get_dummies(data=titanic_df, drop_first=True)

titanic_df.head()
ranges = titanic_df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Deck', 'IsChild']].max().to_frame().T

ranges.iplot(kind='bar', xTitle='Features', yTitle='Range', title='Range of feature before scaling')
X = titanic_df.drop(['Survived'], 1)

y = titanic_df['Survived']



# Feature scaling of our data

sc = StandardScaler()

X = pd.DataFrame(sc.fit_transform(X.values), index=X.index, columns=X.columns)

X.head()