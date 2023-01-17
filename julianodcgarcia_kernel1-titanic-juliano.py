import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import tree

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.preprocessing import MinMaxScaler

%matplotlib inline
# Defining the filepath for each dataset

train_path = '../input/train.csv'

test_path = '../input/test.csv'



# Importing as a pandas dataframe

train = pd.read_csv(train_path)

test = pd.read_csv(test_path)



# let's take a look on the train set

print(train.shape)

train.head(10)
# Checking if the provided test set is in the same format

print(test.shape)

test.head(10)
# Checking missing data with .innull() method

missing_data = train.isnull()

missing_data.head()
missing_data.describe()
train.drop("Cabin", axis=1, inplace=True)

train.head()
mode_embarked = train["Embarked"].mode()

train['Embarked'].replace(np.nan, mode_embarked, inplace=True)
avg_age = train['Age'].astype('float').mean(axis=0)

train['Age'].replace(np.nan, avg_age, inplace=True)
train.dtypes
train.describe(include='all')
train.corr()
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
# Let's see the Age distribution of the survivors

plt.hist(train[train['Survived'] == 1]['Age'])
# Now not the so lucky ones

plt.hist(train[train['Survived'] == 0]['Age'])
# Binning the Age distribution

bins = [0., 10., 20., 30., 40., 50., 60., 70., 80.]

bin_name = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80']

train['binned-age'] = pd.cut(train['Age'], bins, labels=bin_name, include_lowest=True )

train[['Age', 'binned-age']].head(10)
boat_pref = train[['Sex', 'binned-age', 'Survived']].groupby(['Sex', 'binned-age'], as_index=False).mean()

boat_pivot = boat_pref.pivot(index='Sex',columns='binned-age')

boat_pivot
# Let's define a Plot function

def heatPlot(pivot):

    fig, ax = plt.subplots()

    im = ax.pcolor(pivot, cmap='RdBu')



    #label names

    row_labels = pivot.columns.levels[1]

    col_labels = pivot.index



    #move ticks and labels to the center

    ax.set_xticks(np.arange(pivot.shape[1]) + 0.5, minor=False)

    ax.set_yticks(np.arange(pivot.shape[0]) + 0.5, minor=False)



    #insert labels

    ax.set_xticklabels(row_labels, minor=False)

    ax.set_yticklabels(col_labels, minor=False)



    #rotate label if too long

    plt.xticks(rotation=90)



    fig.colorbar(im)

    plt.show()
heatPlot(boat_pivot)
age_sibsp = train[['SibSp', 'binned-age', 'Survived']].groupby(['SibSp', 'binned-age'], as_index=False).mean()

sibsp_pivot = age_sibsp.pivot(index='SibSp', columns='binned-age')
heatPlot(sibsp_pivot)
age_parch = train[['Parch', 'binned-age', 'Survived']].groupby(['Parch', 'binned-age'], as_index=False).mean()

parch_pivot = age_parch.pivot(index='Parch', columns='binned-age')
heatPlot(parch_pivot)
sns.boxplot(x='Pclass', y='Fare', data=train)
class1_df = train[train['Pclass'] == 1]

class1_df = class1_df[['Pclass', 'Fare', 'Survived']]

class1_df.shape
sns.boxplot(x='Survived', y='Fare', data=class1_df)
# Creting our matrix of features X, and vector of dependent variable y

X = train[['Embarked', 'Pclass', 'Sex', 'Age']]

y = train['Survived']
dummy_emb = pd.get_dummies(X['Embarked'])

dummy_emb.rename(columns={'C' : 'Embarked_C', 'Q' : 'Embarked_Q', 'S' : 'Embarked_S'}, inplace=True)

dummy_emb.head()
X = pd.concat([X.iloc[:, 1:], dummy_emb.iloc[:, 0:2]], axis=1)
dummy_sex = pd.get_dummies(X['Sex'])

dummy_sex.head()
X = pd.concat([X, dummy_sex.iloc[:, 0]], axis=1)

X.drop("Sex", axis=1, inplace=True)
X.head()
sc_X = MinMaxScaler()

X[['Age']] = sc_X.fit_transform(X[['Age']])

X.head()
survival = tree.DecisionTreeClassifier()
survival.fit(X, y)
yhat = survival.predict(X)
accuracy = y == yhat

accuracy.value_counts()
test.head()
X_pred = test[['Pclass', 'Age']]
test_blank = X_pred.isnull()

for column in test_blank.columns.values.tolist():

    print (test_blank[column].value_counts())

    print("")
X_pred.replace(np.nan, X_pred['Age'].mean(axis=0), inplace=True)
dummy_emb = pd.get_dummies(test['Embarked'])

dummy_emb.rename(columns={'C' : 'Embarked_C', 'Q' : 'Embarked_Q', 'S' : 'Embarked_S'}, inplace=True)

X_pred = pd.concat([X_pred, dummy_emb.iloc[:, 0:2]], axis=1)

X_pred.head()
dummy_sex = pd.get_dummies(test['Sex'])

X_pred = pd.concat([X_pred, dummy_sex.iloc[:, 0]], axis=1)

X_pred.head()
X_pred[['Age']] = sc_X.transform(X_pred[['Age']])

X_pred.head()
y_pred = survival.predict(X_pred)

y_pred[:6]
output = pd.concat([test['PassengerId'], pd.Series(y_pred).to_frame()], axis=1)
output.rename(columns={0 : 'Survived'}, inplace=True)

output.head()
output.to_csv('titanic_prdictions.csv', index=None)