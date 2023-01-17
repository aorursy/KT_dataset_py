import numpy as np 
import pandas as pd 
import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn import model_selection

import xgboost as xgb

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]

print(train_df.columns.values)

# preview the data
train_df.head()

train_df.tail()

train_df.info()
print('_'*40)
test_df.info()
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)   #Drop ticket price and Cabin 
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape

# Extract titles from the names data.
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()
# drop the name columns and passengerID columns

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape
#convert the sex to numerical data
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
guess_ages = np.zeros((2,3))
guess_ages
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()


#convert age to age bands
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df.head()
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

train_df.head()
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
freq_port = train_df.Embarked.dropna().mode()[0]
freq_port

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
### Converting categorical feature to numeric


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df.head(10)


X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log
from xgboost import XGBClassifier, plot_importance
model = XGBClassifier()
model.fit(X_train, Y_train)

from matplotlib import pyplot
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.show()
print(model.feature_importances_)

plot_importance(model) # plot feature importances

# USE TOP 5 Features in for training and get the accuracy
x_new=X_train[["Fare","Title","Embarked","Pclass","Age"]]
x_test_new=X_test[["Fare","Title","Embarked","Pclass","Age"]]
logreg = LogisticRegression()
logreg.fit(x_new, Y_train)
Y_pred = logreg.predict(x_test_new)
acc_log = round(logreg.score(x_new, Y_train) * 100, 2)
acc_log
# USE TOP 6 Features in for training and get the accuracy
x_new=X_train[["Fare","Title","Embarked","Pclass","Age","IsAlone"]]
x_test_new=X_test[["Fare","Title","Embarked","Pclass","Age","IsAlone"]]
logreg = LogisticRegression()
logreg.fit(x_new, Y_train)
Y_pred = logreg.predict(x_test_new)
acc_log = round(logreg.score(x_new, Y_train) * 100, 2)
acc_log
# USE TOP 7 Features in for training and get the accuracy
x_new=X_train[["Fare","Title","Embarked","Pclass","Age","IsAlone","Age*Class"]]
x_test_new=X_test[["Fare","Title","Embarked","Pclass","Age","IsAlone","Age*Class"]]
logreg = LogisticRegression()
logreg.fit(x_new, Y_train)
Y_pred = logreg.predict(x_test_new)
acc_log = round(logreg.score(x_new, Y_train) * 100, 2)
acc_log
# USE TOP  Features in for training and get the accuracy
x_new=X_train[["IsAlone","Pclass","Sex","Age","Age*Class"]]
x_test_new=X_test[["IsAlone","Pclass","Sex","Age","Age*Class"]]
logreg = LogisticRegression()
logreg.fit(x_new, Y_train)
Y_pred = logreg.predict(x_test_new)
acc_log = round(logreg.score(x_new, Y_train) * 100, 2)
acc_log
x_new=X_train[["Fare","Title","Embarked","Pclass","Age","IsAlone","Age*Class","Sex"]]
x_test_new=X_test[["Fare","Title","Embarked","Pclass","Age","IsAlone","Age*Class","Sex"]]
logreg = LogisticRegression()
logreg.fit(x_new, Y_train)
Y_pred = logreg.predict(x_test_new)
acc_log = round(logreg.score(x_new, Y_train) * 100, 2)
acc_log
# USe XG BOOST Classifier.
from xgboost import XGBClassifier, plot_importance
model = XGBClassifier()
model.fit(X_train, Y_train)

acc_log = round(model.score(X_train, Y_train) * 100, 2)
acc_log
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
test.head()
data = pd.concat([train, test])
train_size = train.shape[0] # 891
test_size = test.shape[0] # 418
data['Title'] = data['Name'].str.extract('([A-Za-z]+)\.', expand=False) #Extracting Title From oTHe 
pd.crosstab(data['Title'], data['Pclass'])
age_ref = data.groupby('Title').Age.mean()  #Calculating Average Age based on the P class
data = data.assign(
    Age = data.apply(lambda r: r.Age if pd.notnull(r.Age) else age_ref[r.Title] , axis=1)
)
del age_ref
data['AgeBand'] = pd.cut(data['Age'], 5, labels=range(5)).astype(int)
data[['AgeBand', 'Survived']].groupby(['AgeBand']).agg(['count','mean'])
data['Title'] = data['Title'].replace(['Don', 'Capt', 'Col', 'Major', 'Sir', 'Jonkheer', 'Rev', 'Dr'], 'Honored')
data['Title'] = data['Title'].replace(['Lady', 'Dona', 'Mme', 'Countess'], 'Mrs')
data['Title'] = data['Title'].replace(['Mlle', 'Ms'], 'Miss')
data[['Title', 'Survived']].groupby(['Title']).agg(['count','mean'])
data['Fare'] = data['Fare'].fillna(13.30)
data['FareBand'] = 0
data.loc[(data.Fare > 0) & (data.Fare <= 7.5), 'FareBand'] = 1
data.loc[(data.Fare > 7.5) & (data.Fare <= 12.5), 'FareBand'] = 2
data.loc[(data.Fare > 12.5) & (data.Fare <= 17), 'FareBand'] = 3
data.loc[(data.Fare > 17) & (data.Fare <= 29), 'FareBand'] = 4
data.loc[data.Fare > 29, 'FareBand'] = 5
data[['FareBand', 'Survived']].groupby(['FareBand']).agg(['count','mean'])
data['Embarked'] = data['Embarked'].fillna('S')
# deck may ensure the chance of escape
data['DeckCode'] = (data['Cabin']  # taken from other kernel
                        .str.slice(0,1)
                        .map({
                            'C':1, 
                            'E':2, 
                            'G':3,
                            'D':4, 
                            'A':5, 
                            'B':6, 
                            'F':7, 
                            #'T':8 #to rare
                        })
                        .fillna(0)
                        .astype(int))
data[['DeckCode', 'Survived']].groupby(['DeckCode']).agg(['count','mean'])
data['Room'] = (data['Cabin']
                    .str.slice(1,5).str.extract('([0-9]+)', expand=False)
                    .fillna(0)
                    .astype(int))

data['RoomBand'] = 0
data.loc[(data.Room > 0) & (data.Room <= 20), 'RoomBand'] = 1
data.loc[(data.Room > 20) & (data.Room <= 40), 'RoomBand'] = 2
data.loc[(data.Room > 40) & (data.Room <= 80), 'RoomBand'] = 3
data.loc[data.Room > 80, 'RoomBand'] = 4

data[['RoomBand', 'Survived']].groupby(['RoomBand']).agg(['count','mean'])
data.loc[data.Ticket=='LINE', 'Ticket'] = 'LINE1'
data['Odd'] = (data['Ticket']
                   .str.slice(-1) # last symbol
                   .astype(int)
                   .map(lambda x: x % 2 == 0)
                   .astype(int)
              )
data[['Odd', 'Survived']].groupby(['Odd']).agg(['count','mean'])
data['FamilySize'] = (data['SibSp'] + data['Parch']).astype(int)
data[['FamilySize', 'Survived']].groupby(['FamilySize']).agg(['count','mean'])
data[['Sex', 'Survived']].groupby(['Sex']).agg(['count','mean'])
data['FamilySizeBand'] = 0
data.loc[(data.FamilySize == 1), 'FamilySizeBand'] = 1
data.loc[(data.FamilySize == 2), 'FamilySizeBand'] = 2
data.loc[(data.FamilySize == 3), 'FamilySizeBand'] = 2
data.loc[data.FamilySize > 3, 'FamilySizeBand'] = 2
data[['FamilySizeBand', 'Survived']].groupby(['FamilySizeBand']).agg(['count','mean'])
data['IsAlone'] = (data['SibSp'] + data['Parch'] == 0).astype(int)
data[['IsAlone', 'Survived']].groupby(['IsAlone']).agg(['count','mean'])
data['SexCode'] = LabelEncoder().fit_transform(data['Sex'])
data['TitleCode'] = LabelEncoder().fit_transform(data['Title'])
data['EmbarkedCode'] = LabelEncoder().fit_transform(data['Embarked'])
data.shape
features = data[:train_size][[
    'Survived',
    'Pclass',
    'SexCode',
    'TitleCode',
    'FamilySize',
    'FamilySizeBand',
    'SibSp',
    'Parch',
    'IsAlone',
    'Age',
    'AgeBand',
    'Fare',
    'FareBand',    
    'EmbarkedCode',
    'DeckCode',
    'Room',
    'RoomBand',
    'Odd'
]]
plt.figure(figsize=(20,18))
sns.heatmap(features.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=plt.cm.RdBu, annot=True)
cols = [
    'Pclass',
    'Sex',
    'FamilySize',
    #'FamilySizeBand',
    'SibSp',
    'Parch',
    'IsAlone',
    #'Age',
    'AgeBand',
    'Fare',
    #'FareBand',
    'Title',
    #'Embarked',
    'DeckCode',
    #'Room',
    'RoomBand',
    #'Odd',
]
X_train = data[:train_size][cols]
Y_train = data[:train_size]['Survived'].astype(int)
X_test = data[train_size:][cols]

print(X_train.shape, Y_train.shape, X_test.shape)
X_train.head()
one_hot_features = [
    #'Pclass',
    'Sex',    
    #'FamilySizeBand',
    'AgeBand',
    #'FareBand',
    'Title',
    #'Embarked',
    'DeckCode',
    'RoomBand',
    #'Odd'
]
X_train = pd.get_dummies(X_train, columns = one_hot_features)
X_test = pd.get_dummies(X_test, columns = one_hot_features)

print(X_train.shape, Y_train.shape, X_test.shape)
#Taking Top 5 Features to test the model accuracy this is better than the previous model.
x_new=X_train[["Fare", 'FamilySize','Title_Honored','Title_Mr','AgeBand_1']]
logreg = LogisticRegression()
logreg.fit(x_new, Y_train)
acc_log = round(logreg.score(x_new, Y_train) * 100, 2)
acc_log
#Taking all Features to test the model accuracy this is better than the previous model.

x_new=X_train
logreg = LogisticRegression()
logreg.fit(x_new, Y_train)
acc_log = round(logreg.score(x_new, Y_train) * 100, 2)
acc_log
from xgboost import XGBClassifier, plot_importance
model = XGBClassifier()
model.fit(X_train, Y_train)
acc_log = round(model.score(x_new, Y_train) * 100, 2)
acc_log
