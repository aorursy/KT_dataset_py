# Import data analysis libraries

import pandas as pd

import numpy as np



# Import libraries for visualisation

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline



# Show all columns

pd.set_option('display.max_columns', None)

pd.set_option('mode.chained_assignment', None)



print('Libraries Imported!')
# Read into dataframe

df = pd.read_csv('../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv')

df.head()
# Get column names

cols = df.columns

print(cols)
# Seperate target variable from dataframe

y = df.blueWins



# Drop target and unnecessary features

drop_cols = ['gameId','blueWins']

x = df.drop(drop_cols, axis=1)



x.head()
# Visualise blueWins using countplot

ax = sns.countplot(y, label='Count', palette='RdBu')

W, L = y.value_counts()



print('Red Wins: {} ({}%), Blue Wins: {}({}%)'.format(W,round(100*W/(W+L),3),L,round(100*L/(W+L),3)))
x.describe()
# Drop unnecessary features (same as blueFirstBlood, blueDeaths etc.)

drop_cols = ['redFirstBlood','redKills','redDeaths'

             ,'redGoldDiff','redExperienceDiff', 'blueCSPerMin',

            'blueGoldPerMin','redCSPerMin','redGoldPerMin']

x.drop(drop_cols, axis=1, inplace=True)

x.head()
# Copy feature matrix and standardise

data = x

data_std = (data - data.mean()) / data.std()

data = pd.concat([y, data_std.iloc[:, 0:9]], axis=1)

data = pd.melt(data, id_vars='blueWins', var_name='Features', value_name='Values')



fig, ax = plt.subplots(1,2,figsize=(15,5))



# Create violin plot of features

#plt.figure(figsize=(8,5))

sns.violinplot(x='Features', y='Values', hue='blueWins', data=data, split=True,

               inner='quart', ax=ax[0], palette='Blues')

fig.autofmt_xdate(rotation=45)



data = x

data_std = (data - data.mean()) / data.std()

data = pd.concat([y, data_std.iloc[:, 9:18]], axis=1)

data = pd.melt(data, id_vars='blueWins', var_name='Features', value_name='Values')



# Create violin plot

#plt.figure(figsize=(8,5))

sns.violinplot(x='Features', y='Values', hue='blueWins', 

               data=data, split=True, inner='quart', ax=ax[1], palette='Blues')

fig.autofmt_xdate(rotation=45)



plt.show()
plt.figure(figsize=(18,14))

sns.heatmap(round(x.corr(),2), cmap='Blues', annot=True)

plt.show()
# Drop unnecessary features

drop_cols = ['redAvgLevel','blueAvgLevel']

x.drop(drop_cols, axis=1, inplace=True)
sns.set(style='whitegrid', palette='muted')



x['wardsPlacedDiff'] = x['blueWardsPlaced'] - x['redWardsPlaced']

x['wardsDestroyedDiff'] = x['blueWardsDestroyed'] - x['redWardsDestroyed']



data = x[['blueWardsPlaced','blueWardsDestroyed','wardsPlacedDiff','wardsDestroyedDiff']].sample(1000)

data_std = (data - data.mean()) / data.std()

data = pd.concat([y, data_std], axis=1)

data = pd.melt(data, id_vars='blueWins', var_name='Features', value_name='Values')



plt.figure(figsize=(10,6))

sns.swarmplot(x='Features', y='Values', hue='blueWins', data=data)

plt.xticks(rotation=45)

plt.show()
# Drop unnecessary features

drop_cols = ['blueWardsPlaced','blueWardsDestroyed','wardsPlacedDiff',

            'wardsDestroyedDiff','redWardsPlaced','redWardsDestroyed']

x.drop(drop_cols, axis=1, inplace=True)
x['killsDiff'] = x['blueKills'] - x['blueDeaths']

x['assistsDiff'] = x['blueAssists'] - x['redAssists']



x[['blueKills','blueDeaths','blueAssists','killsDiff','assistsDiff','redAssists']].hist(figsize=(12,10), bins=20)

plt.show()
sns.set(style='whitegrid', palette='muted')



data = x[['blueKills','blueDeaths','blueAssists','killsDiff','assistsDiff','redAssists']].sample(1000)

data_std = (data - data.mean()) / data.std()

data = pd.concat([y, data_std], axis=1)

data = pd.melt(data, id_vars='blueWins', var_name='Features', value_name='Values')



plt.figure(figsize=(10,6))

sns.swarmplot(x='Features', y='Values', hue='blueWins', data=data)

plt.xticks(rotation=45)

plt.show()
data = pd.concat([y, x], axis=1).sample(500)



sns.pairplot(data, vars=['blueKills','blueDeaths','blueAssists','killsDiff','assistsDiff','redAssists'], 

             hue='blueWins')



plt.show()
data = pd.concat([y, x], axis=1)



fig, ax = plt.subplots(1,2, figsize=(15,6))

sns.scatterplot(x='killsDiff', y='assistsDiff', hue='blueWins', data=data, ax=ax[0])



sns.scatterplot(x='blueKills', y='blueAssists', hue='blueWins', data=data, ax=ax[1])

plt.show()
# Drop unnecessary features

drop_cols = ['blueFirstBlood','blueKills','blueDeaths','blueAssists','redAssists']

x.drop(drop_cols, axis=1, inplace=True)
x['dragonsDiff'] = x['blueDragons'] - x['redDragons']

x['heraldsDiff'] = x['blueHeralds'] - x['redHeralds']

x['eliteDiff'] = x['blueEliteMonsters'] - x['redEliteMonsters']



data = pd.concat([y, x], axis=1)



eliteGroup = data.groupby(['eliteDiff'])['blueWins'].mean()

dragonGroup = data.groupby(['dragonsDiff'])['blueWins'].mean()

heraldGroup = data.groupby(['heraldsDiff'])['blueWins'].mean()



fig, ax = plt.subplots(1,3, figsize=(15,4))



eliteGroup.plot(kind='bar', ax=ax[0])

dragonGroup.plot(kind='bar', ax=ax[1])

heraldGroup.plot(kind='bar', ax=ax[2])



print(eliteGroup)

print(dragonGroup)

print(heraldGroup)



plt.show()
# Drop unnecessary features

drop_cols = ['blueEliteMonsters','blueDragons','blueHeralds',

            'redEliteMonsters','redDragons','redHeralds']

x.drop(drop_cols, axis=1, inplace=True)
x['towerDiff'] = x['blueTowersDestroyed'] - x['redTowersDestroyed']



data = pd.concat([y, x], axis=1)



towerGroup = data.groupby(['towerDiff'])['blueWins']

print(towerGroup.count())

print(towerGroup.mean())



fig, ax = plt.subplots(1,2,figsize=(15,5))



towerGroup.mean().plot(kind='line', ax=ax[0])

ax[0].set_title('Proportion of Blue Wins')

ax[0].set_ylabel('Proportion')



towerGroup.count().plot(kind='line', ax=ax[1])

ax[1].set_title('Count of Towers Destroyed')

ax[1].set_ylabel('Count')
# Drop unnecessary features

drop_cols = ['blueTowersDestroyed','redTowersDestroyed']

x.drop(drop_cols, axis=1, inplace=True)
data = pd.concat([y, x], axis=1)



data[['blueGoldDiff','blueExperienceDiff']].hist(figsize=(15,5))

plt.show()
plt.figure(figsize=(8,6))

sns.scatterplot(x='blueExperienceDiff', y='blueGoldDiff', hue='blueWins', data=data)
# Drop unnecessary features

drop_cols = ['blueTotalGold','blueTotalExperience','redTotalGold','redTotalExperience']

x.drop(drop_cols, axis=1, inplace=True)



x.rename(columns={'blueGoldDiff':'goldDiff', 'blueExperienceDiff':'expDiff'}, inplace=True)
data = pd.concat([y, x], axis=1)



data[['blueTotalMinionsKilled','blueTotalJungleMinionsKilled',

      'redTotalMinionsKilled','redTotalJungleMinionsKilled']].hist(figsize=(15,10))

plt.show()
sns.set(style='whitegrid', palette='muted')



data = x[['blueTotalMinionsKilled','blueTotalJungleMinionsKilled',

      'redTotalMinionsKilled','redTotalJungleMinionsKilled']].sample(1000)

data_std = (data - data.mean()) / data.std()

data = pd.concat([y, data_std], axis=1)

data = pd.melt(data, id_vars='blueWins', var_name='Features', value_name='Values')



plt.figure(figsize=(10,6))

sns.swarmplot(x='Features', y='Values', hue='blueWins', data=data)

plt.xticks(rotation=45)

plt.show()
# Drop unnecessary features

drop_cols = ['blueTotalMinionsKilled','blueTotalJungleMinionsKilled',

      'redTotalMinionsKilled','redTotalJungleMinionsKilled']

x.drop(drop_cols, axis=1, inplace=True)
# Import libraries for machine learning models

from sklearn import preprocessing, metrics

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB



print('Machine Learning Libraries Imported!')
print(x.shape,y.shape)

x.head()
X = preprocessing.StandardScaler().fit(x).transform(x.astype(float))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)
from prettytable import PrettyTable

table = PrettyTable()

table.field_names = ['Algorithm', 'Accuracy', 'Recall', 'Precision', 'F-Score']
def get_confusion_matrix(algorithm, y_pred, y_actual):

    # Create confusion matrix and interpret values

    con = confusion_matrix(y_test, y_pred)

    tp, fn, fp, tn = con[0][0], con[0][1], con[1][0], con[1][1]

    algorithm = algorithm

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    recall = tp / (tp + fn)

    precision = tp / (tp + fp)

    f_score = (2 * precision * recall) / (recall + precision)

    return algorithm, accuracy, recall, precision, f_score
# Test different values of k

Ks = 10

mean_acc = np.zeros((Ks-1))

for n in range(1,Ks):

    kneigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)

    y_pred = kneigh.predict(X_test)

    mean_acc[n-1] = metrics.accuracy_score(y_test, y_pred)



# Use most accurate k value to predict test values

k = mean_acc.argmax()+1

neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)

y_pred = neigh.predict(X_test)
# Call confusion matrix and accuracy

algorithm, accuracy, recall, precision, f_score = get_confusion_matrix('KNN', y_pred, y_test)



# Add values to table

table.add_row([algorithm, round(accuracy,5), round(recall,5),

               round(precision,5), round(f_score,5)])
# Initialise Decision Tree classifier and predict

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)

drugTree.fit(X_train,y_train)

y_pred = drugTree.predict(X_test)
# Call confusion matrix and accuracy

algorithm, accuracy, recall, precision, f_score = get_confusion_matrix('Decision', y_pred, y_test)



# Add values to table

table.add_row([algorithm, round(accuracy,5), round(recall,5),

               round(precision,5), round(f_score,5)])
# Train and predict logistic regression model

LR = LogisticRegression(C=0.01, solver='liblinear')

y_pred = LR.fit(X_train,y_train).predict(X_test)
# Call confusion matrix and accuracy

algorithm, accuracy, recall, precision, f_score = get_confusion_matrix('LR', y_pred, y_test)



# Add values to table

table.add_row([algorithm, round(accuracy,5), round(recall,5),

               round(precision,5), round(f_score,5)])
clf = svm.SVC(kernel='rbf')

y_pred = clf.fit(X_train, y_train).predict(X_test)
# Call confusion matrix and accuracy

algorithm, accuracy, recall, precision, f_score = get_confusion_matrix('SVM', y_pred, y_test)



# Add values to table

table.add_row([algorithm, round(accuracy,5), round(recall,5),

               round(precision,5), round(f_score,5)])
gnb = GaussianNB()

y_pred = gnb.fit(X_train, y_train).predict(X_test)
# Call confusion matrix and accuracy

algorithm, accuracy, recall, precision, f_score = get_confusion_matrix('Bayes', y_pred, y_test)



# Add values to table

table.add_row([algorithm, round(accuracy,5), round(recall,5),

               round(precision,5), round(f_score,5)])
# Instantiate Random Forest Classifier and predict values

clf = RandomForestClassifier(max_depth=2, random_state=0)

y_pred = clf.fit(X_train, y_train).predict(X_test)
# Call confusion matrix and accuracy

algorithm, accuracy, recall, precision, f_score = get_confusion_matrix('R Forest', y_pred, y_test)



# Add values to table

table.add_row([algorithm, round(accuracy,5), round(recall,5),

               round(precision,5), round(f_score,5)])
print(table)