import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os
data = pd.read_csv('../input/world-happiness-report-2019.csv')
data.head(10)
data = data.drop(['Ladder', 'SD of Ladder'], axis=1)



data = data.rename(columns={

    'Country (region)':'Country',

    'Positive affect':'Pos',

    'Negative affect':'Neg',

    'Log of GDP\nper capita':'GDP',

    'Healthy life\nexpectancy':'Life expectancy'

})
data.shape
data.info()
data.isnull().sum()
data = data[~data.isnull().any(axis=1)]

data.shape
data.describe()
fig, ax = plt.subplots(2, 4, figsize=(16, 8))

plt.tight_layout()



for i, feature in enumerate(list(data)[1:]):

    sns.boxplot(x=feature, data=data, orient='v', ax=ax[int(i-4>=0)][i%4]);
plt.figure(figsize=(8,8))

sns.heatmap(data.corr(), cmap="Blues");
data.corr()
size = int(data.shape[0]/3)

size
data['Class'] = 0

data.iloc[size:2*size]['Class'] = 1

data.iloc[2*size:]['Class'] = 2
# Happiest countries among each group



data.iloc[[0, size, 2*size], :]
def distplot(col, bins=10):

    

    fig, ax = plt.subplots(1,3,figsize=(16, 4))



    sns.distplot(data[data['Class']==0][col], bins=10, ax=ax[0])

    ax[0].set_title('Class 0')



    sns.distplot(data[data['Class']==1][col], bins=10, ax=ax[1])

    ax[1].set_title('Class 1')



    sns.distplot(data[data['Class']==2][col], bins=10, ax=ax[2])

    ax[2].set_title('Class 2')



    plt.show();
distplot('Freedom')
distplot('Corruption')
distplot('Social support')
distplot('GDP')
distplot('Life expectancy')
def scatterplot(x, y):

    

    fig, ax = plt.subplots(1, 3, figsize=(16, 6))



    sns.regplot(x, y, data=data[data['Class']==0], ax=ax[0])

    ax[0].set_title('Class 0', size=15)

    ax[0].set_xlabel(x, size=15)

    ax[0].set_ylabel(y, size=15)

    

    sns.regplot(x, y, data=data[data['Class']==1], ax=ax[1])

    ax[1].set_title('Class 1')

    ax[1].set_title('Class 1', size=15)

    ax[1].set_xlabel(x, size=15)

    ax[1].set_ylabel(y, size=15)

    

    sns.regplot(x, y, data=data[data['Class']==2], ax=ax[2])

    ax[2].set_title('Class 2')

    ax[2].set_title('Class 2', size=15)

    ax[2].set_xlabel(x, size=15)

    ax[2].set_ylabel(y, size=15)

    

    plt.show();
x = 'Social support'

y = 'GDP'

z = 'Life expectancy'
scatterplot(x, y)
scatterplot(y, z)
scatterplot(x, z)
fig = sns.pairplot(data=data[['GDP', 'Social support', 'Life expectancy']])



fig.fig.set_size_inches(12, 12);
from sklearn.tree import DecisionTreeClassifier
# Reset index since some samples were dropped before that a few numbers skip

data.index = np.arange(data.shape[0])
# Randomly choose testing samples

happy_idx = np.random.choice(np.arange(size), size=5, replace=False)

neutral_idx = np.random.choice(np.arange(size, 2*size), size=5, replace=False)

sad_idx = np.random.choice(np.arange(2*size, data.shape[0]), size=5, replace=False)



test_idx = list(happy_idx) + list(neutral_idx) + list(sad_idx)
test = data.iloc[test_idx]

test
train = data[~data.index.isin(test_idx)]



train.shape, test.shape
def split_data(dat):

    

    X = dat.loc[:, ['Social support', 'GDP', 'Life expectancy']]

    y = dat.loc[:, 'Class']

    

    return X, y
# Only use three features

X_train, y_train = split_data(train)

X_test, y_test = split_data(test)
# Set random_state for reproducibility

clf = DecisionTreeClassifier(random_state=123)



clf.fit(X_train, y_train)

clf.score(X_test, y_test)
X_train, y_train = train.drop(['Class', 'Country'], axis=1), train.loc[:, 'Class']

X_test, y_test = test.drop(['Class', 'Country'], axis=1), test.loc[:, 'Class']
clf2 = DecisionTreeClassifier(random_state=123)



clf2.fit(X_train, y_train)

clf2.score(X_test, y_test)
feature_importances = np.stack((clf2.feature_importances_, list(X_train)), axis=1)

feature_importances = feature_importances[feature_importances.argsort(axis=0)[:, 0]][::-1]

feature_importances
scores = []



for i in range(1, len(feature_importances)+1):

    

    features = feature_importances[:i, 1]



    clf = DecisionTreeClassifier(random_state=123)

    

    clf.fit(X_train.loc[:, features], y_train)

    

    scores.append(clf.score(X_test.loc[:, features], y_test))
plt.figure(figsize=(8, 6))

plt.plot(scores)

plt.xlabel('Numer of Features', size=15)

plt.ylabel('Scores', size=15)

plt.show();
feature_importances