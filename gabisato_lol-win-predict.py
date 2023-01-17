import numpy as np

import pandas as pd 

import os



dataset = pd.read_csv('../input/dataset.csv')

dataset.head()
dataset.drop(['gameId', 'queueId', 'seasonId', 'firstBloodAssistPartId'], axis=1, inplace=True)

for t in range(1,3):

    col1 = 'vilemawKillsTeam' + str(t)

    col2 = 'dominionVictoryScoreTeam' + str(t)

    dataset.drop([col1, col2], axis=1, inplace=True)

    for p in range(1, 6):

        v = 'T' + str(t) + '_P' + str(p)

        c1 = v + '_participantId'

        c2 = v + '_pentaKills'

        c3 = v + '_unrealKills'

        c4 = v + '_sightWardsBoughtInGame'

        #print(c1, c2, c3, c4)

        dataset.drop([c1, c2, c3, c4], axis=1, inplace=True)

print(dataset.columns)
dataset.describe(include = ('O'))

#20 dados categóricos
dataset.isnull().sum().sort_values(ascending = False)
import seaborn as sns

import matplotlib.pyplot as plt
num = dataset

for t in range(1,3):

    for p in range(1,6):

        col1 = "T" + str(t) + "_P" + str(p) + "_highestAchievedSeasonTier"

        col2 = "T" + str(t) + "_P" + str(p) + "_lane/role"

        num.drop(col1, axis= 1, inplace = True)

        num.drop(col2, axis= 1, inplace = True)

        

print(num.columns)
f, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

sns.heatmap(

    num.iloc[:,0:8].corr(),

    annot=True,

    ax=axes[0]

)

position = 0

for x in range (9,28, 7):

    position = position + 1

    size = []

    size.append(1)

    for i in range(x, x+7):

        size.append(i)

    sns.heatmap(

    num.iloc[:, size].corr(),

    annot=True,

    ax=axes[position]

    )
f, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

position = 0

for x in range (30, 56, 7):

    size = []

    size.append(1)

    for i in range(x, x+7):

        size.append(i)

    sns.heatmap(

    num.iloc[:, size].corr(),

    annot=True,

    ax=axes[position]

    )

    position = position + 1
f, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

position = 0

for x in range (57, 84, 7):

    size = []

    size.append(1)

    for i in range(x, x+7):

        size.append(i)

    sns.heatmap(

    num.iloc[:, size].corr(),

    annot=True,

    ax=axes[position]

    )

    position = position + 1
f, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

position = 0

for x in range (85, 112, 7):

    size = []

    size.append(1)

    for i in range(x, x+7):

        size.append(i)

    sns.heatmap(

    num.iloc[:, size].corr(),

    annot=True,

    ax=axes[position]

    )

    position = position + 1
f, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

position = 0

for x in range (113, 140, 7):

    size = []

    size.append(1)

    for i in range(x, x+7):

        size.append(i)

    sns.heatmap(

    num.iloc[:, size].corr(),

    annot=True,

    ax=axes[position]

    )

    position = position + 1
f, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

position = 0

for x in range (141, 168, 7):

    size = []

    size.append(1)

    for i in range(x, x+7):

        size.append(i)

    sns.heatmap(

    num.iloc[:, size].corr(),

    annot=True,

    ax=axes[position]

    )

    position = position + 1
f, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

position = 0

for x in range (169, 196, 7):

    size = []

    size.append(1)

    for i in range(x, x+7):

        size.append(i)

    sns.heatmap(

    num.iloc[:, size].corr(),

    annot=True,

    ax=axes[position]

    )

    position = position + 1
f, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

position = 0

for x in range (197, 224, 7):

    size = []

    size.append(1)

    for i in range(x, x+7):

        size.append(i)

    sns.heatmap(

    num.iloc[:, size].corr(),

    annot=True,

    ax=axes[position]

    )

    position = position + 1
f, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

position = 0

for x in range (225, 252, 7):

    size = []

    size.append(1)

    for i in range(x, x+7):

        size.append(i)

    sns.heatmap(

    num.iloc[:, size].corr(),

    annot=True,

    ax=axes[position]

    )

    position = position + 1
f, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

position = 0

for x in range (253, 280, 7):

    size = []

    size.append(1)

    for i in range(x, x+7):

        size.append(i)

    sns.heatmap(

    num.iloc[:, size].corr(),

    annot=True,

    ax=axes[position]

    )

    position = position + 1
f, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

position = 0

for x in range (281, 308, 7):

    size = []

    size.append(1)

    for i in range(x, x+7):

        size.append(i)

    sns.heatmap(

    num.iloc[:, size].corr(),

    annot=True,

    ax=axes[position]

    )

    position = position + 1
f, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

position = 0

for x in range (309, 336, 7):

    size = []

    size.append(1)

    for i in range(x, x+7):

        size.append(i)

    sns.heatmap(

    num.iloc[:, size].corr(),

    annot=True,

    ax=axes[position]

    )

    position = position + 1
f, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

position = 0

for x in range (337, 364, 7):

    size = []

    size.append(1)

    for i in range(x, x+7):

        size.append(i)

    sns.heatmap(

    num.iloc[:, size].corr(),

    annot=True,

    ax=axes[position]

    )

    position = position + 1
f, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

position = 0

for x in range (365, 392, 7):

    size = []

    size.append(1)

    for i in range(x, x+7):

        size.append(i)

    sns.heatmap(

    num.iloc[:, size].corr(),

    annot=True,

    ax=axes[position]

    )

    position = position + 1
f, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

position = 0

for x in range (393, 420, 7):

    size = []

    size.append(1)

    for i in range(x, x+7):

        size.append(i)

    sns.heatmap(

    num.iloc[:, size].corr(),

    annot=True,

    ax=axes[position]

    )

    position = position + 1
f, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

position = 0

for x in range (421, 448, 7):

    size = []

    size.append(1)

    for i in range(x, x+7):

        size.append(i)

    sns.heatmap(

    num.iloc[:, size].corr(),

    annot=True,

    ax=axes[position]

    )

    position = position + 1
f, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

position = 0

for x in range (449, 476, 7):

    size = []

    size.append(1)

    for i in range(x, x+7):

        size.append(i)

    sns.heatmap(

    num.iloc[:, size].corr(),

    annot=True,

    ax=axes[position]

    )

    position = position + 1
sns.heatmap(

    num.iloc[:, [1, 477, 478, 479, 480, 481, 482]].corr(),

    annot=True

)
for name in num.iloc[:, 0:50].columns:

    sns.FacetGrid(num, col='win').map(plt.hist, name, bins=20)
for name in num.iloc[:, 51:100].columns:

    sns.FacetGrid(num, col='win').map(plt.hist, name, bins=20)
for name in num.iloc[:, 101:150].columns:

    sns.FacetGrid(num, col='win').map(plt.hist, name, bins=20)
for name in num.iloc[:, 152:200].columns:

    sns.FacetGrid(num, col='win').map(plt.hist, name, bins=20)
for name in num.iloc[:, 201:250].columns:

    sns.FacetGrid(num, col='win').map(plt.hist, name, bins=20)
for name in num.iloc[:, 251:300].columns:

    sns.FacetGrid(num, col='win').map(plt.hist, name, bins=20)
for name in num.iloc[:, 301:350].columns:

    sns.FacetGrid(num, col='win').map(plt.hist, name, bins=20)
for name in num.iloc[:, 351:400].columns:

    sns.FacetGrid(num, col='win').map(plt.hist, name, bins=20)
for name in num.iloc[:, 401:450].columns:

    sns.FacetGrid(num, col='win').map(plt.hist, name, bins=20)
for name in num.iloc[:, 451:483].columns:

    sns.FacetGrid(num, col='win').map(plt.hist, name, bins=20)
cols = []

cols.append('win')

cols.append('firstBlood')

cols.append('firstTower')

cols.append('firstInhibitor')

cols.append('firstBaron')

cols.append('firstDragon')

cols.append('firstRiftHerald')

for t in range(1, 3):

    cols.append('towerKillsTeam' + str(t))

    cols.append('inhibitorKillsTeam' + str(t))

    cols.append('baronKillsTeam' + str(t))

    cols.append('riftHeraldKillsTeam' + str(t))

    for p in range(1, 6):

        part = 'T' + str(t) + '_P' + str(p) + '_'

        cols.append(part + 'kills')

        cols.append(part + 'deaths')

        cols.append(part + 'assists')

        cols.append(part + 'longestTimeSpentLiving')

        cols.append(part + 'totalDamageDealtToChampions')

        cols.append(part + 'totalHeal')

        cols.append(part + 'damageSelfMitigated')

        cols.append(part + 'damageDealtToObjectives')

        cols.append(part + 'damageDealtToTurrets')

        cols.append(part + 'visionScore')

        cols.append(part + 'timeCCingOthers')

        cols.append(part + 'totalDamageTaken')

        cols.append(part + 'goldEarned')

        cols.append(part + 'turretKills')

        cols.append(part + 'inhibitorKills')

        cols.append(part + 'neutralMinionsKilled')

        cols.append(part + 'neutralMinionsKilledEnemyJungle')

        cols.append(part + 'totalTimeCrowdControlDealt')

        cols.append(part + 'wardsPlaced')

        cols.append(part + 'wardsKilled')
train = dataset.loc[:, cols]

print(train.columns)
# x - atributos

# y - classe

x = train.drop('win', axis=1)

y = train['win']
print(x.shape)

print(y.shape)


y = y.replace(1, 0)

y = y.replace(2, 1)

print(y)
plt.scatter(y.iloc[:],

            y.iloc[:],

            c=list(np.array(y).ravel()),

            s=15,

            cmap='bwr')
from sklearn.model_selection import train_test_split



# divide o dataset de trainamento em treinamento e teste, separando 25% em teste

# x = atributos e y = classes

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
from sklearn.tree import DecisionTreeClassifier

# cria a arvore

tree = DecisionTreeClassifier(criterion='gini', splitter='random', max_depth=3, random_state=0)

#cria o modelo

model = tree.fit(x_train, y_train)
from sklearn.metrics import accuracy_score

# tenta predicao dos dados de teste 

predict = model.predict(x_test)
acc = accuracy_score(y_test, predict)

print("A probabilidade de acerto é: ", format(acc))
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout



model = Sequential()

#primeira camada

# 214+1 / 2 = 107

model.add(Dense(units=108, init='uniform', activation='relu', input_dim=214))

model.add(Dropout(p=0.1))

model.add(Dense(units=108, init='uniform', activation='relu'))

model.add(Dropout(p=0.1))

model.add(Dense(units=108, init='uniform', activation='relu'))

model.add(Dropout(p=0.1))

model.add(Dense(units=1, init='uniform', activation='sigmoid'))



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



model.fit(x_train, y_train, batch_size=100, epochs=400, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

print('Erro:', score[0])

print('Precisão:', score[1]*100, "%" )