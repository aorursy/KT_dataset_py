import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df=pd.read_csv('../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv')
df.head()
df.info()
df=df.drop(['gameId', 'redFirstBlood', 'redKills', 'redEliteMonsters', 'redDragons','redTotalMinionsKilled',

       'redTotalJungleMinionsKilled', 'redGoldDiff', 'redExperienceDiff', 'redCSPerMin', 'redGoldPerMin', 'redHeralds',

       'blueGoldDiff', 'blueExperienceDiff', 'blueCSPerMin', 'blueGoldPerMin', 'blueTotalMinionsKilled'],axis=1)
df
plt.figure(figsize=(16, 12))

sns.heatmap(df.drop('blueWins', axis=1).corr(),cmap="mako", annot=True, fmt='.2f', vmin=0);
df=df.drop(['blueAvgLevel', 'redWardsPlaced', 'redWardsDestroyed', 'redDeaths', 'redAssists', 'redTowersDestroyed',

       'redTotalExperience', 'redTotalGold', 'redAvgLevel'],axis=1)
df
corr=df.corr()

plt.figure(figsize=(16, 12))

sns.heatmap(corr,cmap="mako", annot=True, fmt='.2f', vmin=0)
corr_list = df[df.columns[1:]].apply(lambda x: x.corr(df['blueWins']))

cols = []

for col in corr_list.index:

    if (corr_list[col]>0.2 or corr_list[col]<-0.2):

        cols.append(col)

cols
df_final=df[cols]
X=df_final.values

y=df['blueWins'].values
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split



scaler = MinMaxScaler()

scaler.fit(X)

X = scaler.transform(X)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(random_state=0)

lr.fit(X_train,y_train)

print(lr.score(X_test,y_test))
import keras

from keras.models import *

from keras.layers import *

from keras import *
X_train.shape
model=Sequential()



model.add(Dense(64,activation='relu',input_shape=(8,)))

# model.add(Dense(64,activation='relu'))

# model.add(Dense(32,activation='relu'))

# model.add(Dense(32,activation='relu'))

model.add(Dense(16,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer='adam')

model.summary()
hist=model.fit(X_train,y_train,

    batch_size=128,

    epochs=300,

    verbose=1,

    validation_split=0.2,

    shuffle=True

)
res=hist.history



plt.plot(res['accuracy'],label="accuracy")

plt.plot(res['val_accuracy'],label="val acc")

plt.plot(res['loss'],label='loss')

plt.plot(res['val_loss'],label='val loss')

plt.legend()

plt.show()
model.evaluate(X_test,y_test)