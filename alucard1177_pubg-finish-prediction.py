import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

train_data = pd.read_csv("../input/train_V2.csv")
test_data = pd.read_csv("../input/test_V2.csv")
train_data.head(10)
train_data.describe()
#Посмотрим, есть ли незаполненные значения в данных
train_data.isna().sum()
test_data.isna().sum()
#Посмотрим на строку, где не заполнено target значение
train_data[train_data['winPlacePerc'].isnull()]
#Удалим ее
train_data.drop(2744604, inplace=True)
train_data['playersJoined'] = train_data.groupby('matchId')['matchId'].transform('count')
train_data['killsNorm'] = train_data['kills']*((100-train_data['playersJoined'])/100 + 1)
train_data['damageDealtNorm'] = train_data['damageDealt']*((100-train_data['playersJoined'])/100 + 1)
train_data['maxPlaceNorm'] = train_data['maxPlace']*((100-train_data['playersJoined'])/100 + 1)
train_data['matchDurationNorm'] = train_data['matchDuration']*((100-train_data['playersJoined'])/100 + 1)

test_data['playersJoined'] = test_data.groupby('matchId')['matchId'].transform('count')
test_data['killsNorm'] = test_data['kills']*((100-test_data['playersJoined'])/100 + 1)
test_data['damageDealtNorm'] = test_data['damageDealt']*((100-test_data['playersJoined'])/100 + 1)
test_data['maxPlaceNorm'] = test_data['maxPlace']*((100-test_data['playersJoined'])/100 + 1)
test_data['matchDurationNorm'] = test_data['matchDuration']*((100-test_data['playersJoined'])/100 + 1)
to_show = ['Id', 'kills','killsNorm','damageDealt', 'damageDealtNorm', 'maxPlace', 'maxPlaceNorm', 'matchDuration', 'matchDurationNorm']
test_data[to_show][0:11]
#Всего падает 100 человек на большой карте, поэтому сомнительно, что один человек
#убьет более 25 человек за матч
train_data[train_data['kills'] > 25].shape
train_data.drop(train_data[train_data['kills'] > 25].index, inplace=True)
plt.figure(figsize=(12,4))
sns.distplot(train_data['walkDistance'], bins=10)
plt.show()
train_data[train_data['walkDistance'] >= 10000].shape
train_data.drop(train_data[train_data['walkDistance'] > 10000].index, inplace=True)
plt.figure(figsize=(12,4))
sns.distplot(train_data['rideDistance'], bins=10)
plt.show()
train_data[train_data['rideDistance'] >= 20000].shape
train_data.drop(train_data[train_data['rideDistance'] >= 20000].index, inplace=True)
train_data[train_data['weaponsAcquired'] >= 60].shape
train_data.drop(train_data[train_data['weaponsAcquired'] >= 60].index, inplace=True)
plt.figure(figsize=(12,4))
sns.distplot(train_data['heals'], bins=10)
plt.show()
train_data[train_data['heals'] >= 40].shape
train_data.drop(train_data[train_data['heals'] >= 40].index, inplace=True)
train_data[train_data['roadKills'] > 10].shape
train_data.drop(train_data[train_data['roadKills'] >= 10].index, inplace=True)
train_data.drop(columns=['kills', 'damageDealt', 'maxPlace', 'matchDuration'], inplace=True)
test_data.drop(columns=['kills', 'damageDealt', 'maxPlace', 'matchDuration'], inplace=True)
#Посмотрим на корреляцию признаков с целевой переменной
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(train_data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
#Закодируем тип матча
from sklearn.preprocessing import LabelEncoder
def encode_features(data, features):
    for feature in features:
        le = LabelEncoder()
        le.fit(data[feature])
        encoded_column = le.transform(data[feature])
        data[feature] = encoded_column
    return data

to_encode = ['matchType']
train_data = encode_features(train_data, to_encode)
test_data = encode_features(test_data, to_encode)
train_data.head()
#Бинарно закодируем тип матча
match_binaries = pd.get_dummies(train_data['matchType'], prefix='matchType_')
train_data = pd.concat([train_data, match_binaries], axis=1)

match_binaries = pd.get_dummies(test_data['matchType'], prefix='matchType_')
test_data = pd.concat([test_data, match_binaries], axis=1)

test_data.head(10)
X_train = train_data.drop(columns=['Id', 'groupId', 'matchId', 'matchType'])
y_train = train_data['winPlacePerc']
X = X_train.sample(500000)

y = X['winPlacePerc']
X = X.drop(columns=['winPlacePerc'])

X_test = test_data.drop(columns=['Id', 'groupId', 'matchId', 'matchType'])
from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.33, random_state=177)
#Обучим линейную регрессию
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as mea
linreg = LinearRegression()
linreg.fit(X_tr, y_tr)
mea(linreg.predict(X_val), y_val)
#Обучим случайный лес
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=10)
forest.fit(X_tr, y_tr)
mea(forest.predict(X_val), y_val)
X_train.columns
X_test.columns
#Случайный лес справился лучше, но линейная регрессия - быстрее
forest.fit(X_train.drop(columns=['winPlacePerc']), y_train)
prediction = forest.predict(X_test)
submission = pd.DataFrame({
        "Id": test_data["Id"],
        "winPlacePerc": prediction 
    })
submission.to_csv("submission.csv", index=False)
submission.head()