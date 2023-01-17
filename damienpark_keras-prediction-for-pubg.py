import numpy as np

import pandas as pd

import tqdm



import matplotlib.pyplot as plt



import keras

from keras.layers.core import Dense

from keras.layers.normalization import BatchNormalization



from sklearn.preprocessing import MinMaxScaler, RobustScaler, MaxAbsScaler
train = pd.read_csv("../input/train_V2.csv")
train.head()
train.describe().T
pd.DataFrame(train.dtypes, columns=["Type"])
plt.figure(figsize=(25, 25))

for idx, v in enumerate(train.columns[train.dtypes != "O"]):

    plt.subplot(5, 5, idx+1)

    plt.hist(train[v].dropna(), bins = 50)

    plt.title(v)



plt.show()
# plt.figure(figsize=(25, 25))

# for idx, v in enumerate(train.columns[train.dtypes != "O"]):

#     plt.subplot(5, 5, idx+1)

#     plt.scatter(train["winPlacePerc"], train[v], alpha=0.5)

#     plt.title(v)



# plt.show()
# plt.figure(figsize=(25, 25))

# for idx, v in enumerate(train.columns[train.dtypes == "float64"][:5]):

#     plt.subplot(2, 3, idx+1)

#     for i in pd.unique(train.matchType):

#         plt.scatter(train.loc[train.matchType == i, "winPlacePerc"], train.loc[train.matchType == i, v], alpha = 0.5, label = i)

#     plt.legend()

#     plt.title(v)



# plt.show()
# plt.figure(figsize=(25, 25))

# for idx, v in enumerate(train.columns[train.dtypes == "int"]):

#     plt.subplot(5, 4, idx+1)

#     for i in pd.unique(train.matchType):

#         plt.scatter(train.loc[train.matchType == i, "winPlacePerc"], train.loc[train.matchType == i, v], alpha = 0.5, label = i)

#     plt.legend()

#     plt.title(v)



# plt.show()
print(list(train.columns[train.dtypes == "O"]))
print("Number of record:", len(train), "\nNumber of Unique Id:", len(pd.unique(train.Id)))
print("Number of match: ", len(pd.unique(train.matchId)), "\nNumber of match(<9): ", sum(train.groupby("matchId").size() < 9))
temp = train.loc[train.matchId.isin(train.groupby("matchId").size()[train.groupby("matchId").size() < 9].index), :]

temp.loc[temp.matchId == "e263f4a227313a"]
temp = pd.DataFrame(train.groupby("matchId").size(), columns=["player"])

temp.reset_index(level=0, inplace=True)
train = train.merge(temp, left_on="matchId", right_on="matchId")
print("Type: ", pd.unique(train.matchType), "\nCount: ", len(pd.unique(train.matchType)))
#게임인원별 분류(Division by number of player in group)

train["matchType_1"] = "-"

train.loc[(train.matchType == "solo-fpp") | 

          (train.matchType == "solo") | 

          (train.matchType == "normal-solo-fpp") | 

          (train.matchType == "normal-solo"), "matchType_1"] = "solo"



train.loc[(train.matchType == "duo-fpp") | 

          (train.matchType == "duo") | 

          (train.matchType == "normal-duo-fpp") | 

          (train.matchType == "normal-duo"), "matchType_1"] = "duo"



train.loc[(train.matchType == "squad-fpp") | 

          (train.matchType == "squad") | 

          (train.matchType == "normal-squad-fpp") | 

          (train.matchType == "normal-squad"), "matchType_1"] = "squad"



train.loc[(train.matchType == "flarefpp") | 

          (train.matchType == "flaretpp") | 

          (train.matchType == "crashfpp") | 

          (train.matchType == "crashtpp"), "matchType_1"] = "etc"
# 게임시점별 분류(Division by viewpoint)

train["matchType_2"] = "-"

train.loc[(train.matchType == "solo-fpp") | 

          (train.matchType == "duo-fpp") | 

          (train.matchType == "squad-fpp") | 

          (train.matchType == "normal-solo-fpp") | 

          (train.matchType == "normal-duo-fpp") | 

          (train.matchType == "normal-squad-fpp") | 

          (train.matchType == "crashfpp") | 

          (train.matchType == "flarefpp"), "matchType_2"] = "fpp"



train.loc[(train.matchType == "solo") | 

          (train.matchType == "duo") | 

          (train.matchType == "squad") | 

          (train.matchType == "normal-solo") | 

          (train.matchType == "normal-duo") | 

          (train.matchType == "normal-squad") | 

          (train.matchType == "crashtpp") | 

          (train.matchType == "flaretpp"), "matchType_2"] = "tpp"
train["solo"] = 0

train["duo"] = 0

train["squad"] = 0

train["etc"] = 0



train.loc[train.matchType_1 == "solo", "solo"] = 1

train.loc[train.matchType_1 == "duo", "duo"] = 1

train.loc[train.matchType_1 == "squad", "squad"] = 1

train.loc[train.matchType_1 == "etc", "etc"] = 1
train["fpp"] = 0

train["tpp"] = 0



train.loc[train.matchType_2 == "fpp", "fpp"] = 1

train.loc[train.matchType_2 == "tpp", "tpp"] = 1
print(list(train.columns[train.dtypes != "O"]))
feature = ["assists", "boosts", "damageDealt", "DBNOs", "headshotKills", "heals", 

           "killPlace", "killPoints", "kills", "killStreaks", "longestKill", 

           "matchDuration", "maxPlace", "rankPoints", "revives", "rideDistance", 

           "roadKills", "swimDistance", "teamKills", "vehicleDestroys", "walkDistance", "weaponsAcquired", "winPoints", "player"]
feature_1 = ["matchId", "assists", "boosts", "damageDealt", "DBNOs", "headshotKills", "heals", 

             "killPlace", "killPoints", "kills", "killStreaks", "longestKill", 

             "revives", "rideDistance", "roadKills", "swimDistance", "teamKills", 

             "vehicleDestroys", "walkDistance", "weaponsAcquired", "winPoints"]
feature_2 = ["matchDuration", "maxPlace", "rankPoints", "player", "fpp", "tpp"]
for i in list(train.columns[train.dtypes != "O"]):

    print(i, ":", sum(train[i].isna()))
# for i in pd.unique(train.matchId):

#     train.loc[(train.matchId == i) & (train.killPoints == 0), "killPoints"] = np.mean(train.loc[train.matchId == i, "killPoints"])
# for i in pd.unique(train.matchId):

#     train.loc[(train.matchId == i) & (train.winPoints == 0), "winPoints"] = np.mean(train.loc[train.matchId == i, "winPoints"])
np.sum(train.winPlacePerc.isna())
train = train.loc[train.winPlacePerc.notna(), :]
# solo_minmax = MinMaxScaler()

# duo_minmax = MinMaxScaler()

# squad_minmax = MinMaxScaler()

# etc_minmax = MinMaxScaler()
# solo_minmax.fit(train.loc[train.matchType_1 == "solo", feature])

# duo_minmax.fit(train.loc[train.matchType_1 == "duo", feature])

# squad_minmax.fit(train.loc[train.matchType_1 == "squad", feature])

# etc_minmax.fit(train.loc[train.matchType_1 == "etc", feature])
# solo_scale = solo_minmax.transform(train.loc[train.matchType_1 == "solo", feature])

# duo_scale = duo_minmax.transform(train.loc[train.matchType_1 == "duo", feature])

# squad_scale = squad_minmax.transform(train.loc[train.matchType_1 == "squad", feature])

# etc_scale = etc_minmax.transform(train.loc[train.matchType_1 == "etc", feature])
# solo_scale = pd.DataFrame(solo_scale, columns=feature)

# duo_scale = pd.DataFrame(duo_scale, columns=feature)

# squad_scale = pd.DataFrame(squad_scale, columns=feature)

# etc_scale = pd.DataFrame(etc_scale, columns=feature)
# _ = train.loc[train.matchType_1 == "solo", ["matchId", "matchType_1", "matchType_2", "solo", "duo", "squad", "etc", "fpp", "tpp", "winPlacePerc"]]

# _ = _.reset_index()

# solo_scale = pd.concat([solo_scale, _], axis=1)
# _ = train.loc[train.matchType_1 == "duo", ["matchId", "matchType_1", "matchType_2", "solo", "duo", "squad", "etc", "fpp", "tpp", "winPlacePerc"]]

# _ = _.reset_index()

# duo_scale = pd.concat([duo_scale, _], axis=1)
# _ = train.loc[train.matchType_1 == "squad", ["matchId", "matchType_1", "matchType_2", "solo", "duo", "squad", "etc", "fpp", "tpp", "winPlacePerc"]]

# _ = _.reset_index()

# squad_scale = pd.concat([squad_scale, _], axis=1)
# _ = train.loc[train.matchType_1 == "etc", ["matchId", "matchType_1", "matchType_2", "solo", "duo", "squad", "etc", "fpp", "tpp", "winPlacePerc"]]

# _ = _.reset_index()

# etc_scale = pd.concat([etc_scale, _], axis=1)
# X = pd.concat([solo_scale, duo_scale, squad_scale, etc_scale])
# for i in ["boosts", "damageDealt", "heals", "killPlace", "kills", "killStreaks", "longestKill", "walkDistance", "weaponsAcquired"]:

#     for t in ["tpp", "fpp"]:

#         train.loc[train.matchType_2 == t, i] = (train.loc[train.matchType_2 == t, i] - np.min(train.loc[train.matchType_2 == t, i])) / (np.max(train.loc[train.matchType_2 == t, i]) - np.min(train.loc[train.matchType_2 == t, i]))
train.set_index("Id", inplace=True)

train.index.name = "Id"
temp_1 = train.loc[:, feature_1]

temp_2 = train.loc[:, feature_2]
def minmax(attr):

    if max(attr) - min(attr) == 0:

        return 0

    return (attr - min(attr)) / (max(attr) - min(attr))
temp_1.groupby("matchId").transform(minmax)

for i in temp_2.columns[:4]:

    temp_2[i] = (temp_2[i] - min(temp_2[i])) / (max(temp_2[i]) - min(temp_2[i]))
X = pd.merge(temp_1, temp_2, on="Id")

X = pd.merge(X, train.loc[:, ["matchType_1", "winPlacePerc"]], on="Id")
X.reset_index()
# for i in ["boosts", "damageDealt", "heals", "killPlace", "kills", "killStreaks", "longestKill", "walkDistance", "weaponsAcquired"]:

#     for idx, g in enumerate(pd.unique(train.matchId)):

#         train.loc[train.matchId == g, i] = robust_scale(train.loc[train.matchId == g, i])
# for i in ["boosts", "damageDealt", "heals", "killPlace", "kills", "killStreaks", "longestKill", "walkDistance", "weaponsAcquired"]:

#     for idx, g in enumerate(pd.unique(train.matchId)):

#         train.loc[train.matchId == g, i] = minmax_scale(train.loc[train.matchId == g, i])
# plt.figure(figsize=(20, 20))

# plt.suptitle("Assists distribution by matchType", fontsize = 20)



# for idx, v in enumerate(pd.unique(train.matchType)):

#     plt.subplot(4, 4, idx+1)

#     plt.hist(train[train.matchType == v]["assists"], density=True)

#     plt.title(v)

    

# plt.show()
# plt.figure(figsize=(20, 20))

# plt.suptitle("DamageDealt distribution by matchType", fontsize = 20)



# for idx, v in enumerate(pd.unique(train.matchType)):

#     plt.subplot(4, 4, idx+1)

#     plt.hist(train[train.matchType == v]["damageDealt"], density=True)

#     plt.title(v)

    

# plt.show()
# plt.figure(figsize=(25, 25))

# plt.suptitle("Continuous variables distribution by matchType(fpp-tpp)", fontsize = 20)



# for idx, v in enumerate(train.columns[train.dtypes != "O"]):

#     plt.subplot(6, 5, idx+1)

#     plt.hist(train[train.matchType_2 == "fpp"][v].dropna(), color = "red", alpha = 0.5, label = "fpp", density = True, cumulative = True)

#     plt.hist(train[train.matchType_2 == "tpp"][v].dropna(), color = "grey", alpha = 0.8, label = "tpp", density = True, cumulative = True)

#     plt.legend()

#     plt.title(v)

    

# plt.show()
# plt.figure(figsize=(25, 25))

# plt.suptitle("Continuous variables distribution by matchType(solo-duo-squad)", fontsize = 20)



# for idx, v in enumerate(train.columns[train.dtypes != "O"]):

#     plt.subplot(6, 5, idx+1)

#     plt.hist(train[train.matchType_1 == "solo"][v].dropna(), color = "red", alpha = 0.5, label = "solo", density = True)

#     plt.hist(train[train.matchType_1 == "duo"][v].dropna(), color = "grey", alpha = 0.8, label = "duo", density = True)

#     plt.hist(train[train.matchType_1 == "squad"][v].dropna(), color = "yellow", alpha = 0.2, label = "squad", density = True)

#     plt.legend()

#     plt.title(v)

    

# plt.show()
#train.loc[train.killPoints == 0, "killPoints"] = np.mean(train.loc[train.killPoints != 0, "killPoints"])

#train.loc[train.winPoints == 0, "winPoints"] = np.mean(train.loc[train.winPoints != 0, "winPoints"])
#train = train[train.killPoints != 0]

#train = train[train.winPoints != 0]
train.corr()
corre = train.corr()

pd.DataFrame(data = corre[(corre>0.35) | (corre < -0.35)]["winPlacePerc"].rename("Correlation"))
print("Name: ", feature, "\nCount: ", len(feature))
list_feat = ["assists", "boosts", "damageDealt", "DBNOs", "headshotKills", "heals", 

             "killPlace", "killPoints", "kills", "killStreaks", "longestKill", 

             "matchDuration", "maxPlace", "rankPoints", "revives", "rideDistance", 

             "roadKills", "swimDistance", "teamKills", "vehicleDestroys", "walkDistance", 

             "weaponsAcquired", "winPoints", "player", "fpp", "tpp"]
list_feat_1 = ["assists", "boosts", "damageDealt", "DBNOs", "headshotKills", "heals", 

               "killPlace", "killPoints", "kills", "killStreaks", "longestKill", 

               "matchDuration", "maxPlace", "rankPoints", "revives", "rideDistance", 

               "roadKills", "swimDistance", "teamKills", "vehicleDestroys", "walkDistance", 

               "weaponsAcquired", "winPoints", "player", "fpp", "tpp", "matchId"]
train = X
# 모델 1(solo)

model_1 = keras.models.Sequential()



model_1.add(Dense(32, input_dim=len(list_feat), activation="elu", kernel_initializer="he_normal"))

model_1.add(Dense(64, activation="elu", kernel_initializer="he_normal"))

model_1.add(Dense(128, activation="elu", kernel_initializer="he_normal"))

model_1.add(keras.layers.Dropout(0.25))



model_1.add(Dense(256, activation="elu", kernel_initializer="he_normal"))

model_1.add(Dense(256, activation="elu", kernel_initializer="he_normal"))

model_1.add(keras.layers.Dropout(0.25))



model_1.add(Dense(128, activation="elu", kernel_initializer="he_normal"))

model_1.add(Dense(64, activation="elu", kernel_initializer="he_normal"))

model_1.add(Dense(32, activation="elu", kernel_initializer="he_normal"))

model_1.add(keras.layers.Dropout(0.25))



model_1.add(Dense(1, activation="sigmoid"))



model_1.compile(optimizer="RMSprop", loss='MAE', metrics=["MAE"])
x_train = train.loc[train.matchType_1 == "solo", list_feat]

y_train = train.loc[train.matchType_1 == "solo", ["winPlacePerc"]]
model_1.fit(x=x_train, y=y_train, epochs=50, batch_size=10000, validation_split=0.2, shuffle=True)

model_1.fit(x=x_train, y=y_train, epochs=30, batch_size=2000, validation_split=0.2, shuffle=True)
# for epoch in tqdm.tqdm(range(1, 2)):

#     for i in pd.unique(x_train.matchId):

#         model_1.fit(x=x_train.loc[x_train.matchId == i, list_feat], y=y_train.loc[y_train.matchId == i, "winPlacePerc"], batch_size=len(y_train.loc[y_train.matchId == i, "winPlacePerc"]), epochs=1, verbose=0)
model_1.save("model_1_solo.h5")

# model_1 = keras.models.load_model("../input/model-pubg/model_1_fpp.h5")
# model_1.fit(x=x_train, y=y_train, epochs=50, batch_size=100, validation_split=0.2, shuffle=True)
# keras.models.save_model(model_1, "model_1_fpp.h5")
# 모델 2(duo)

model_2 = keras.models.Sequential()



model_2.add(Dense(32, input_dim=len(list_feat), activation="elu", kernel_initializer="he_normal"))

model_2.add(Dense(64, activation="elu", kernel_initializer="he_normal"))

model_2.add(Dense(128, activation="elu", kernel_initializer="he_normal"))

model_2.add(keras.layers.Dropout(0.25))



model_2.add(Dense(256, activation="elu", kernel_initializer="he_normal"))

model_2.add(Dense(256, activation="elu", kernel_initializer="he_normal"))

model_2.add(keras.layers.Dropout(0.25))



model_2.add(Dense(128, activation="elu", kernel_initializer="he_normal"))

model_2.add(Dense(64, activation="elu", kernel_initializer="he_normal"))

model_2.add(Dense(32, activation="elu", kernel_initializer="he_normal"))

model_2.add(keras.layers.Dropout(0.25))



model_2.add(Dense(1, activation="sigmoid"))



model_2.compile(optimizer="RMSprop", loss='MAE', metrics=["MAE"])
x_train = train.loc[train.matchType_1 == "duo", list_feat]

y_train = train.loc[train.matchType_1 == "duo", ["winPlacePerc"]]
model_2.fit(x=x_train, y=y_train, epochs=50, batch_size=10000, validation_split=0.2, shuffle=True)

model_2.fit(x=x_train, y=y_train, epochs=40, batch_size=2000, validation_split=0.2, shuffle=True)
# for epoch in tqdm.tqdm(range(1, 2)):

#     for i in pd.unique(x_train.matchId):

#         model_2.fit(x=x_train.loc[x_train.matchId == i, list_feat], y=y_train.loc[y_train.matchId == i, "winPlacePerc"], batch_size=len(y_train.loc[y_train.matchId == i, "winPlacePerc"]), epochs=1, verbose=0)
model_2.save("model_2_duo.h5")

# model_1 = keras.models.load_model("../input/model-pubg/model_1_fpp.h5")
# 모델 3(squad)

model_3 = keras.models.Sequential()



model_3.add(Dense(32, input_dim=len(list_feat), activation="elu", kernel_initializer="he_normal"))

model_3.add(Dense(64, activation="elu", kernel_initializer="he_normal"))

model_3.add(Dense(128, activation="elu", kernel_initializer="he_normal"))

model_3.add(keras.layers.Dropout(0.25))



model_3.add(Dense(256, activation="elu", kernel_initializer="he_normal"))

model_3.add(Dense(256, activation="elu", kernel_initializer="he_normal"))

model_3.add(keras.layers.Dropout(0.35))



model_3.add(Dense(128, activation="elu", kernel_initializer="he_normal"))

model_3.add(Dense(64, activation="elu", kernel_initializer="he_normal"))

model_3.add(Dense(32, activation="elu", kernel_initializer="he_normal"))

model_3.add(keras.layers.Dropout(0.25))



model_3.add(Dense(1, activation="sigmoid"))



model_3.compile(optimizer="RMSprop", loss='MAE', metrics=["MAE"])
x_train = train.loc[train.matchType_1 == "squad", list_feat]

y_train = train.loc[train.matchType_1 == "squad", ["winPlacePerc"]]
model_3.fit(x=x_train, y=y_train, epochs=60, batch_size=10000, validation_split=0.2, shuffle=True)

model_3.fit(x=x_train, y=y_train, epochs=50, batch_size=3000, validation_split=0.2, shuffle=True)
# for epoch in tqdm.tqdm(range(1, 2)):

#     for i in pd.unique(x_train.matchId):

#         model_3.fit(x=x_train.loc[x_train.matchId == i, list_feat], y=y_train.loc[y_train.matchId == i, "winPlacePerc"], batch_size=len(y_train.loc[y_train.matchId == i, "winPlacePerc"]), epochs=1, verbose=0)
model_3.save("model_3_squad.h5")

# model_1 = keras.models.load_model("../input/model-pubg/model_1_fpp.h5")
# 모델 4(etc)

model_4 = keras.models.Sequential()



model_4.add(Dense(32, input_dim=len(list_feat), activation="elu", kernel_initializer="he_normal"))

model_4.add(Dense(64, activation="elu", kernel_initializer="he_normal"))

model_4.add(Dense(128, activation="elu", kernel_initializer="he_normal"))

model_4.add(keras.layers.Dropout(0.25))



model_4.add(Dense(128, activation="elu", kernel_initializer="he_normal"))

model_4.add(Dense(64, activation="elu", kernel_initializer="he_normal"))

model_4.add(Dense(32, activation="elu", kernel_initializer="he_normal"))

model_4.add(keras.layers.Dropout(0.25))



model_4.add(Dense(1, activation="sigmoid"))



model_4.compile(optimizer="RMSprop", loss='MAE', metrics=["MAE"])
x_train = train.loc[train.matchType_1 == "etc", list_feat]

y_train = train.loc[train.matchType_1 == "etc", ["winPlacePerc"]]
model_4.fit(x=x_train, y=y_train, epochs=70, batch_size=10000, validation_split=0.2, shuffle=True)

model_4.fit(x=x_train, y=y_train, epochs=150, batch_size=1000, validation_split=0.2, shuffle=True)
# for epoch in tqdm.tqdm(range(1, 5)):

#     for i in pd.unique(x_train.matchId):

#         model_4.fit(x=x_train.loc[x_train.matchId == i, list_feat], y=y_train.loc[y_train.matchId == i, "winPlacePerc"], batch_size=len(y_train.loc[y_train.matchId == i, "winPlacePerc"]), epochs=1, verbose=0)
model_4.save("model_4_etc.h5")

# model_1 = keras.models.load_model("../input/model-pubg/model_1_fpp.h5")
del(train, x_train, y_train, X)
plt.figure(figsize=(30, 30))

plt.suptitle("model History", fontsize = 20)



plt.subplot(2, 2, 1)

plt.title("model_1")

plt.plot(model_1.history.history["mean_absolute_error"], label="training")

plt.plot(model_1.history.history["val_mean_absolute_error"], label="validation")

plt.axhline(0.3, c="red", linestyle="--")

plt.axhline(0.2, c="yellow", linestyle="--")

plt.axhline(0.15, c="green", linestyle="--")

plt.xticks(model_1.history.epoch)

plt.xlabel("Epoch")

plt.ylabel("MAE")

plt.legend()



plt.subplot(2, 2, 2)

plt.title("model_2")

plt.plot(model_2.history.history["mean_absolute_error"], label="training")

plt.plot(model_2.history.history["val_mean_absolute_error"], label="validation")

plt.axhline(0.3, c="red", linestyle="--")

plt.axhline(0.2, c="yellow", linestyle="--")

plt.axhline(0.15, c="green", linestyle="--")

plt.xticks(model_2.history.epoch)

plt.xlabel("Epoch")

plt.ylabel("MAE")

plt.legend()



plt.subplot(2, 2, 3)

plt.title("model_3")

plt.plot(model_3.history.history["mean_absolute_error"], label="training")

plt.plot(model_3.history.history["val_mean_absolute_error"], label="validation")

plt.axhline(0.3, c="red", linestyle="--")

plt.axhline(0.2, c="yellow", linestyle="--")

plt.axhline(0.15, c="green", linestyle="--")

plt.xticks(model_3.history.epoch)

plt.xlabel("Epoch")

plt.ylabel("MAE")

plt.legend()



plt.subplot(2, 2, 4)

plt.title("model_4")

plt.plot(model_4.history.history["mean_absolute_error"], label="training")

plt.plot(model_4.history.history["val_mean_absolute_error"], label="validation")

plt.axhline(0.3, c="red", linestyle="--")

plt.axhline(0.2, c="yellow", linestyle="--")

plt.axhline(0.15, c="green", linestyle="--")

plt.xticks(model_4.history.epoch)

plt.xlabel("Epoch")

plt.ylabel("MAE")

plt.legend()



plt.show()
test = pd.read_csv("../input/test_V2.csv")
print("Check The NA value in test data")

for i in list(test.columns[test.dtypes != "O"]):

    print(i, ":", sum(test[i].isna()))
len(pd.unique(test.matchId)), sum(test.groupby("matchId").size() < 9)
temp = pd.DataFrame(test.groupby("matchId").size(), columns=["player"])

temp.reset_index(level=0, inplace=True)

test = test.merge(temp, left_on="matchId", right_on="matchId")
test["matchType_1"] = "-"

test.loc[(test.matchType == "solo-fpp") | 

         (test.matchType == "solo") | 

         (test.matchType == "normal-solo-fpp") | 

         (test.matchType == "normal-solo"), "matchType_1"] = "solo"



test.loc[(test.matchType == "duo-fpp") | 

         (test.matchType == "duo") | 

         (test.matchType == "normal-duo-fpp") | 

         (test.matchType == "normal-duo"), "matchType_1"] = "duo"



test.loc[(test.matchType == "squad-fpp") | 

         (test.matchType == "squad") | 

         (test.matchType == "normal-squad-fpp") | 

         (test.matchType == "normal-squad"), "matchType_1"] = "squad"



test.loc[(test.matchType == "flarefpp") | 

         (test.matchType == "flaretpp") | 

         (test.matchType == "crashfpp") | 

         (test.matchType == "crashtpp"), "matchType_1"] = "etc"
test["matchType_2"] = "-"

test.loc[(test.matchType == "solo-fpp") | 

         (test.matchType == "duo-fpp") | 

         (test.matchType == "squad-fpp") | 

         (test.matchType == "normal-solo-fpp") | 

         (test.matchType == "normal-duo-fpp") | 

         (test.matchType == "normal-squad-fpp") | 

         (test.matchType == "crashfpp") | 

         (test.matchType == "flarefpp"), "matchType_2"] = "fpp"



test.loc[(test.matchType == "solo") | 

         (test.matchType == "duo") | 

         (test.matchType == "squad") | 

         (test.matchType == "normal-solo") | 

         (test.matchType == "normal-duo") | 

         (test.matchType == "normal-squad") | 

         (test.matchType == "crashtpp") | 

         (test.matchType == "flaretpp"), "matchType_2"] = "tpp"
test["solo"] = 0

test["duo"] = 0

test["squad"] = 0

test["etc"] = 0



test.loc[test.matchType_1 == "solo", "solo"] = 1

test.loc[test.matchType_1 == "duo", "duo"] = 1

test.loc[test.matchType_1 == "squad", "squad"] = 1

test.loc[test.matchType_1 == "etc", "etc"] = 1
test["fpp"] = 0

test["tpp"] = 0



test.loc[test.matchType_2 == "fpp", "fpp"] = 1

test.loc[test.matchType_2 == "tpp", "tpp"] = 1
# for i in pd.unique(train.matchId):

#     train.loc[(train.matchId == i) & (train.killPoints == 0), "killPoints"] = np.mean(train.loc[train.matchId == i, "killPoints"])
# for i in pd.unique(train.matchId):

#     train.loc[(train.matchId == i) & (train.winPoints == 0), "winPoints"] = np.mean(train.loc[train.matchId == i, "winPoints"])
# solo_scale = solo_minmax.transform(test.loc[test.matchType_1 == "solo", feature])

# duo_scale = duo_minmax.transform(test.loc[test.matchType_1 == "duo", feature])

# squad_scale = squad_minmax.transform(test.loc[test.matchType_1 == "squad", feature])

# etc_scale = etc_minmax.transform(test.loc[test.matchType_1 == "etc", feature])
# solo_scale = pd.DataFrame(solo_scale, columns=feature)

# duo_scale = pd.DataFrame(duo_scale, columns=feature)

# squad_scale = pd.DataFrame(squad_scale, columns=feature)

# etc_scale = pd.DataFrame(etc_scale, columns=feature)
# _ = test.loc[test.matchType_1 == "solo", ["matchType_1", "matchType_2", "solo", "duo", "squad", "etc", "fpp", "tpp", "winPlacePerc", "Id"]]

# _ = _.reset_index()

# solo_scale = pd.concat([solo_scale, _], axis=1)
# _ = test.loc[test.matchType_1 == "duo", ["matchType_1", "matchType_2", "solo", "duo", "squad", "etc", "fpp", "tpp", "winPlacePerc", "Id"]]

# _ = _.reset_index()

# duo_scale = pd.concat([duo_scale, _], axis=1)
# _ = test.loc[test.matchType_1 == "squad", ["matchType_1", "matchType_2", "solo", "duo", "squad", "etc", "fpp", "tpp", "winPlacePerc", "Id"]]

# _ = _.reset_index()

# squad_scale = pd.concat([squad_scale, _], axis=1)
# _ = test.loc[test.matchType_1 == "etc", ["matchType_1", "matchType_2", "solo", "duo", "squad", "etc", "fpp", "tpp", "winPlacePerc", "Id"]]

# _ = _.reset_index()

# etc_scale = pd.concat([etc_scale, _], axis=1)
# X = pd.concat([solo_scale, duo_scale, squad_scale, etc_scale])
# for i in ["boosts", "damageDealt", "heals", "killPlace", "kills", "killStreaks", "longestKill", "walkDistance", "weaponsAcquired"]:

#     for t in ["tpp", "fpp"]:

#         test.loc[test.matchType_2 == t, i] = (test.loc[test.matchType_2 == t, i] - np.min(test.loc[test.matchType_2 == t, i])) / (np.max(test.loc[test.matchType_2 == t, i]) - np.min(test.loc[test.matchType_2 == t, i]))
test.set_index("Id", inplace=True)

test.index.name = "Id"
temp_1 = test.loc[:, feature_1]

temp_2 = test.loc[:, feature_2]
def minmax(attr):

    if max(attr) - min(attr) == 0:

        return 0

    return (attr - min(attr)) / (max(attr) - min(attr))
temp_1.groupby("matchId").transform(minmax)

for i in temp_2.columns[:4]:

    temp_2[i] = (temp_2[i] - min(temp_2[i])) / (max(temp_2[i]) - min(temp_2[i]))
X = pd.merge(temp_1, temp_2, on="Id")

X = pd.merge(X, test.loc[:, ["matchType_1", "winPlacePerc"]], on="Id")
X.reset_index()
# for i in ["boosts", "damageDealt", "heals", "killPlace", "kills", "killStreaks", "longestKill", "walkDistance", "weaponsAcquired"]:

#     for idx, g in enumerate(pd.unique(test.matchId)):

#         test.loc[test.matchId == g, i] = robust_scale(test.loc[test.matchId == g, i])
# for i in ["boosts", "damageDealt", "heals", "killPlace", "kills", "killStreaks", "longestKill", "walkDistance", "weaponsAcquired"]:

#     for idx, g in enumerate(pd.unique(test.matchId)):

#         test.loc[test.matchId == g, i] = minmax_scale(test.loc[test.matchId == g, i])
test = X

test.reset_index()
result_1 = model_1.predict(test.loc[test.matchType_1 == "solo", list_feat])

result_2 = model_2.predict(test.loc[test.matchType_1 == "duo", list_feat])

result_3 = model_3.predict(test.loc[test.matchType_1 == "squad", list_feat])

result_4 = model_4.predict(test.loc[test.matchType_1 == "etc", list_feat])
temp = pd.DataFrame(test.loc[test.matchType_1 == "solo", "Id"]).append(pd.DataFrame(test.loc[test.matchType_1 == "duo", "Id"])).append(pd.DataFrame(test.loc[test.matchType_1 == "squad", "Id"])).append(pd.DataFrame(test.loc[test.matchType_1 == "etc", "Id"]))

_ = pd.DataFrame(result_1, columns = ["winPlacePerc"]).append(pd.DataFrame(result_2, columns = ["winPlacePerc"])).append(pd.DataFrame(result_3, columns = ["winPlacePerc"])).append(pd.DataFrame(result_4, columns = ["winPlacePerc"]))
result = pd.concat([temp.reset_index(drop=True), _.reset_index(drop=True)], axis=1)
np.sum(result.winPlacePerc.isna())
np.sum(result.winPlacePerc < 0)
np.sum(result.winPlacePerc > 1)
result.loc[result.winPlacePerc.isna(), "winPlacePerc"] = 0

result.loc[result.winPlacePerc < 0, "winPlacePerc"] = 0

result.loc[result.winPlacePerc > 1, "winPlacePerc"] = 1

result.to_csv('submission.csv', index=False)