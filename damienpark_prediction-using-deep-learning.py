import numpy as np

import pandas as pd



import matplotlib.pyplot as plt



import keras

from keras.layers.core import Dense

from keras.layers.normalization import BatchNormalization
train = pd.read_csv("../input/train_V2.csv")
train.head()
train.describe().T
pd.DataFrame(train.dtypes, columns=["Type"])
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
feature_2 = ["matchDuration", "maxPlace", "rankPoints", "player", "fpp", "tpp", "matchType_1"]
for i in list(train.columns[train.dtypes != "O"]):

    print(i, ":", sum(train[i].isna()))
np.sum(train.winPlacePerc.isna())
train = train.loc[train.winPlacePerc.notna(), :]
train.set_index("Id", inplace=True)

train.index.name = "Id"
temp_1 = train.loc[:, feature_1]

temp_2 = train.loc[:, feature_2]
def minmax(attr):

    return (attr - min(attr)) / (max(attr) - min(attr))
temp_1 = temp_1.groupby("matchId").transform(minmax)

temp_2 = temp_2.groupby("matchType_1").transform(minmax)



# for i in temp_2.columns[:4]:

#     temp_2[i] = (temp_2[i] - min(temp_2[i])) / (max(temp_2[i]) - min(temp_2[i]))
temp_1.fillna(value=0, inplace=True)

temp_2.fillna(value=0, inplace=True)
X = pd.merge(temp_1, temp_2, on="Id")

X = pd.merge(X, train.loc[:, ["matchType_1", "winPlacePerc"]], on="Id")
del temp_1, temp_2

X = X.reset_index()
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
x_train = X.loc[X.matchType_1 == "solo", list_feat]

y_train = X.loc[X.matchType_1 == "solo", ["winPlacePerc"]]
model_1.fit(x=x_train, y=y_train, epochs=50, batch_size=2000, validation_split=0.2, shuffle=True)
model_1.save("model_1_solo.h5")
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
x_train = X.loc[X.matchType_1 == "duo", list_feat]

y_train = X.loc[X.matchType_1 == "duo", ["winPlacePerc"]]
model_2.fit(x=x_train, y=y_train, epochs=75, batch_size=2000, validation_split=0.2, shuffle=True)
model_2.save("model_2_duo.h5")
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
x_train = X.loc[X.matchType_1 == "squad", list_feat]

y_train = X.loc[X.matchType_1 == "squad", ["winPlacePerc"]]
model_3.fit(x=x_train, y=y_train, epochs=25, batch_size=3000, validation_split=0.2, shuffle=True)
model_3.save("model_3_squad.h5")
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
x_train = X.loc[X.matchType_1 == "etc", list_feat]

y_train = X.loc[X.matchType_1 == "etc", ["winPlacePerc"]]
model_4.fit(x=x_train, y=y_train, epochs=100, validation_split=0.2, shuffle=True)
model_4.save("model_4_etc.h5")

# model_1 = keras.models.load_model("../input/model-pubg/model_1_fpp.h5")
del(train, x_train, y_train, X)
plt.figure(figsize=(30, 15))

plt.suptitle("model History", fontsize = 20)



plt.subplot(1, 2, 1)

plt.title("model_1")

plt.plot(model_1.history.history["mean_absolute_error"], label="training")

plt.plot(model_1.history.history["val_mean_absolute_error"], label="validation")

plt.axhline(0.1, c="red", linestyle="--")

plt.axhline(0.05, c="yellow", linestyle="--")

plt.axhline(0.025, c="green", linestyle="--")

plt.xticks(model_1.history.epoch)

plt.xlabel("Epoch")

plt.ylabel("MAE")

plt.legend()



plt.subplot(1, 2, 2)

plt.title("model_2")

plt.plot(model_2.history.history["mean_absolute_error"], label="training")

plt.plot(model_2.history.history["val_mean_absolute_error"], label="validation")

plt.axhline(0.1, c="red", linestyle="--")

plt.axhline(0.05, c="yellow", linestyle="--")

plt.axhline(0.025, c="green", linestyle="--")

plt.xticks(model_2.history.epoch)

plt.xlabel("Epoch")

plt.ylabel("MAE")

plt.legend()



plt.show()
plt.figure(figsize=(30, 15))

plt.suptitle("model History", fontsize = 20)



plt.subplot(1, 2, 1)

plt.title("model_3")

plt.plot(model_3.history.history["mean_absolute_error"], label="training")

plt.plot(model_3.history.history["val_mean_absolute_error"], label="validation")

plt.axhline(0.1, c="red", linestyle="--")

plt.axhline(0.05, c="yellow", linestyle="--")

plt.axhline(0.025, c="green", linestyle="--")

plt.xticks(model_3.history.epoch)

plt.xlabel("Epoch")

plt.ylabel("MAE")

plt.legend()



plt.subplot(1, 2, 2)

plt.title("model_4")

plt.plot(model_4.history.history["mean_absolute_error"], label="training")

plt.plot(model_4.history.history["val_mean_absolute_error"], label="validation")

plt.axhline(0.1, c="red", linestyle="--")

plt.axhline(0.05, c="yellow", linestyle="--")

plt.axhline(0.025, c="green", linestyle="--")

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
test.set_index("Id", inplace=True)

test.index.name = "Id"
temp_1 = test.loc[:, feature_1]

temp_2 = test.loc[:, feature_2]
temp_1 = temp_1.groupby("matchId").transform(minmax)

temp_2 = temp_2.groupby("matchType_1").transform(minmax)



# for i in temp_2.columns[:4]:

#     temp_2[i] = (temp_2[i] - min(temp_2[i])) / (max(temp_2[i]) - min(temp_2[i]))
temp_1.fillna(value=0, inplace=True)

temp_2.fillna(value=0, inplace=True)
X = pd.merge(temp_1, temp_2, on="Id")

X = pd.merge(X, test.loc[:, ["matchType_1", "winPlacePerc"]], on="Id")
del temp_1, temp_2

X.reset_index()
test = X

test = test.reset_index()
solo = test[test.matchType_1 == "solo"].index

duo = test[test.matchType_1 == "duo"].index

squad = test[test.matchType_1 == "squad"].index

etc = test[test.matchType_1 == "etc"].index



solo = solo.append([duo, squad, etc])



temp = pd.DataFrame(index=solo)

temp.index.name = "Id"
result_1 = model_1.predict(test.loc[test.matchType_1 == "solo", list_feat])

result_2 = model_2.predict(test.loc[test.matchType_1 == "duo", list_feat])

result_3 = model_3.predict(test.loc[test.matchType_1 == "squad", list_feat])

result_4 = model_4.predict(test.loc[test.matchType_1 == "etc", list_feat])
temp = pd.DataFrame(test.loc[test.matchType_1 == "solo", "Id"]).append(pd.DataFrame(test.loc[test.matchType_1 == "duo", "Id"])).append(pd.DataFrame(test.loc[test.matchType_1 == "squad", "Id"])).append(pd.DataFrame(test.loc[test.matchType_1 == "etc", "Id"]))

_ = pd.DataFrame(result_1, columns = ["winPlacePerc"]).append(pd.DataFrame(result_2, columns = ["winPlacePerc"])).append(pd.DataFrame(result_3, columns = ["winPlacePerc"])).append(pd.DataFrame(result_4, columns = ["winPlacePerc"]))
result = pd.concat([temp.reset_index(drop=True), _.reset_index(drop=True)], axis=1)
print(np.sum(result.winPlacePerc.isna()), np.sum(result.winPlacePerc < 0), np.sum(result.winPlacePerc > 1))
result.loc[result.winPlacePerc.isna(), "winPlacePerc"] = 0

result.loc[result.winPlacePerc < 0, "winPlacePerc"] = 0

result.loc[result.winPlacePerc > 1, "winPlacePerc"] = 1

result.to_csv('submission.csv', index=False)