# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Imports

import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 500)
# Training data is in the competition dataset as usual

train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)



train_df.head()
train_df.shape # (509762, 49) (rows, columns)
train_df.columns
train_df.info()
train_df.isna().sum().sort_values(ascending=False)[0:12] #11 columns are missing data
print(f"Total games: {train_df.GameId.nunique()}")

print(f"Total plays: {train_df.PlayId.nunique()}")

print(f"Total players: {train_df.NflId.nunique()}")

print(f"Total rushers: {train_df.NflIdRusher.nunique()}")
print(f"Total Teams: {train_df.PossessionTeam.nunique()}")

playPoss = train_df.groupby(["PlayId", 'PossessionTeam']).GameId.count().reset_index()

playPoss.columns = ["PlayId", 'PossessionTeam', 'PlayersOnField']

playPoss.PlayersOnField.describe() #22 players on the field for each play, as it should be
teamPoss = playPoss.groupby("PossessionTeam").PlayId.count().sort_values(ascending=True)

print(sum(teamPoss.values)) #2371, boom
#All Unique Team Plays

plt.figure(figsize=(20,10))

plt.barh(teamPoss.index, teamPoss.values, color="firebrick")

plt.title("All Team Plays", weight="bold", fontsize=20)

plt.xlabel("Number of Plays", fontsize=16)

plt.ylabel("")



plt.grid(axis="x")

plt.show()
playsPerGame = train_df.groupby("GameId").PlayId.nunique()

playsPerGame.sum() #23171, boom
playsPerGame.describe() #wow which game had 85 plays that is bonkers
#All Unique Plays per game

plt.figure(figsize=(20,10))

plt.hist(playsPerGame.values, color="firebrick")

plt.title("Plays per Game", weight="bold", fontsize=20)

plt.xlabel("Number of Plays", fontsize=16)

plt.ylabel("")



plt.show()
playDF = pd.DataFrame(playsPerGame)

playDF.loc[playDF.PlayId > 80].index[0] #2017121000
train_df.loc[train_df.GameId == 2017121000].head() #It's the SNOW BOWL
stads = pd.DataFrame(train_df.groupby("GameId").Stadium.max().reset_index())

stads.columns = ["GameId", "Stadium"]

stadGames = stads.groupby("Stadium").count().sort_values(by="GameId", ascending=True)

print(stadGames.GameId.sum()) #512 boom
#All Unique Stadiums by games

plt.figure(figsize=(20,10))

plt.barh(stadGames.index, stadGames.GameId, color="seagreen")

plt.title("Games per Stadium", weight="bold", fontsize=20)

plt.xlabel("Number of Games", fontsize=16)

plt.ylabel("")



plt.grid(axis="x")

plt.show() #ugh more pats
yardsToGo = train_df.groupby("PlayId").Distance.max()

yardsToGo.describe() #ha what play had 40 yards to go, that is also bonkers
#All Unique Plays per game

plt.figure(figsize=(20,10))

plt.hist(yardsToGo.values, color="purple")

plt.title("Yards to Go per Play", weight="bold", fontsize=20)

plt.xlabel("Number of Yards", fontsize=16)

plt.ylabel("Count")



plt.show()
downOfPlay = train_df.groupby("PlayId").Down.max()

downOfPlay.describe()
#All Unique downs per play

plt.figure(figsize=(20,10))

plt.hist(downOfPlay.values, color="purple")

plt.title("Down of Play", weight="bold", fontsize=20)

plt.xlabel("Down Number", fontsize=16)

plt.ylabel("Count")



plt.show() #real ground-breaking stuff here
yardlineOfPlay = train_df.groupby(["PlayId", "PlayDirection"]).YardLine.max().reset_index()

yardlineOfPlayDF = yardlineOfPlay.groupby(["YardLine", "PlayDirection"]).count().reset_index()

yardlineOfPlayDF.columns = ["YardLine", "PlayDirection", "Count"]

left = yardlineOfPlayDF.loc[yardlineOfPlayDF.PlayDirection == "left"].sort_values("YardLine", ascending=False)

right = yardlineOfPlayDF.loc[yardlineOfPlayDF.PlayDirection == "right"].sort_values("YardLine", ascending=True)



sortedLeftRight = pd.concat([right, left]).reset_index(drop=True)

sortedLeftRight.head()
plt.figure(figsize=(20,10))



plt.bar(sortedLeftRight.index, sortedLeftRight.Count, color="purple")



plt.title("Plays vs YardLine", weight="bold", fontsize=20)

plt.xlabel("YardLine", fontsize=16)

plt.ylabel("Count")



plt.show() #most plays start on the 25 yeard line, brilliant.
#without direction

sortedLeftRight2 = sortedLeftRight.groupby("YardLine").Count.sum()



plt.figure(figsize=(20,10))



plt.bar(sortedLeftRight2.index, sortedLeftRight2.values, color="purple")



plt.title("Plays vs YardLine", weight="bold", fontsize=20)

plt.xlabel("YardLine", fontsize=16)

plt.ylabel("Count")



plt.show() #most plays start on the 25 yeard line, brilliant.
quarts = pd.DataFrame(train_df.groupby("PlayId").Quarter.max()).reset_index()

quarts.columns = ["PlayId", "Quarter"]



quartsCount = quarts.groupby("Quarter").count() 

quartsCount#quarter 5?!?
#All Unique Plays per quarter

plt.figure(figsize=(20,10))

plt.bar(quartsCount.index, quartsCount.PlayId, color="lightblue")

plt.title("Quarter of Play", weight="bold", fontsize=20)

plt.xlabel("Quarter", fontsize=16)

plt.ylabel("Count")



plt.show()
yrds = train_df.groupby("PlayId").Yards.max()

yrds.describe()
sum(yrds.values < 0) #oh, can gain negative yards, right I knew that
print("Percent of plays ending in negative yards: " + str(round(100* 2561/23171,1)) + "%")
#All Unique Yards gained

plt.figure(figsize=(20,10))

plt.hist(yrds.values, color="maroon")

plt.title("Yards Gained Per Play", weight="bold", fontsize=20)

plt.xlabel("Yards Gained", fontsize=16)

plt.ylabel("Count")



plt.show() #quite right skewed
teamCols = ["PlayId", "PossessionTeam", "Down", "Quarter", "Distance", "PlayDirection", "DefensePersonnel", "OffensePersonnel",

           'FieldPosition', 'HomeScoreBeforePlay', 'VisitorScoreBeforePlay', 

                 'OffenseFormation', 'DefendersInTheBox', 'Week', 'Turf']#, 'Temperature', "Humidity"]

numCols = ["Distance", 'DefendersInTheBox', 'HomeScoreBeforePlay', 'VisitorScoreBeforePlay']#, 'Temperature', "Humidity"]

yCols = ["PlayId", "Yards"]
X = train_df[teamCols].drop_duplicates(subset="PlayId")

y = train_df[yCols].drop_duplicates(subset="PlayId")
X.Turf.value_counts()
X.Turf = X.Turf.map({'Grass': 'Grass', 'Natural Grass': 'Grass', 'Naturall Grass': 'Grass', 'Natural': 'Grass', 'Natural grass': 'Grass', 'grass': 'Grass',

           'natural grass': 'Grass'})

X.Turf = X.Turf.fillna("Turf")

X.Turf.value_counts()
X.DefendersInTheBox = X.DefendersInTheBox.fillna(X.DefendersInTheBox.mean())

#X.Temperature = X.Temperature.fillna(X.Temperature.mean())

#X.Humidity = X.Humidity.fillna(X.Humidity.mean())

X.OffenseFormation = X.OffenseFormation.fillna(X.OffenseFormation.value_counts().index[0])

X.FieldPosition = X.FieldPosition.fillna(X.FieldPosition.value_counts().index[0])
X.info()
y.info()
X = X.astype(str)

X[numCols] = X[numCols].astype(float)

playIds = X.PlayId

X.drop("PlayId", axis=1, inplace=True)

X.info()
y = y.astype(int)

y.drop("PlayId", axis=1, inplace=True)

y.info()
from sklearn.model_selection import train_test_split



X_train, X_val, Y_train, Y_val = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)
from catboost import CatBoostRegressor
cat_features = [0, 1, 2, 4, 5, 6, 7, 10, 12, 13]
model = CatBoostRegressor(iterations = 100, learning_rate = 0.5, depth = 15)
model.fit(X_train, Y_train, cat_features)
model.feature_importances_
from sklearn.metrics import mean_squared_error

val_preds = model.predict(X_val)

print(mean_squared_error(Y_val, val_preds))

print(mean_squared_error(Y_val, np.repeat(np.mean(Y_val.Yards), len(Y_val))))





accuracy = model.score(X_val,Y_val)

print(accuracy)
plt.hist(Y_val.Yards - val_preds)
pd.Series(val_preds).describe()
from xgboost import XGBRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder
X2 = X.copy()

X2.drop("Distance", axis=1, inplace=True)

X2.drop("DefendersInTheBox", axis=1, inplace=True)

X2.drop("HomeScoreBeforePlay", axis=1, inplace=True)

X2.drop("VisitorScoreBeforePlay", axis=1, inplace=True)

#X2.drop("Temperature", axis=1, inplace=True)

#X2.drop("Humidity", axis=1, inplace=True)



X2 = pd.get_dummies(X2)

X2["Distance"] = X.Distance

X2["DefendersInTheBox"] = X.DefendersInTheBox

X2["HomeScoreBeforePlay"] = X.HomeScoreBeforePlay

X2["VisitorScoreBeforePlay"] = X.VisitorScoreBeforePlay

#X2["Temperature"] = X.Temperature

#X2["Humidity"] = X.Humidity



print("X shape: : ", X2.shape)
X_train, X_val, Y_train, Y_val = train_test_split(X2, y, shuffle=True, test_size=0.2, random_state=42)
xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3)
xgb.fit(X_train, Y_train)
val_preds = xgb.predict(X_val)

print(mean_squared_error(Y_val, val_preds))

print(mean_squared_error(Y_val, np.repeat(np.mean(Y_val.Yards), len(Y_val))))





accuracy = xgb.score(X_val,Y_val)

print(accuracy)
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=20)
rfr.fit(X_train, Y_train)
val_preds = rfr.predict(X_val)

print(mean_squared_error(Y_val, val_preds))

print(mean_squared_error(Y_val, np.repeat(np.mean(Y_val.Yards), len(Y_val))))





accuracy = rfr.score(X_val,Y_val)

print(accuracy)
from kaggle.competitions import nflrush
X_train.columns
def make_my_predictions(test_df, sample_prediction_df):

    X_test = test_df[teamCols].drop_duplicates(subset="PlayId")

    

    temp = pd.DataFrame(np.zeros(shape = (1,len(X2.columns))))

    temp.columns = X2.columns



    temp["Distance"] = X_test.Distance

    temp["DefendersInTheBox"] = X_test.DefendersInTheBox

    temp["HomeScoreBeforePlay"] = X_test.HomeScoreBeforePlay

    temp["VisitorScoreBeforePlay"] = X_test.VisitorScoreBeforePlay



    temp["PossessionTeam_" + X_test.PossessionTeam.values[0]] = 1

    temp["Down_" + str(X_test.Down.values[0])] = 1

    temp["Quarter_" + str(X_test.Quarter.values[0])] = 1

    temp["PlayDirection_" + X_test.PlayDirection.values[0]] = 1

    temp["Week_" + str(X_test.Week.values[0])] = 1

    

    if (np.logical_not(pd.isnull(X_test.FieldPosition.values[0]))):

        temp["FieldPosition_" + X_test.FieldPosition.values[0]] = 1



    if (sum([X_test.OffensePersonnel.values[0] in x for x in X2.columns]) > 0):

        temp["OffensePersonnel_" + X_test.OffensePersonnel.values[0]] = 1

    if (sum([X_test.DefensePersonnel.values[0] in x for x in X2.columns]) > 0):

        temp["DefensePersonnel_" + X_test.DefensePersonnel.values[0]] = 1

    if (sum([X_test.OffenseFormation.values[0] in x for x in X2.columns]) > 0):   

        temp["OffenseFormation_" + X_test.OffenseFormation.values[0]] = 1

        

    X_test.Turf = X_test.Turf.map({'Grass': 'Grass', 'Natural Grass': 'Grass', 'Naturall Grass': 'Grass', 'Natural': 'Grass', 'Natural grass': 'Grass', 'grass': 'Grass',

           'natural grass': 'Grass'})

    X_test.Turf = X_test.Turf.fillna("Turf")

    temp["Turf_" + X_test.Turf.values[0]] = 1

    

    pred = xgb.predict(temp)

    sample_prediction_df.iloc[:, 0:int(round(pred[0]))+ 100] = 0

    sample_prediction_df.iloc[:, int(round(pred[0])+ 100):-1] = 1

    sample_prediction_df.iloc[:, -1] = 1

    sample_prediction_df.iloc[:, int(round(pred[0]) + 100)] = .95

    sample_prediction_df= sample_prediction_df.T

    sample_prediction_df = sample_prediction_df.interpolate(axis = 0, method = 'linear').T

    return sample_prediction_df
env = nflrush.make_env()

for (test_df, sample_prediction_df) in env.iter_test():

    predictions_df = make_my_predictions(test_df, sample_prediction_df)

    env.predict(predictions_df)



env.write_submission_file()