# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy

import lightgbm as lgb

from kaggle.competitions import nflrush

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

import re

from collections import defaultdict

from sklearn.neighbors import NearestNeighbors

import datetime

import copy

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

env = nflrush.make_env()

categories = []

labelEncoder = LabelEncoder()

rushers =[]



# open (1) or closed (0)

in_out_map = defaultdict(int,{

    "open": 1,

    "field": 1, 

    "out": 1,

    "oud": 1,

    "our": 1,

    "cloud": 1,

    "close": 0,

    "retract": 0,

    "dome": 0,

})



#map teams with diffrent visitor/home names





# map to values between 0-1

weather_map = {

    "controlled": 1,

    "indoor": 1,

    "indoors": 1,

    "indoors": 1,

    "sunny": 0.8,

    "clear": 0.6, 

    "cloudy": 0.4,

    "coudy": 0.4,

    "hazy": 0.4,

    "cool": 0.4,

    "rain": 0.2,

    "rainy": 0.2,

    "cold": 0,

    "snow": 0,

}



turf_map = {

    "Field Turf": "Artificial",

    "A-Turf Titan": "Artificial",

    "Grass": "Natural",

    "UBU Sports Seed S5-M": "Artificial",

    "Artificial": "Artificial",

    "DD GrassMaster": "Artificial",

    "Natural Grass": "Natural",

    "UBU Seed Series-S5-M": "Artificial",

    "FieldTurf": "Artificial",

    "FieldTurf 360": "Artificial",

    "Natural grass": "Natural",

    "grass": "Natural",

    "Natural": "Natural",

    "Artifical": "Artificial",

    "FieldTurf360": "Artificial",

    "Naturall Grass": "Natural",

    "Field turf": "Artificial",

    "SISGrass": "Artificial",

    "Twenty-Four/Seven Turf": "Artificial",

    "natural grass": "Natural" 

}



# Training data is in the competition dataset as usual

train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
def windSpeedToInt(windSpeed):

    x = re.findall('[0-9]*', str(windSpeed))

    if len(x) > 1 and x[0].isdigit() and x[1].isdigit():

        return (int(x[0]) + int(x[1]))/ 2

    if not x[0].isdigit():

        return np.nan

    return(x[0])



def stadiumTypeToInt(stadium):

    x = re.findall(r"[\w']+", str(stadium).lower())

    weight = 0

    if len(x) == 0:

        return np.nan

    for word in x:

        weight += in_out_map[word]

    weight = weight / len(x)

    return round(weight)



def heightToInches(height):

    return int(height.split('-')[0]) * 12 + int(height.split('-')[1])



def convertToProb(num):

    return np.array([1 if i > num + 99 else 0 for i in range(199)])

    

def crps(y_true, y_pred):

    y_pred = np.clip(np.cumsum(y_pred,axis = 1),0,1)

    return np.mean((y_pred-y_true)**2)



def gameWeatherToInt(weather):

    x = re.findall(r"[\w']+", str(weather).lower())

    valid = 0

    weight = 0

    for word in x:

        if word in weather_map:

            weight += weather_map[word]

            valid += 1

    if valid == 0:

        return np.nan

    return str(weight / valid)



def gameClockToSeconds(gameClock):

    times = str(gameClock).split(':')

    return (int((int(times[0]) * 60) + int(times[1]) * int(times[2]) / 60))



def cleanData(df):

    df = df.replace(["ARZ","BLT","CLV","HST"],["ARI","BAL","CLE","HOU"])

    # clean WindSpeed column

    df["WindSpeed"] = df["WindSpeed"].apply(windSpeedToInt).astype("float64")

    

    # convert Height column to inches

    df["PlayerHeight"] = df["PlayerHeight"].apply(heightToInches)

    

    # clean StadiumType column

    df["StadiumType"] = df["StadiumType"].apply(stadiumTypeToInt).astype("category")

    

    # clean GameWeather column

    df["GameWeather"] = df["GameWeather"].apply(gameWeatherToInt).astype("category")

    

    # team offense cleaning

    df['TeamOnOffense'] = "home"

    df.loc[df.PossessionTeam != df.HomeTeamAbbr, 'TeamOnOffense'] = "away"

    df['IsOnOffense'] = df.Team == df.TeamOnOffense

    

    # convert orientation to move in a standard direction

    df['MovingLeft'] = df.PlayDirection == "left"

    df['StdOrientation'] = df.Orientation

    df.loc[df.MovingLeft, 'StdOrientation'] = np.mod(180 + df.loc[df.MovingLeft, 'StdOrientation'], 360)

    

    # convert X and Y to move in a standard direction

    df['StdX'] = df.X

    df.loc[df.MovingLeft, 'StdX'] = 120 - df.loc[df.MovingLeft, 'X'] 

    df['StdY'] = df.Y

    df.loc[df.MovingLeft, 'StdY'] = 53.3 - df.loc[df.MovingLeft, 'Y'] 

    

    # convert direction to standard

    df['StdDir'] = df.Dir

    df.loc[df.MovingLeft, 'StdDir'] = np.mod(180 + df.loc[df.MovingLeft, 'StdDir'], 360)

    df.loc[df['Season'] == 2017, 'Orientation'] = np.mod(90 + df.loc[df['Season'] == 2017, 'Orientation'], 360) 

    

    df['StdYardline'] = 100 - df.YardLine

    df.loc[df.FieldPosition.fillna('') == df.PossessionTeam,'StdYardline'] = df.loc[df.FieldPosition.fillna('') == df.PossessionTeam,'YardLine']

    

    # turf cleaning

    df['Turf'] = df['Turf'].map(turf_map)

    

    # time cleaning

    df['GameClock_seconds'] = train_df['GameClock'].apply(gameClockToSeconds)

    df['GameClock_minutes'] = train_df['GameClock'].apply(lambda time: int(time.split(':')[0]))

    

    df['TimeHandoff'] = df['TimeHandoff'].apply(lambda time: datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%fZ"))

    df['TimeSnap'] = df['TimeSnap'].apply(lambda time: datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%fZ"))

    

    df['TimeDelta'] = df.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)

    

    df = df.drop(["X","Y","Dir","Orientation","YardLine","TimeHandoff","TimeSnap"],axis=1)

    

    return df
def pipeline(train_df):

    train_df = cleanData(train_df)

    

    train_df["WindDirection"] = train_df["WindDirection"].fillna("Unknown")

    train_df["FieldPosition"] = train_df["FieldPosition"].fillna("Unknown")

    train_df["OffenseFormation"] = train_df["OffenseFormation"].fillna("Unknown")

    train_df["Turf"] = train_df["Turf"].fillna("Unknown")



    categorical_features = train_df.select_dtypes(include=["object"]).columns

    

    global labelEncoder

    global categories

    global rushers

    if len(categories) == 0:

        for feature in categorical_features:

            train_df[feature] = feature + "__" + train_df[feature].astype("str")

            categories += train_df[feature].unique().tolist()

            categories += [feature +"__Unknown"]

        labelEncoder.fit(categories)



        for feature in categorical_features:

            train_df[feature] = labelEncoder.transform(train_df[feature])

        rushers = train_df[train_df["NflId"] == train_df["NflIdRusher"]]

        rushers = rushers.groupby(["NflId"])["Yards"].agg(AvgCarry="mean")

    else:

        for feature in categorical_features:

            train_df[feature] = feature + "__" + train_df[feature].astype("str")

            unseenLabels = ~train_df[feature].isin(categories)

            train_df.loc[unseenLabels,[feature]] = feature + "__Unknown"

            train_df[feature] = labelEncoder.transform(train_df[feature])

        

    # get distance to nearest neighbor of rusher

    train_df["NearestOffensivePlayer"] = 0

    train_df["NearestDefensivePlayer"] = 0

    for play in train_df.PlayId.unique():

        # get all rows in play

        playXY = train_df.loc[train_df["PlayId"] == play, ['StdX', 'StdY', 'NflId', 'NflIdRusher', 'PossessionTeam', 'Team', 'HomeTeamAbbr',"Position"]]

        # find rusher and their team

        rusher_team = playXY.loc[playXY['NflId'] == playXY['NflIdRusher'], ['Team']]

        rusherXY = playXY.loc[playXY['NflId'] == playXY['NflIdRusher'], ['StdX', 'StdY']]

        # find players and split by offense/defense

        playersXY = playXY.loc[playXY['NflId'] != playXY['NflIdRusher'], ['StdX', 'StdY', 'Team',"Position"]]

        playerOffenseXY = playersXY.loc[playersXY['Team'] == rusher_team.iloc[0]['Team'], ['StdX', 'StdY',"Position"]]

        playerDefenseXY = playersXY.loc[playersXY['Team'] != rusher_team.iloc[0]['Team'], ['StdX', 'StdY',"Position"]]

        # find X,Y coordinate of nearest offensive/defensive neighbor

        o_neighbours = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(playerOffenseXY.drop(["Position"],axis=1))

        d_neighbours = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(playerDefenseXY.drop(["Position"],axis=1))

        # find hypotenuse of neighbor

        o_dist, _ = o_neighbours.kneighbors(rusherXY)

        d_dist, _ = d_neighbours.kneighbors(rusherXY)

        train_df.loc[train_df['PlayId'] == play, 'NearestOffensivePlayer'] = o_dist[0][0]

        train_df.loc[train_df['PlayId'] == play, 'NearestDefensivePlayer'] = d_dist[0][0]

        

        # find x, y of likely tackle positions

        playerOLBXY = playerDefenseXY.loc[playerDefenseXY["Position"] == labelEncoder.transform(['Position__OLB'])[0], ['StdX', 'StdY']]

        playerILBXY = playerDefenseXY.loc[playerDefenseXY["Position"] == labelEncoder.transform(['Position__ILB'])[0], ['StdX', 'StdY']]

        playerDEXY = playerDefenseXY.loc[playerDefenseXY["Position"] == labelEncoder.transform(['Position__DE'])[0], ['StdX', 'StdY']]

        playerDTXY = playerDefenseXY.loc[playerDefenseXY["Position"] == labelEncoder.transform(['Position__DT'])[0], ['StdX', 'StdY']]

        playersTackleXY = pd.concat([playerOLBXY,playerILBXY,playerDEXY,playerDTXY])

        if not playersTackleXY.empty:

            tackle_neighbours = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(playersTackleXY)

            dis, _ = tackle_neighbours.kneighbors(rusherXY)

        else:

            dis = [[1000000]]

        # find minimum and add feature

        train_df.loc[train_df['NflId'] == play, 'NearestLikelyTackle'] = dis[0][0]

        

    train_df[categorical_features] = train_df[categorical_features].astype('category')

    

    train_df = train_df[train_df["NflId"] == train_df["NflIdRusher"]]

    

    train_df["hspeed"] = train_df["S"] * np.sin(np.deg2rad(train_df["StdDir"]))

    train_df["to_line"] = train_df["StdYardline"] - train_df["StdX"]

    train_df["force"] = train_df["A"] * train_df["PlayerWeight"]

    train_df = pd.merge(train_df,rushers, on="NflId",how="left")

    train_df["AvgCarry"] = train_df["AvgCarry"].fillna(0)

    

    dropCols = categorical_features.drop(["Team","OffenseFormation","OffensePersonnel","DefensePersonnel","WindDirection"])

    dropCols = dropCols.append(pd.Index(["NflIdRusher","Season","WindSpeed","GameId","PlayId","Week"]))

    #print(dropCols)

    train_df = train_df.drop(dropCols, axis=1)

    

    return train_df
def train_my_model(train_df):    

    train_df = pipeline(train_df)



    columns = list(train_df.columns)

    columns.remove("Yards")

    

    X_train = train_df[columns]

    Y_train = train_df["Yards"]

    

    params = { 'objective' : 'multiclass', 'num_classes': 199,

              'num_leaves': 19, 'feature_fraction': 0.4,

              'subsample': 0.4, 'min_child_samples': 10, 'num_threads': 5,

              'learning_rate': 0.01, 'num_iterations': 100, 'random_state': 42}

   

    bestScore = 1000

    bestModel = ""

    

    kfold = KFold(n_splits=3, shuffle=True, random_state=42)



    for train_index, test_index in kfold.split(X_train, Y_train):

        # train folds

        X_train_fold = X_train.iloc[train_index]

        Y_train_fold = Y_train.iloc[train_index]



        # test folds

        X_test_fold = X_train.iloc[test_index]

        Y_test_fold = Y_train.iloc[test_index]

        

        Y_train_fold = Y_train_fold + 99

 

        dataSet = lgb.Dataset(X_train_fold,label=Y_train_fold)

        model = lgb.train(params,dataSet)

    

        Y_pred = model.predict(X_test_fold)

       

        Y_test_fold = np.vstack(pd.Series(Y_test_fold).apply(lambda x: convertToProb(x)))

        score = crps(Y_test_fold,Y_pred)

        if score < bestScore:

            bestModel = copy.deepcopy(model)

            bestScore = score

        print(score)

    return bestModel
model = train_my_model(train_df)

lgb.plot_importance(model,importance_type="gain",max_num_features=30)
def predict(test_df,model):

    test_df = pipeline(test_df)

    return  pd.DataFrame(np.clip(np.cumsum(model.predict(test_df),axis = 1),0,1),columns=sample_prediction_df.columns)



for (test_df, sample_prediction_df) in env.iter_test():

    predictions_df = predict(test_df, model)

    env.predict(predictions_df)



env.write_submission_file()