# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from kaggle.competitions import nflrush

import altair as alt

import os

import datetime

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from lightgbm import LGBMRegressor

from sklearn.model_selection import GridSearchCV

# Any results you write to the current directory are saved as output.
#env = nflrush.make_env()
train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv')
EDA_df = train.copy()
alt.data_transformers.disable_max_rows()



alt.Chart(EDA_df[["Yards"]]).mark_bar().encode(

    x = alt.X("Yards:Q", bin=alt.Bin(maxbins=100)),

    y = alt.Y("count()")



)
pd.set_option('display.max_columns', None) 

EDA_df.describe()
EDA_df.select_dtypes('object').columns
EDA_df['PlayDirection'].value_counts()
EDA_df.StadiumType.unique()
EDA_df.OffenseFormation.value_counts()
# def StadiumType(txt):

#     if pd.isna(txt):

#         return np.nan

#     txt = txt.lower()

#     txt = re.sub(' +', ' ', txt)

#     txt = txt.strip()

#     txt = txt.replace('outside', 'outdoor')

#     txt = txt.replace('outdor', 'outdoor')

#     txt = txt.replace('outddors', 'outdoor')

#     txt = txt.replace('outdoors', 'outdoor')

#     txt = txt.replace('oudoor', 'outdoor')

#     txt = txt.replace('indoors', 'indoor')

#     txt = txt.replace('ourdoor', 'outdoor')

#     txt = txt.replace('retractable', 'rtr.')

#     return txt
def transform_StadiumType(txt):

    if pd.isna(txt):

        return np.nan

    if 'outdoor' in txt or 'open' in txt:

        return "open"

    if 'indoor' in txt or 'closed' in txt:

        return "closed"

    

    return np.nan
# from https://www.kaggle.com/prashantkikani/nfl-starter-lgb-feature-engg

def weather(x):

    x = str(x).lower()

    if 'indoor' in x:

        return  'indoor'

    elif 'cloud' in x or 'coudy' in x or 'clouidy' in x:

        return 'cloudy'

    elif 'rain' in x or 'shower' in x:

        return 'rain'

    elif 'sunny' in x or 'clear' in x:

        return 'sunny'

#     elif 'clear' in x:

#         return 'clear'

    elif 'cold' in x or 'cool' in x:

        return 'cool'

    elif 'snow' in x:

        return 'snow'

    return x
#from https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/112681#latest-649087

def Turf(df):

    

    grass_labels = ['grass', 'natural grass', 'natural', 'naturall grass']

    df['Grass'] = np.where(df.Turf.str.lower().isin(grass_labels), 1, 0)

    df.drop(columns = "Turf", inplace = True)
def player_age(df):

    

    df["Age"] = df["Season"] - df["PlayerBirthDate"].apply(lambda text: int(text[6:]))

    df.drop(columns = "PlayerBirthDate", inplace = True)

def height(df):

    df["PlayerHeight"] = df["PlayerHeight"].str.split("-").apply(lambda x: int(x[0]) + int(x[1])/12)
#Dropping Unused columns



def drop_ids(df):



        df = df.drop(columns = ["GameId", "PlayId"])
# https://www.kaggle.com/mrkmakr/neural-network-with-mae-objective-0-01381

def timedelta(df):

    

    df["TimeHandoff"] =  df["TimeHandoff"].apply(lambda time: datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%fZ"))

    df["TimeSnap"] =  df["TimeSnap"].apply(lambda time: datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%fZ"))

    df["timedelta"] = df.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)

    df.drop(columns = ["TimeHandoff", "TimeSnap"], inplace = True)

def OffensePersonnel(df):

    arr = [[int(s[0]) for s in t.split(", ")] for t in df["OffensePersonnel"]]

    df["RB"] = pd.Series([a[0] for a in arr])

    df["TE"] = pd.Series([a[1] for a in arr])

    df["WR"] = pd.Series([a[2] for a in arr])

    df.drop(columns = "OffensePersonnel", inplace = True)
def DefensePersonnel(df):

    arr = [[int(s[0]) for s in t.split(", ")] for t in df["DefensePersonnel"]]

    df["DL"] = pd.Series([a[0] for a in arr])

    df["LB"] = pd.Series([a[1] for a in arr])

    df["DB"] = pd.Series([a[2] for a in arr])

    df.drop(columns = "DefensePersonnel", inplace = True)
def fill_missing(df):

    for col in df.select_dtypes(include='object').columns:

        df[col] = df[col].fillna("missing")

    for col in df.select_dtypes(exclude = 'object').columns:

        df[col] = df[col].fillna(-99)
# https://www.kaggle.com/mrkmakr/neural-network-with-mae-objective-0-01381



def strtomins(txt):

    txt = txt.split(':')

    ans = int(txt[0])*60 + int(txt[1]) + int(txt[2])/60

    return ans
def fe(df):

    

    df["GameClock"] = df["GameClock"].apply(strtomins)

    drop_ids(df)

    fill_missing(df)

    OffensePersonnel(df)

    DefensePersonnel(df)

    timedelta(df)

    height(df)

    player_age(df)

#     df['StadiumType'] = df['StadiumType'].apply(StadiumType)

#     df['StadiumType'] = df['StadiumType'].apply(transform_StadiumType)

    df["IsRusher"] = df['NflId'] == df['NflIdRusher']

    df.drop(columns = ["NflId", "NflIdRusher"], inplace = True)

    Turf(df)

    df["GameWeather"] = df["GameWeather"].apply(weather)

    #Dropping wind speed and direction since the direction of the player is not known

    df.drop(columns = ["WindDirection", "WindSpeed"], inplace = True)
fe(train)
from sklearn.model_selection import train_test_split

X = train.drop(columns = "Yards")

y = train["Yards"]



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown='ignore')

cat_col = X.select_dtypes("object").columns.to_list()

X_train = enc.fit_transform(X_train, cat_col)

X_valid = enc.transform(X_valid)
model = LGBMRegressor(eval_metric = "rmse", learning_rate = 0.1).fit(X_train, y_train)



# n_estimators = [200, 300]

# learning_rate = [0.1, 0.2]

# param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators)

# grid_search = GridSearchCV(model, param_grid, scoring="neg_root_mean_squared_error", n_jobs=-1, cv=5).fit(X_train, y_train)
model.score(X_train, y_train)
from sklearn.metrics import mean_squared_error

pred_valid = model.predict(X_valid)

print("validation set root mean squared error {} ".format(np.sqrt(mean_squared_error(y_valid, pred_valid))))
pred_train = model.predict(X_train)

print("Training set root mean squared error {} ".format(np.sqrt(mean_squared_error(y_train, pred_train))))