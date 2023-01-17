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
import datetime

import matplotlib.pyplot as plt

import re

from string import punctuation

import time

import tqdm
from kaggle.competitions import nflrush



# You can only call make_env() once, so don't lose it!

env = nflrush.make_env()
start_time = time.time()

original_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)

train_df = original_df

train_df.head()
# Some Rough Plots for Empirical Distributions, will skip when submit



# Each play only has one yards

play_yards_list = train_df.groupby(["GameId", "PlayId"])["Yards"].unique().values.astype(int)



fig, (hist1, hist2) = plt.subplots(1, 2)

fig.set_figwidth(15)

rough_density_hist = hist1.hist(play_yards_list, bins=100, range=(-25, 75), density=True)

title1 = hist1.set_title("pmf")

rough_cumulative_hist = hist2.hist(play_yards_list, bins=100, range=(-25, 75), density=True, cumulative=True)

title2 = hist2.set_title("cdf")
# Precomputed New Features



# From https://www.kaggle.com/prashantkikani/nfl-starter-lgb-feature-engg

train_df['DefendersInTheBox_vs_Distance'] = train_df['DefendersInTheBox'] / train_df['Distance']
# Categorical features



cat_features = []

for col in train_df.columns:

    if train_df[col].dtype =='object':

        cat_features.append((col, len(train_df[col].unique())))

        

cat_features
### StadiumType ###





# Fix Typos

def clean_StadiumType(txt):

    if pd.isna(txt):

        return np.nan

    txt = txt.lower()

    txt = ''.join([c for c in txt if c not in punctuation])

    txt = re.sub(' +', ' ', txt)

    txt = txt.strip()

    txt = txt.replace('outside', 'outdoor')

    txt = txt.replace('outdor', 'outdoor')

    txt = txt.replace('outddors', 'outdoor')

    txt = txt.replace('outdoors', 'outdoor')

    txt = txt.replace('oudoor', 'outdoor')

    txt = txt.replace('indoors', 'indoor')

    txt = txt.replace('ourdoor', 'outdoor')

    txt = txt.replace('retractable', 'rtr.')

    return txt





# We are just going to focus on the words: outdoor, indoor, closed and open

def transform_StadiumType(txt):

    if pd.isna(txt):

        return np.nan

    if 'outdoor' in txt or 'open' in txt:

        return 1

    if 'indoor' in txt or 'closed' in txt:

        return 0

    return np.nan





# Run functions

train_df["StadiumType"] = train_df["StadiumType"].apply(clean_StadiumType)

train_df['StadiumType'] = train_df['StadiumType'].apply(transform_StadiumType)
### Truf ###



#from https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/112681#latest-649087





Turf = {'Field Turf':'Artificial', 'A-Turf Titan':'Artificial', 'Grass':'Natural', 'UBU Sports Speed S5-M':'Artificial', 

        'Artificial':'Artificial', 'DD GrassMaster':'Artificial', 'Natural Grass':'Natural', 

        'UBU Speed Series-S5-M':'Artificial', 'FieldTurf':'Artificial', 'FieldTurf 360':'Artificial', 'Natural grass':'Natural', 'grass':'Natural', 

        'Natural':'Natural', 'Artifical':'Artificial', 'FieldTurf360':'Artificial', 'Naturall Grass':'Natural', 'Field turf':'Artificial', 

        'SISGrass':'Artificial', 'Twenty-Four/Seven Turf':'Artificial', 'natural grass':'Natural'} 



train_df['Turf'] = train_df['Turf'].map(Turf)



# Change to binary feature: is Natural

train_df['Turf'] = train_df['Turf'] == 'Natural'
### PossessionTeam ###



# We have some problem with the enconding of the teams such as BLT and BAL or ARZ and ARI.

# Uncomment this code to show the issue

# train_df[(train_df['PossessionTeam']!=train_df['HomeTeamAbbr']) & (train_df['PossessionTeam']!=train_df['VisitorTeamAbbr'])][['PossessionTeam', 'HomeTeamAbbr', 'VisitorTeamAbbr']]



# Fix them manually



map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}

for abb in train_df['PossessionTeam'].unique():

    map_abbr[abb] = abb



train_df['PossessionTeam'] = train_df['PossessionTeam'].map(map_abbr)

train_df['HomeTeamAbbr'] = train_df['HomeTeamAbbr'].map(map_abbr)

train_df['VisitorTeamAbbr'] = train_df['VisitorTeamAbbr'].map(map_abbr)





# New Features: HomePossesion, Field_eq_Possession, HomeField

train_df['HomePossesion'] = train_df['PossessionTeam'] == train_df['HomeTeamAbbr']

train_df['Field_eq_Possession'] = train_df['FieldPosition'] == train_df['PossessionTeam']

train_df['HomeField'] = train_df['FieldPosition'] == train_df['HomeTeamAbbr']
### OffensiveFormation ###



off_form = train_df['OffenseFormation'].unique()

train_df = pd.concat([train_df.drop(['OffenseFormation'], axis=1), pd.get_dummies(train_df['OffenseFormation'], prefix='Formation')], axis=1)



# Here the `dummy_col` is actually including df.columns and dummy columns, will be used in further function

dummy_col = train_df.columns
### GameClock ###



# Change `m:s:ms` string type to seconds

def strtoseconds(txt):

    txt = txt.split(':')

    ans = int(txt[0])*60 + int(txt[1]) + int(txt[2])/60

    return ans





train_df['GameClock'] = train_df['GameClock'].apply(strtoseconds)
### PlayerHeight ### 



# Change `feet-inch` string type to inches, 1ft = 12 in

def strtoinches(txt):

    txt = txt.split('-')

    ans = int(txt[0]) * 12 + int(txt[1])

    return ans





train_df['PlayerHeight'] = train_df['PlayerHeight'].apply(strtoinches)



# New Feature: PlayerBMI

train_df['PlayerBMI'] = 703 * (train_df['PlayerWeight']/(train_df['PlayerHeight'])**2)
### TimeHandoff & TimeSnap ###



train_df['TimeHandoff'] = train_df['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

train_df['TimeSnap'] = train_df['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))



# New Feature: TimeDelta

train_df['TimeDelta'] = train_df.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)
### PlayerBirthDate ###



train_df['PlayerBirthDate'] = train_df['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))



# New Feature: PlayerAge

seconds_in_year = 60*60*24*365.25

train_df['PlayerAge'] = train_df.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds() / seconds_in_year, axis=1)



train_df = train_df.drop(['TimeHandoff', 'TimeSnap', 'PlayerBirthDate'], axis=1)
### WindSpeed & WindDirection ###



train_df['WindSpeed'] = train_df['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)



# Replace the ones that has x-y by (x+y)/2

# And also the ones with x gusts up to y

train_df['WindSpeed'] = train_df['WindSpeed'].apply(lambda x: (int(x.split('-')[0]) + int(x.split('-')[1])) / 2 if not pd.isna(x) and '-' in x else x)

train_df['WindSpeed'] = train_df['WindSpeed'].apply(lambda x: (int(x.split()[0]) + int(x.split()[-1])) / 2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)

train_df['WindSpeed'] = train_df['WindSpeed'].apply(lambda x: float(x) if str(x).isnumeric() else -1)





# Clean WindDirection

def clean_WindDirection(txt):

    if pd.isna(txt):

        return np.nan

    txt = txt.lower()

    txt = ''.join([c for c in txt if c not in punctuation])

    txt = txt.replace('from', '')

    txt = txt.replace(' ', '')

    txt = txt.replace('north', 'n')

    txt = txt.replace('south', 's')

    txt = txt.replace('west', 'w')

    txt = txt.replace('east', 'e')

    return txt



def transform_WindDirection(txt):

    if pd.isna(txt):

        return np.nan

    if txt=='n':

        return 0

    if txt=='nne' or txt=='nen':

        return 1/8

    if txt=='ne':

        return 2/8

    if txt=='ene' or txt=='nee':

        return 3/8

    if txt=='e':

        return 4/8

    if txt=='ese' or txt=='see':

        return 5/8

    if txt=='se':

        return 6/8

    if txt=='ses' or txt=='sse':

        return 7/8

    if txt=='s':

        return 8/8

    if txt=='ssw' or txt=='sws':

        return 9/8

    if txt=='sw':

        return 10/8

    if txt=='sww' or txt=='wsw':

        return 11/8

    if txt=='w':

        return 12/8

    if txt=='wnw' or txt=='nww':

        return 13/8

    if txt=='nw':

        return 14/8

    if txt=='nwn' or txt=='nnw':

        return 15/8

    return np.nan





train_df['WindDirection'] = train_df['WindDirection'].apply(clean_WindDirection)

train_df['WindDirection'] = train_df['WindDirection'].apply(transform_WindDirection)
### PlayDirection ###



train_df['PlayDirection'] = train_df['PlayDirection'].apply(lambda x: x.strip() == 'right')
### Team ###



train_df['Team'] = train_df['Team'].apply(lambda x: x.strip()=='home')
### GameWeather ###



# TODO:

# Lower case

# N/A Indoor, N/A (Indoors) and Indoor => indoor Let's try to cluster those together.

# coudy and clouidy => cloudy

# party => partly

# sunny and clear => clear and sunny

# skies and mostly => ""



train_df['GameWeather'] = train_df['GameWeather'].str.lower()

indoor = "indoor"

train_df['GameWeather'] = train_df['GameWeather'].apply(lambda x: indoor if not pd.isna(x) and indoor in x else x)

train_df['GameWeather'] = train_df['GameWeather'].apply(lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isna(x) else x)

train_df['GameWeather'] = train_df['GameWeather'].apply(lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)

train_df['GameWeather'] = train_df['GameWeather'].apply(lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)



# TODO:

# climate controlled or indoor => 3, sunny or sun => 2, clear => 1, cloudy => -1, rain => -2, snow => -3, others => 0

# partly => multiply by 0.5



def map_weather(txt):

    ans = 1

    if pd.isna(txt):

        return 0

    if 'partly' in txt:

        ans*=0.5

    if 'climate controlled' in txt or 'indoor' in txt:

        return ans*3

    if 'sunny' in txt or 'sun' in txt:

        return ans*2

    if 'clear' in txt:

        return ans

    if 'cloudy' in txt:

        return -ans

    if 'rain' in txt or 'rainy' in txt:

        return -2*ans

    if 'snow' in txt:

        return -3*ans

    return 0



train_df['GameWeather'] = train_df['GameWeather'].apply(map_weather)
### NflId & NflIdRusher ###



# New Feature: IsRusher

train_df['IsRusher'] = train_df['NflId'] == train_df['NflIdRusher']

train_df.drop(['NflId', 'NflIdRusher'], axis=1, inplace=True)
### X & Orientation & Dir ###



train_df['X'] = train_df.apply(lambda row: row['X'] if row['PlayDirection'] else 120-row['X'], axis=1)



# From https://www.kaggle.com/scirpus/hybrid-gp-and-nn

def new_orientation(angle, play_direction):

    if play_direction == 0:

        new_angle = 360.0 - angle

        if new_angle == 360.0:

            new_angle = 0.0

        return new_angle

    else:

        return angle



train_df['Orientation'] = train_df.apply(lambda row: new_orientation(row['Orientation'], row['PlayDirection']), axis=1)

train_df['Dir'] = train_df.apply(lambda row: new_orientation(row['Dir'], row['PlayDirection']), axis=1)
### YardsLeft ###



# New Feature: YardsLeft

# Compute how many yards are left to the end-zone

train_df['YardsLeft'] = train_df.apply(lambda row: 100 - row['YardLine'] if row['HomeField'] else row['YardLine'], axis=1)

train_df['YardsLeft'] = train_df.apply(lambda row: row['YardsLeft'] if row['PlayDirection'] else 100 - row['YardsLeft'], axis=1)



# Yards<=YardsLeft and YardsLeft-100<=Yards, thus we are going to drop those wrong lines

train_df.drop(train_df.index[(train_df['YardsLeft'] < train_df['Yards']) | (train_df['YardsLeft'] - 100 > train_df['Yards'])], inplace=True)
### Drop the categorical features and fillna ###

    

train_df = train_df.sort_values(by=['PlayId', 'Team', 'IsRusher', 'JerseyNumber']).reset_index()

train_df = train_df.drop(['index', 'IsRusher', 'Team'], axis=1)

train_df = train_df.set_index(['GameId', 'PlayId'])

cat_features = []

for col in train_df.columns:

    if train_df[col].dtype =='object':

        cat_features.append(col)

train_df = train_df.drop(cat_features, axis=1)

train_df.fillna(-999, inplace=True)



clean_time = time.time()

print("Finish Data Cleaning, session time: {}s".format(clean_time - start_time))
def clean_data(df):

    df['StadiumType'] = df['StadiumType'].apply(clean_StadiumType)

    df['StadiumType'] = df['StadiumType'].apply(transform_StadiumType)

    df['DefendersInTheBox_vs_Distance'] = df['DefendersInTheBox'] / df['Distance']

    df['OffenseFormation'] = df['OffenseFormation'].apply(lambda x: x if x in off_form else np.nan)

    df = pd.concat([df.drop(['OffenseFormation'], axis=1), pd.get_dummies(df['OffenseFormation'], prefix='Formation')], axis=1)

    missing_cols = set( dummy_col ) - set( df.columns )-set('Yards')

    for c in missing_cols:

        df[c] = 0

    df = df[dummy_col]

#     df.drop(['Yards'], axis=1, inplace=True)

    df['Turf'] = df['Turf'].map(Turf)

    df['Turf'] = df['Turf'] == 'Natural'

    df['PossessionTeam'] = df['PossessionTeam'].map(map_abbr)

    df['HomeTeamAbbr'] = df['HomeTeamAbbr'].map(map_abbr)

    df['VisitorTeamAbbr'] = df['VisitorTeamAbbr'].map(map_abbr)

    df['HomePossesion'] = df['PossessionTeam'] == df['HomeTeamAbbr']

    df['Field_eq_Possession'] = df['FieldPosition'] == df['PossessionTeam']

    df['HomeField'] = df['FieldPosition'] == df['HomeTeamAbbr']

    df['GameClock'] = df['GameClock'].apply(strtoseconds)

    df['PlayerHeight'] = df['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))

    df['PlayerBMI'] = 703*(df['PlayerWeight']/(df['PlayerHeight'])**2)

    df['TimeHandoff'] = df['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

    df['TimeSnap'] = df['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

    df['TimeDelta'] = df.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)

    df['PlayerBirthDate'] = df['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))

    seconds_in_year = 60*60*24*365.25

    df['PlayerAge'] = df.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)

    df['WindSpeed'] = df['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)

    df['WindSpeed'] = df['WindSpeed'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)

    df['WindSpeed'] = df['WindSpeed'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)

    df['WindSpeed'] = df['WindSpeed'].apply(lambda x: float(x) if str(x).isnumeric() else -1)

    df['WindDirection'] = df['WindDirection'].apply(clean_WindDirection)

    df['WindDirection'] = df['WindDirection'].apply(transform_WindDirection)

    df['PlayDirection'] = df['PlayDirection'].apply(lambda x: x.strip() == 'right')

    df['Team'] = df['Team'].apply(lambda x: x.strip()=='home')

    indoor = "indoor"

    df['GameWeather'] = df['GameWeather'].apply(lambda x: indoor if not pd.isna(x) and indoor in x else x)

    df['GameWeather'] = df['GameWeather'].apply(lambda x: x.lower().replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly').replace('clear and sunny', 'sunny and clear').replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)

    df['GameWeather'] = df['GameWeather'].apply(map_weather)

    df['IsRusher'] = df['NflId'] == df['NflIdRusher']

    df['X'] = df.apply(lambda row: row['X'] if row['PlayDirection'] else 120-row['X'], axis=1)

    df['Orientation'] = df.apply(lambda row: new_orientation(row['Orientation'], row['PlayDirection']), axis=1)

    df['Dir'] = df.apply(lambda row: new_orientation(row['Dir'], row['PlayDirection']), axis=1)

    df['YardsLeft'] = df.apply(lambda row: 100-row['YardLine'] if row['HomeField'] else row['YardLine'], axis=1)

    df['YardsLeft'] = df.apply(lambda row: row['YardsLeft'] if row['PlayDirection'] else 100-row['YardsLeft'], axis=1)

    

    # Drop the categorical features and fillna

    

    df = df.sort_values(by=['PlayId', 'Team', 'IsRusher', 'JerseyNumber']).reset_index()

    df = df.drop(['TimeHandoff', 'TimeSnap', 'PlayerBirthDate', 'NflId', 'NflIdRusher', 'index', 'IsRusher', 'Team'], axis=1)

    df = df.set_index(['GameId', 'PlayId'])

    cat_features = []

    for col in df.columns:

        if df[col].dtype =='object':

            cat_features.append(col)

    df = df.drop(cat_features, axis=1)

    df.fillna(-999, inplace=True)

    return df





# Run Function

# original_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)

# clean_df = clean_data(original_df)
from sklearn.preprocessing import StandardScaler

import torch

import torch.nn as nn

import torchvision
# Split Design Matrix



play_cols = []

player_cols = []

for col in train_df.columns:

    if train_df[col][:22].std() == 0:

        play_cols.append(col)

    else:

        player_cols.append(col)



try:

    player_cols.remove("level_0")

except:

    pass
print(play_cols)

print(player_cols)
# Transform Function



# The train_df will be transformed into (play_matrix(2D), player_cube(3D), targets), also get standardized



def transform_df(df, is_train=True, col1=play_cols, col2=player_cols):

     

    if is_train:

        

        play_cols.remove('Yards')

        play_df = df[play_cols]

        player_df = df[player_cols]

        targets = df['Yards']

        

        # Transform play

        play_df = play_df[::22]

        x1 = torch.tensor(StandardScaler().fit_transform(play_df).astype(float), dtype=torch.float32)



        # Transform player -> 3D Cube

        player_df = StandardScaler().fit_transform(player_df)

        player_df = player_df.reshape(play_df.shape[0], player_df.shape[1], 22)

        x2 = torch.tensor(player_df.astype(float), dtype=torch.float32)



        # Transform targets

        index = play_df.reset_index("GameId").index



        y = torch.tensor(targets[::22].values, dtype=torch.float32)



        y_dis = torch.zeros([y.shape[0], 199])

        y_dis[torch.arange(y.shape[0]), (y + 99).long()] = 1



        return x1, x2, y_dis, index



    else:

        

        play_df = df[play_cols]

        player_df = df[player_cols]

        

        # Transform play

        play_df = play_df[::22]

        x1 = torch.tensor(StandardScaler().fit_transform(play_df).astype(float), dtype=torch.float32)



        # Transform player -> 3D Cube

        player_df = StandardScaler().fit_transform(player_df)

        player_df = player_df.reshape(play_df.shape[0], player_df.shape[1], 22)

        x2 = torch.tensor(player_df.astype(float), dtype=torch.float32)

        

        index = play_df.reset_index("GameId").index



        return x1, x2, index





# Run Function

x1, x2, y, index = transform_df(train_df)
x1.shape, x2.shape, y.shape, index.shape
# Define Customized mixture nn



"""

Neural Network Structure:



Play Data -> Dense Layer -> 

                            Mixture Matrix -> Dense Layers -> Softmax -> NN output

Player Data -> CNN Layer -> 



The NN output stands for the probabilities in yards range [-99, 99]

"""



class MixtureNN(nn.Module):

    def __init__(self):

        super(MixtureNN, self).__init__()

        

        self.x1_dense = nn.Sequential(

            nn.Linear(len(play_cols), 200),

            nn.ReLU(),

            nn.Linear(200, 100),

            nn.ReLU(),

        )

        

        #################

        # params tuning #

        # MaxPooling ?  #

        # kernel size   #

        #################

        

        self.x2_conv = nn.Sequential(

            nn.Conv1d(in_channels=len(player_cols), out_channels=22, kernel_size=5, stride=1, padding=1),

            nn.ReLU(),

            nn.Conv1d(in_channels=22, out_channels=11, kernel_size=3, stride=1, padding=1),

            nn.ReLU(),

        )

        self.x_dense = nn.Sequential(

            nn.Linear(320, 480),

            nn.BatchNorm1d(480),

#             nn.Dropout(0.05),

            nn.ReLU(),

            nn.Linear(480, 240),

            nn.BatchNorm1d(240),

#             nn.Dropout(0.05),

            nn.ReLU(),

            nn.Linear(240, 199),

            nn.Softmax(dim=1)

        )

    

    def forward(self, x1, x2):

        x1 = self.x1_dense(x1)

        x2 = self.x2_conv(x2).reshape(x1.shape[0], 220)

        x = torch.cat((x1, x2), dim=1)

#         print(x.shape, x1.shape, x2.shape)

        x = self.x_dense(x)

        return x
MixtureNN()(x1,x2).shape
"""

Loss = (pred_dis - actual_dist)**2.sum(dim=1).mean()

Similar to Cross Entrophy

"""



class CRPS(nn.Module):

    def __init__(self):

        super(CRPS, self).__init__()

        

    def forward(self, x, y):

#         y_index = (y + 99).long()

#         loss = (x**2).sum(dim=1) - 2 * x[torch.arange(y_index.shape[0]), y_index] + 1

#         loss = -torch.log(x[torch.arange(y_index.shape[0]), y_index] + 1)

        loss = ((torch.cumsum(x, dim=1) - torch.cumsum(y, dim=1))**2).mean(1)

        return loss.sum()
# V10



before_training_time = time.time()

loss_func = CRPS()





################

# split method #

################



# Define train validation split function

def random_split(val_rate=0.2, batch=512):

    total_index = np.arange(len(y))

    val_index = np.random.choice(total_index, int(len(total_index) * 0.2), replace=False)

    train_index = np.delete(total_index, val_index)



    # Generate DataLoader

    train_set = torch.utils.data.TensorDataset(x1[train_index], x2[train_index], y[train_index])

    val_set = torch.utils.data.TensorDataset(x1[val_index], x2[val_index], y[val_index])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=8)

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch, shuffle=True, num_workers=8)

    return train_loader, val_loader





# Define function to calculate average loss for all batches

def calculate_average_loss(dataloader: torch.utils.data.DataLoader):

    running_loss = 0

    for x1_batch, x2_batch, y_batch in dataloader:

        prediction = model(x1_batch, x2_batch)

        loss = loss_func(prediction, y_batch)

        running_loss += float(loss)

    return running_loss / len(dataloader.dataset)







print("======== START TRAINING......")

cv_times = 0

models_params = []

sum_best_loss = 0



while cv_times < 20:

    

    stable_step = 0 

    best_val_loss = 100

    epoch = 0

    session_time = time.time()

    

    # Define Network

    model = MixtureNN()

    model.train()

    # Define Optimizer

#     optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, weight_decay=0.01)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)

    

    print("======== Start Model {} ========".format(cv_times))

    train_loader, val_loader = random_split(val_rate=0.3, batch=512)

    

    while stable_step < 50:

        for x1_batch, x2_batch, y_batch in train_loader:



            output = model(x1_batch, x2_batch)

            loss_val = loss_func(output, y_batch)

            loss_val.backward()

            optimizer.step()



        train_ave_loss = calculate_average_loss(train_loader)

        val_ave_loss = calculate_average_loss(val_loader)



        if val_ave_loss < best_val_loss:

            best_val_loss = val_ave_loss

            best_params = model.state_dict()

            stable_step = 0

        else:

            stable_step += 1



        epoch += 1

        print("epoch: {}, train_loss: {}, val_loss: {}, best_val_loss: {}, stable_step: {}".format(

            epoch, train_ave_loss, val_ave_loss, best_val_loss, stable_step))



        # MAX epoch

        if epoch >= 500:

            break

    

    # Save model 

    print("======== Finish Model {} ========".format(cv_times))

    print("Model training time: {}s. The best evaluation loss is {}.".format(time.time() - session_time, best_val_loss))

    models_params.append(best_params)

    sum_best_loss += best_val_loss

    cv_times += 1



training_time = time.time()

print("======== END, total training time: {}s".format(training_time - before_training_time))

print("average_best_loss: {}".format(sum_best_loss / cv_times))
# V20



# before_training_time = time.time()

# loss_func = CRPS()





# ################

# # split method #

# ################



# # Define random train validation split function

# def random_split(val_rate=0.2, batch=512):

#     total_index = np.arange(len(y))

#     val_index = np.random.choice(total_index, int(len(total_index) * 0.2), replace=False)

#     train_index = np.delete(total_index, val_index)



#     # Generate DataLoader

#     train_set = torch.utils.data.TensorDataset(x1[train_index], x2[train_index], y[train_index])

#     val_set = torch.utils.data.TensorDataset(x1[val_index], x2[val_index], y[val_index])

#     train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=8)

#     val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch, shuffle=True, num_workers=8)

#     return train_loader, val_loader





# # Define time series train validation split function

# def time_series_split(total_folds=10, train_folds=4, val_folds=1, step=1, start=0, batch=512):

    

#     if start > total_folds - train_folds - val_folds:

#         raise ValueError('invalid start')

        

#     fold_len = int(len(y) / total_folds)

#     fold_index = np.arange(0, len(y), fold_len)

#     train_index = np.arange(fold_index[start], fold_index[start + train_folds])

#     val_index = np.arange(fold_index[start + train_folds], fold_index[start + train_folds + val_folds])

    

#     # Generate DataLoader

#     train_set = torch.utils.data.TensorDataset(x1[train_index], x2[train_index], y[train_index])

#     val_set = torch.utils.data.TensorDataset(x1[val_index], x2[val_index], y[val_index])

#     train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=8)

#     val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch, shuffle=True, num_workers=8)

#     return train_loader, val_loader

    

    

# # Define function to calculate average loss for all batches

# def calculate_average_loss(dataloader: torch.utils.data.DataLoader):

#     running_loss = 0

#     for x1_batch, x2_batch, y_batch in dataloader:

#         prediction = model(x1_batch, x2_batch)

#         loss = loss_func(prediction, y_batch)

#         running_loss += float(loss)

#     return running_loss / len(dataloader.dataset)





# train_time = 10

# train_result = []

# init_time = time.time()

# for i in range(train_time):

    

#     print("TRAINING TIME {}".format(i))

#     print("======== START TRAINING......")

#     cv_times = 0

#     models_params = []

#     sum_best_loss = 0



#     while cv_times < 2:



#         stable_step = 0 

#         best_val_loss = 100

#         epoch = 0

#         session_time = time.time()



#         # Define Network

#         model = MixtureNN()

#         model.train()

#         # Define Optimizer

#     #     optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, weight_decay=0.01)

#         optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)



#         print("======== Start Model {} ========".format(cv_times))

#     #     train_loader, val_loader = random_split(val_rate=0.3, batch=512)

#         train_loader, val_loader = time_series_split(total_folds=5, train_folds=3, val_folds=1, step=1, start=cv_times, batch=512)



#         while stable_step < 50:

#             for x1_batch, x2_batch, y_batch in train_loader:



#                 output = model(x1_batch, x2_batch)

#                 loss_val = loss_func(output, y_batch)

#                 loss_val.backward()

#                 optimizer.step()



#             train_ave_loss = calculate_average_loss(train_loader)

#             val_ave_loss = calculate_average_loss(val_loader)



#             if val_ave_loss < best_val_loss:

#                 best_val_loss = val_ave_loss

#                 best_params = model.state_dict()

#                 stable_step = 0

#             else:

#                 stable_step += 1



#             epoch += 1

#             print("epoch: {}, train_loss: {}, val_loss: {}, best_val_loss: {}, stable_step: {}".format(

#                 epoch, train_ave_loss, val_ave_loss, best_val_loss, stable_step))



#             # MAX epoch

#             if epoch >= 500:

#                 break



#         # Save model 

#         print("======== Finish Model {} ========".format(cv_times))

#         print("Model training time: {}s. The best evaluation loss is {}.".format(time.time() - session_time, best_val_loss))

#         models_params.append(best_params)

#         sum_best_loss += best_val_loss

#         cv_times += 1



#     training_time = time.time()

#     print("======== END, total training time: {}s".format(session_time - before_training_time))

#     print("average_best_loss: {}".format(sum_best_loss / cv_times))

#     train_result.append(models_params)

    

# print("\nTOTAL: {}s".format(time.time() - init_time))
# V10



def make_pred(df, sample, env, models_params):

    

    clean_df = df

    clean_df = clean_data(clean_df)

    x1, x2, index = transform_df(clean_df, is_train=False)

    y_preds = []

#     print(x1.shape, x2.shape)

    

    for params in models_params:

        model.load_state_dict(params)

        model.eval()

        y_preds.append(torch.cumsum(model(x1, x2), dim=1).detach().data.numpy())

        

    y_ave = np.array(y_preds).mean(0)

    yardsleft = np.array(clean_df["YardsLeft"][::22])



    for i in range(len(yardsleft)):

        y_ave[i, :yardsleft[i] - 1] = 0

        y_ave[i, yardsleft[i] + 99:] = 1

    env.predict(pd.DataFrame(data=y_ave.clip(0, 1), columns=sample.columns))

    

    return y_ave
# V20



# def make_pred(df, sample, env, train_result):

    

#     clean_df = df

#     clean_df = clean_data(clean_df)

#     x1, x2, index = transform_df(clean_df, is_train=False)

#     final_pred = []

# #     print(x1.shape, x2.shape)

    

#     for params_list in train_result:

#         y_preds = []

#         for params in params_list:

#             model.load_state_dict(params)

#             model.eval()

#             y_preds.append(torch.cumsum(model(x1, x2), dim=1).detach().data.numpy())

#         y_ave = np.array(y_preds).mean(0)

#         final_pred.append(y_ave)

    

#     final_ave = np.array(final_pred).mean(0)

#     yardsleft = np.array(clean_df["YardsLeft"][::22])



#     for i in range(len(yardsleft)):

#         final_ave[i, :yardsleft[i] - 1] = 0

#         final_ave[i, yardsleft[i] + 99:] = 1

#     env.predict(pd.DataFrame(data=final_ave.clip(0, 1), columns=sample.columns))

    

#     return final_pred
for test, sample in tqdm.tqdm(env.iter_test()):

    make_pred(test, sample, env, models_params)

env.write_submission_file()
total_time = time.time()

print('Total running time is: {}s'.format(total_time - start_time))