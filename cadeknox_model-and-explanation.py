# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import matplotlib as mpl

import seaborn as sns

import datetime, tqdm

import math

from scipy.spatial import Voronoi, voronoi_plot_2d

import re

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

import catboost as cb

import lightgbm as lgb

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold,TimeSeriesSplit,KFold,GroupKFold

from sklearn.metrics import roc_auc_score,mean_squared_error,mean_absolute_error

import sqlite3

import xgboost as xgb

from sklearn.linear_model import LogisticRegression

import gc

from scipy.stats import pearsonr

from bayes_opt import BayesianOptimization



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



from kaggle.competitions import nflrush

env = nflrush.make_env()



# Training data is in the competition dataset as usual



trn_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False, dtype={'WindSpeed': 'object'})



# I want to be able to look at all data Im working with so change settings to show me all columns

pd.set_option('display.max_columns', 60)

trn_df

# Get a feel for the data I have and summarise how many unique players, games, formations etc



len(trn_df.NflId.unique())

len(trn_df.PlayId.unique())

len(trn_df.GameId.unique())

len(trn_df.Season.unique())

len(trn_df[trn_df.Season==2017].PlayId.unique())

len(trn_df[trn_df.Season==2018].PlayId.unique())





# Must correct some of the team abbreviations to match up



trn_df.loc[trn_df.VisitorTeamAbbr == "ARI",'VisitorTeamAbbr'] = "ARZ"

trn_df.loc[trn_df.HomeTeamAbbr == "ARI",'HomeTeamAbbr'] = "ARZ"



trn_df.loc[trn_df.VisitorTeamAbbr == "BAL",'VisitorTeamAbbr'] = "BLT"

trn_df.loc[trn_df.HomeTeamAbbr == "BAL",'HomeTeamAbbr'] = "BLT"



trn_df.loc[trn_df.VisitorTeamAbbr == "CLE",'VisitorTeamAbbr'] = "CLV"

trn_df.loc[trn_df.HomeTeamAbbr == "CLE",'HomeTeamAbbr'] = "CLV"



trn_df.loc[trn_df.VisitorTeamAbbr == "HOU",'VisitorTeamAbbr'] = "HST"

trn_df.loc[trn_df.HomeTeamAbbr == "HOU",'HomeTeamAbbr'] = "HST"



trn_df
# Want to use just rusher info here



trn_df['is_run'] = trn_df.NflId == trn_df.NflIdRusher

trn_sin = trn_df[trn_df.is_run==True]
# Change the format of time so it takes into account quarters



def transform_time_quarter(str1):

    return int(str1[:2])*60 + int(str1[3:5])

def transform_time_all(str1,quarter):

    if quarter<=4:

        return 15*60 - (int(str1[:2])*60 + int(str1[3:5])) + (quarter-1)*15*60

    if quarter ==5:

        return 10*60 - (int(str1[:2])*60 + int(str1[3:5])) + (quarter-1)*15*60

trn_sin['time_quarter'] = trn_sin.GameClock.map(lambda x:transform_time_quarter(x))

trn_sin['time_end'] = trn_sin.apply(lambda x:transform_time_all(x.loc['GameClock'],x.loc['Quarter']),axis=1)
# Formatting handoff and snap time time to then adjust for how long handoff is from snap, determines different type of plays



trn_sin['TimeHandoff'] = trn_sin['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

trn_sin['TimeSnap'] = trn_sin['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

trn_sin['handoff_snap_diff'] = (trn_sin['TimeHandoff'] - trn_sin['TimeSnap']).map(lambda x:x.seconds)
# Now that I've created the time diff I can drop time snap and handoff, also gameclock and a few of the ids that arent useful



remove_features = ['GameId','PlayId','DisplayName','GameClock','TimeHandoff','TimeSnap']



trn_sin['date_game'] = trn_sin.GameId.map(lambda x:pd.to_datetime(str(x)[:8]))

trn_sin['runner_age'] = (trn_sin.date_game.map(pd.to_datetime) - trn_sin.PlayerBirthDate.map(pd.to_datetime)).map(lambda x:x.days)/365

remove_features.append('HomeTeamAbbr')

remove_features.append('VisitorTeamAbbr')

remove_features.append('PlayerBirthDate')

remove_features.append('is_run')

def transform_height(te):

    return (int(te.split('-')[0])*12 + int(te.split('-')[1]))*2.54/100

trn_sin['runner_height'] = trn_sin.PlayerHeight.map(transform_height)

remove_features.append('PossessionTeam')

remove_features.append('FieldPosition')

remove_features.append('PlayerHeight')

remove_features.append('NflIdRusher')

remove_features.append('date_game')

trn_sin['own_field'] = (trn_sin['FieldPosition'] == trn_sin['PossessionTeam']).astype(int)

dist_to_end_trn = trn_sin.apply(lambda x:(100 - x.loc['YardLine']) if x.loc['own_field']==1 else x.loc['YardLine'],axis=1)

remove_features.append('own_field')



# to drop all I have to just use what i created earlier and axis = 1 for columns



trn_sin.drop(remove_features,axis=1,inplace=True)
trn_sin.fillna(-999,inplace=True)
# Split the data sets here which will be important for my regression to have yards seperated from rest



y_trn_df = trn_sin.Yards

X_trn_df = trn_sin.drop(['Yards'],axis=1)

for f in X_trn_df.columns:

    if X_trn_df[f].dtype=='object': 

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(X_trn_df[f])+[-999])

        X_trn_df[f] = lbl.transform(list(X_trn_df[f]))
# Have to create the submission csv which will end up giving me a CRPS for the data for each play



def get_cdf_df(yards_array):

    pdf, edges = np.histogram(yards_array, bins=199,

                 range=(-99,100), density=True)

    cdf = pdf.cumsum().clip(0, 1)

    cdf_df = pd.DataFrame(data=cdf.reshape(-1, 1).T, 

                            columns=['Yards'+str(i) for i in range(-99,100)])

    return cdf_df

cdf = get_cdf_df(y_trn_df).values.reshape(-1,)



def get_score(y_pred,cdf,w,dist_to_end):

    y_pred = int(y_pred)

    if y_pred ==w:

        y_pred_array = cdf.copy()

    elif y_pred - w >0:

        y_pred_array = np.zeros(199)

        y_pred_array[(y_pred-w):] = cdf[:(-(y_pred-w))].copy()

    elif w - y_pred >0:

        y_pred_array = np.ones(199)

        y_pred_array[:(y_pred-w)] = cdf[(w-y_pred):].copy()

    y_pred_array[-1]=1

    y_pred_array[(dist_to_end+99):]=1

    return y_pred_array    



def get_score_1(y_pred,y_true,cdf,w,dist_to_end):

    y_pred = int(y_pred)

    if y_pred ==w:

        y_pred_array = cdf.copy()

    elif y_pred - w >0:

        y_pred_array = np.zeros(199)

        y_pred_array[(y_pred-w):] = cdf[:(-(y_pred-w))].copy()

    elif w - y_pred >0:

        y_pred_array = np.ones(199)

        y_pred_array[:(y_pred-w)] = cdf[(w-y_pred):].copy()

    y_pred_array[-1]=1

    y_pred_array[(dist_to_end+99):]=1

    y_true_array = np.zeros(199)

    y_true_array[(y_true+99):]=1

    return np.mean((y_pred_array - y_true_array)**2)



# I want to calculate the CRPS for my csv and return it for every row

def CRPS(y_preds,y_trues,w,cdf,dist_to_ends):

    if len(y_preds) != len(y_trues):

        print('length does not match')

        return None

    n = len(y_preds)

    tmp = []

    for a,b,c in zip(y_preds, y_trues,dist_to_ends):

        tmp.append(get_score_1(a,b,cdf,w,c))

    return np.mean(tmp)
# Creating a LightGBM, Light Gradient Boosting Machine, seems like the best way about. Handles large amount of data fast



kf=KFold(n_splits = 5)

resu1 = 0

impor1 = 0

resu2_cprs = 0

resu3_mae=0

y_pred = 0

stack_trn = np.zeros([X_trn_df.shape[0],])

models = []

for trn_index, test_index in kf.split(X_trn_df, y_trn_df):

    X_trn2= X_trn_df.iloc[trn_index,:]

    y_trn2= y_trn_df.iloc[trn_index]

    X_test2= X_trn_df.iloc[test_index,:]

    y_test2= y_trn_df.iloc[test_index]

    clf = lgb.LGBMRegressor(n_estimators=10000, random_state=47,learning_rate=0.005,importance_type = 'gain',

                     n_jobs = -1,metric='mae')

    clf.fit(X_trn2,y_trn2,eval_set = [(X_trn2,y_trn2),(X_test2,y_test2)],early_stopping_rounds=200,verbose=50)

    models.append(clf)

    temp_predict = clf.predict(X_test2)

    stack_trn[test_index] = temp_predict

    #y_pred += clf.predict(X_test2)/5

    mse = mean_squared_error(y_test2, temp_predict)

    crps = CRPS(temp_predict,y_test2,4,cdf,dist_to_end_trn.iloc[test_index])

    mae = mean_absolute_error(y_test2, temp_predict)

    print(crps)

    

# Many people used LightGBM or Neural Networks/ Random Forest Models, LightGBM I was told was the least amount of machine learning so why I chose it

    

    resu1 += mse/5

    resu2_cprs += crps/5

    resu3_mae += mae/5 

    impor1 += clf.feature_importances_/5

    gc.collect()

print('mean mse:',resu1)

print('oof mse:',mean_squared_error(y_trn_df,stack_trn))

print('mean mae:',resu3_mae)

print('oof mae:',mean_absolute_error(y_trn_df,stack_trn))

print('mean cprs:',resu2_cprs)

print('oof cprs:',CRPS(stack_trn,y_trn_df,4,cdf,dist_to_end_trn))
def transform_test(test):

    test.loc[test.VisitorTeamAbbr == "ARI",'VisitorTeamAbbr'] = "ARZ"

    test.loc[test.HomeTeamAbbr == "ARI",'HomeTeamAbbr'] = "ARZ"



    test.loc[test.VisitorTeamAbbr == "BAL",'VisitorTeamAbbr'] = "BLT"

    test.loc[test.HomeTeamAbbr == "BAL",'HomeTeamAbbr'] = "BLT"



    test.loc[test.VisitorTeamAbbr == "CLE",'VisitorTeamAbbr'] = "CLV"

    test.loc[test.HomeTeamAbbr == "CLE",'HomeTeamAbbr'] = "CLV"



    test.loc[test.VisitorTeamAbbr == "HOU",'VisitorTeamAbbr'] = "HST"

    test.loc[test.HomeTeamAbbr == "HOU",'HomeTeamAbbr'] = "HST"

    test['is_run'] = test.NflId == test.NflIdRusher

    test_sin = test[test.is_run==True]

    test_sin['time_quarter'] = test_sin.GameClock.map(lambda x:transform_time_quarter(x))

    test_sin['time_end'] = test_sin.apply(lambda x:transform_time_all(x.loc['GameClock'],x.loc['Quarter']),axis=1)

    test_sin['TimeHandoff'] = test_sin['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

    test_sin['TimeSnap'] = test_sin['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

    test_sin['handoff_snap_diff'] = (test_sin['TimeHandoff'] - test_sin['TimeSnap']).map(lambda x:x.seconds)

    test_sin['date_game'] = test_sin.GameId.map(lambda x:pd.to_datetime(str(x)[:8]))

    test_sin['runner_age'] = (test_sin.date_game.map(pd.to_datetime) - test_sin.PlayerBirthDate.map(pd.to_datetime)).map(lambda x:x.days)/365

    test_sin['runner_height'] = test_sin.PlayerHeight.map(transform_height)

    return test_sin.drop(remove_features,axis=1)
# Finally run my prediction



for (test_df, sample_prediction_df) in env.iter_test():

    test_df['own_field'] = (test_df['FieldPosition'] == test_df['PossessionTeam']).astype(int)

    dist_to_end_test = test_df.apply(lambda x:(100 - x.loc['YardLine']) if x.loc['own_field']==1 else x.loc['YardLine'],axis=1)

    X_test = transform_test(test_df)

    X_test.fillna(-999,inplace=True)

    for f in X_test.columns:

        if X_test[f].dtype=='object':

            X_test[f] = X_test[f].map(lambda x:x if x in set(X_trn_df[f]) else -999)

    for f in X_test.columns:

        if X_test[f].dtype=='object': 

            lbl = preprocessing.LabelEncoder()

            lbl.fit(list(X_trn_df[f])+[-999])

            X_test[f] = lbl.transform(list(X_test[f])) 

    pred_value = 0

    for model in models:

        pred_value += model.predict(X_test)[0]/5

    pred_data = list(get_score(pred_value,cdf,4,dist_to_end_test.values[0]))

    pred_data = np.array(pred_data).reshape(1,199)

    pred_target = pd.DataFrame(index = sample_prediction_df.index, \

                               columns = sample_prediction_df.columns, \

                                #data = np.array(pred_data))

                                data = pred_data)

    env.predict(pred_target)

env.write_submission_file()
# Creating a visual diagram of plays with the runningbacks direction



trn_df['ToLeft'] = trn_df.PlayDirection == "left"

trn_df['IsBallCarrier'] = trn_df.NflId == trn_df.NflIdRusher

trn_df['Dir_rad'] = np.mod(90 - trn_df.Dir, 360) * math.pi/180.0



# First creating the football field figure



def create_football_field(linenumbers=True,

                          endzones=True,

                          highlight_line=False,

                          highlight_line_number=50,

                          highlighted_name='Line of Scrimmage',

                          fifty_is_los=False,

                          figsize=(12*2, 6.33*2)):

    # creating shape

    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,

                             edgecolor='r', facecolor='darkgreen', zorder=0,  alpha=0.5)



    fig, ax = plt.subplots(1, figsize=figsize)

    ax.add_patch(rect)



    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,

              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],

             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,

              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],

             color='white')

    if fifty_is_los:

        plt.plot([60, 60], [0, 53.3], color='gold')

        plt.text(62, 50, '<- Player Yardline at Snap', color='gold')

    # Endzones

    if endzones:

        ez1 = patches.Rectangle((0, 0), 10, 53.3,

                                linewidth=0.1,

                                edgecolor='r',

                                facecolor='blue',

                                alpha=0.2,

                                zorder=0)

        ez2 = patches.Rectangle((110, 0), 120, 53.3,

                                linewidth=0.1,

                                edgecolor='r',

                                facecolor='blue',

                                alpha=0.2,

                                zorder=0)

        ax.add_patch(ez1)

        ax.add_patch(ez2)

    plt.xlim(0, 120)

    plt.ylim(-5, 58.3)

    plt.axis('off')

    if linenumbers:

        for x in range(20, 110, 10):

            numb = x

            if x > 50:

                numb = 120 - x

            plt.text(x, 5, str(numb - 10),

                     horizontalalignment='center',

                     fontsize=20,  # fontname='Arial',

                     color='white')

            plt.text(x - 0.95, 53.3 - 5, str(numb - 10),

                     horizontalalignment='center',

                     fontsize=20,  # fontname='Arial',

                     color='white', rotation=180)

    if endzones:

        hash_range = range(11, 110)

    else:

        hash_range = range(1, 120)



    for x in hash_range:

        ax.plot([x, x], [0.4, 0.7], color='white')

        ax.plot([x, x], [53.0, 52.5], color='white')

        ax.plot([x, x], [22.91, 23.57], color='white')

        ax.plot([x, x], [29.73, 30.39], color='white')



    if highlight_line:

        hl = highlight_line_number + 10

        plt.plot([hl, hl], [0, 53.3], color='yellow')

        plt.text(hl + 2, 50, '<- {}'.format(highlighted_name),

                 color='yellow')

    return fig, ax

create_football_field()



def get_dx_dy(radian_angle, dist):

    dx = dist * math.cos(radian_angle)

    dy = dist * math.sin(radian_angle)

    return dx, dy



# Create a show_play function , to do this we take the scatter of players then want to consider x and y locations



def show_play(play_id, trn_df=trn_df):

    df = trn_df[trn_df.PlayId == play_id]

    fig, ax = create_football_field()

    ax.scatter(df.X, df.Y, cmap='rainbow', c=~(df.Team == 'home'), s=100)

    rusher_row = df[df.NflIdRusher == df.NflId]

    ax.scatter(rusher_row.X, rusher_row.Y, color='black')

    yards_covered = rusher_row["Yards"].values[0]

    x = rusher_row["X"].values[0]

    y = rusher_row["Y"].values[0]

    rusher_dir = rusher_row["Dir_rad"].values[0]

    rusher_speed = rusher_row["S"].values[0]

    dx, dy = get_dx_dy(rusher_dir, rusher_speed)



# We use labeling with plt to give us a better description and almost an if else statement for if its right or left

    

    ax.arrow(x, y, dx, dy, length_includes_head=True, width=0.3, color='black')

    left = 'left' if df.ToLeft.sum() > 0 else 'right'

    plt.title(f'Play # {play_id} moving to {left}, yard distance is {yards_covered}', fontsize=20)

    plt.legend()

    plt.show()

    
# We can see plays visually now with movement, and details at the top

show_play(20181230154157)