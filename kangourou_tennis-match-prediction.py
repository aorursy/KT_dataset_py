import pandas as pd

from urllib.request import urlopen  

import os.path as osp

import os

import logging

import zipfile

from glob import glob

logging.getLogger().setLevel('INFO')



import pickle

import random

import numpy as np

import datetime

import time

import xgboost as xgb



from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report



import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sn

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
def dump(obj, name):

    pickle.dump(obj, open(name+'.p', "wb")) 



def load(name):

    obj = pickle.load(open(name+".p", "rb")) 

    return obj
ATP_DIR = "../input/"

ATP_FILES = sorted(glob("%s/*.xls*" % ATP_DIR))

df_atp = pd.concat([pd.read_excel(f) for f in ATP_FILES], sort=False, ignore_index=True)
# look at the size of the data

df_atp.head()
# numbers of rows and columns

df_atp.shape
# look at what are the columns

df_atp.columns
# look at the type of the data in each columns

df_atp.info()
df_atp.describe()
# columns of which the data type is 'object' (string)

print([c for c in df_atp.columns if df_atp[str(c)].dtypes == 'object'])
# check and find out the erros

for c in ['EXW', 'Lsets', 'WRank', 'LRank', 'L2', 'W2', 'L3','W3']:

    print([x for x in df_atp[str(c)] if type(x)==str])
# make a copy of the original data

df_atp2 = df_atp.copy()



# cleaning the data

df_atp2["EXW"] = df_atp2["EXW"].replace("2.,3", 2.3)

df_atp2["Lsets"] = df_atp2["Lsets"].replace("`1", 1)



df_atp2["LRank"] = df_atp2["LRank"].replace("NR", np.nan).astype('float')

df_atp2["WRank"] = df_atp2["WRank"].replace("NR", np.nan).astype('float')



df_atp2["L2"] = df_atp2["L2"].replace(' ', np.nan)

df_atp2["W2"] = df_atp2["W2"].replace(' ', np.nan)

df_atp2["L3"] = df_atp2["L3"].replace(' ', np.nan)

df_atp2["W3"] = df_atp2["W3"].replace(' ', np.nan)
df_atp2.describe()
# checking out missing values

df_atp2.isnull().sum().sort_values(ascending=False)
# columns without missing values

col_0miss = [i for i, x in df_atp2.isnull().sum().iteritems() if x == 0]

print(col_0miss)
# columns with missing values

col_miss = [x for x in df_atp2.columns if x not in col_0miss]

print(col_miss)
# check the missing values by year

df_atp2['year'] = df_atp2.Date.apply(lambda x: x.year)
col_odds = ['AvgL', 'AvgW', 'B&WL', 'B&WW', 'B365L', 'B365W', 'CBL', 'CBW', 'EXL', 'EXW',

             'GBL', 'GBW', 'IWL', 'IWW', 'LBL', 'LBW', 'MaxL', 'MaxW', 'PSL', 'PSW',

             'SBL', 'SBW', 'SJL', 'SJW', 'UBL', 'UBW']

df_atp2[col_odds + ['year']].groupby('year').count()
df_atp3 = df_atp.copy()

# replace NR by the largest ranking value

df_atp3['WRank'] = df_atp3.WRank.replace('NR', df_atp3.WRank.replace('NR', np.nan).max())

df_atp3['LRank'] = df_atp3.LRank.replace('NR', df_atp3.LRank.replace('NR', np.nan).max())



# select a subset of data

beg = datetime.datetime(2010,1,1)

end = datetime.datetime(2019,1,1)

indices = df_atp3[(df_atp3.Date>=beg)&(df_atp3.Date<=end)].index

test = df_atp3.iloc[indices,:]



# classical ATP ranking

atp_rank = 100*(test.WRank<test.LRank).sum()/len(indices)



# ATP Entry points

atp_pts = 100*(test.WPts>test.LPts).sum()/len(indices)



# Bookmakers

book_pi = 100*(test.PSW<test.PSL).sum()/len(indices)

book_365 = 100*(test.B365W<test.B365L).sum()/len(indices)





# Plot



labels = ["ATP entry ranking", "ATP entry points", "Pinnacle", "Bet365"]

values = [atp_rank, atp_pts, book_pi, book_365]

y_pos = np.arange(len(labels))



fig = plt.figure(figsize=(13,9))

with plt.style.context('ggplot'):

    plt.barh(y_pos, values)



plt.yticks(y_pos, labels, fontsize=16)

plt.xticks(fontsize=14)

plt.xlabel('% of matches correctly predicted', fontsize=16)

plt.title("Prediction of all matches since "+beg.strftime("%Y-%m-%d")+" ("+str(len(indices))+" matches)",

         fontsize=20)

plt.xlim([65,70])

plt.tight_layout()

plt.show()
def past_victories(df, player):

    """

    For each match, return the percentage of victories of the player in the past.

    """



    vic_pers = []

    for i in df[1::].index:

        name = df.loc[i, player]

        df_past = df.iloc[0:i]

        vic_per = df_past[df_past.Winner == name].count().ATP/len(df_past)*100

        vic_pers.append(vic_per)



    vic_pers = [0.0] + vic_pers



    return vic_pers
# Feature of past victories percentage for winner and loser 

win_pers = past_victories(df_atp2, player='Winner')

los_pers = past_victories(df_atp2, player='Loser')

dic_pers = {"vicW": win_pers, "vicL": los_pers}

df_vic = pd.DataFrame(data=dic_pers, columns=dic_pers.keys())
def historical_mean_score(df, col_pair, num_days):

    """

    For each match, return the mean value of past x days (num_days) for column 

    in question of each player. For examle, if col_pair is ('WRank', 'LRank'), and

    num_days is 180, this function returns a dataframe with two columns (WMRank, LMRank)

    of which each row indicates the mean values of ATP Rank of the player (Winner) 

    in the past 180 days and that of the player (Loser).

    """



    col_w, col_l = col_pair

    if col_w.startswith('W'):

        colname_w = 'WM'+col_w[1::]

    elif col_w.endswith('W'):

        colname_w = 'WM'+col_w[0:-1]

    if col_l.startswith('L'):

        colname_l = 'LM'+col_l[1::]

    elif col_l.endswith('L'):

        colname_l = 'LM'+col_l[0:-1]



    mscoresW = []

    mscoresL = []

    for i, row in df.iterrows():

        winner = row.Winner

        loser = row.Loser

        date_cur = row.Date

        date_hist = date_cur - datetime.timedelta(days=num_days)

        df_hist = df.iloc[df[(df.Date>=date_hist)&(df.Date<date_cur)].index, :]

        mscoreW = (df_hist[df_hist.Winner == winner][str(col_w)].mean() + 

                  df_hist[df_hist.Loser == winner][str(col_l)].mean())/2

        mscoreL = (df_hist[df_hist.Winner == loser][str(col_w)].mean() + 

                  df_hist[df_hist.Loser == loser][str(col_l)].mean())/2

        mscoresW.append(mscoreW)

        mscoresL.append(mscoreL)



    dict_mscore = {colname_w: mscoresW, colname_l: mscoresL}

    df_mscore = pd.DataFrame(data=dict_mscore, columns=dict_mscore.keys())



    return df_mscore 
# create new features with recent historical data

num_days = 360 # historical data of 180 days before the match

df_mrank = historical_mean_score(df_atp2, ('WRank', 'LRank'), num_days=num_days)

df_mpts = historical_mean_score(df_atp2, ('WPts', 'LPts'), num_days=num_days)

df_msets = historical_mean_score(df_atp2, ('Wsets', 'Lsets'), num_days=num_days)

df_mPS = historical_mean_score(df_atp2, ('PSW', 'PSL'), num_days=num_days)

df_mB365 = historical_mean_score(df_atp2, ('B365W', 'B365L'), num_days=num_days)



# concat recent score features



df_mscores = pd.concat([df_mrank, df_mpts, df_msets, df_mPS, df_mB365],1)
def cal_diff(df, col_pairs):

    """

    Return the numerical difference between the two columns in col_pairs, 

    and store the values in a dataframe.

    """



    dic = {}

    for col_pair in col_pairs:

        col_win, col_los = col_pair



    if col_win.endswith("W"):

        col_new = col_win[0:-1] + "_diff"

    else:

        col_new = col_win[1::] + "_diff"



    dic[col_new] = abs(df[str(col_win)] - df[str(col_los)])

    df_diff = pd.DataFrame(data=dic, columns=dic.keys())



    return df_diff
cols_win = ['B365W', 'PSW',  'WPts', 'WRank']

cols_los = ['B365L', 'PSL',  'LPts', 'LRank'] 

col_pairs = list(zip(cols_win, cols_los))



odds_diff = cal_diff(df_atp2, col_pairs)

vic_diff = cal_diff(df_vic, [('vicW', 'vicL')])



df_diff = pd.concat([odds_diff, vic_diff], 1)
def randomize_data(df, col_pairs):



    """

    shuffle the original data, so that we have two balanced class, of which

    50% of label "1" and 50% of label "0".

    """

    # random index for player1

    idx1 = random.sample(range(1, len(df)), len(df)//2)

    # index for player2, the rest of the index after substracing the index for player1

    idx2 = list(set(np.array(range(0, len(df)))) - set(idx1))



    cols = []

    col_names = []

    for x in col_pairs:

        col_win, col_los = x

        col1 = list(df[str(col_win)][idx1].append(df[str(col_los)][idx2])

        .sort_index())

        col2 = list(df[str(col_los)][idx1].append(df[str(col_win)][idx2])

        .sort_index())

        cols.append([col1, col2])

    # rename the columns

    if col_win == "Winner":

        col1_name = "player1"

    elif col_win.endswith("W"):

        col1_name = col_win[0:-1] + "_1"

    elif col_win.startswith("W"):

        col1_name = col_win[1::] + "_1" 



    if col_los == "Loser":

        col2_name = "player2"

    elif col_los.endswith('L'):

        col2_name = col_los[0:-1] + "_2"

    elif col_los.startswith('L'):

        col2_name = col_los[1::] + "_2"



    col_names.append([col1_name, col2_name])



    player_data = [v for l in cols for v in l]

    columns = [v for c in col_names for v in c]

    player_data = pd.DataFrame(list(map(list, zip(*player_data))), columns=columns)



    return player_data, idx1
# data to 'shuffle'

player_data = pd.concat([df_atp2[cols_win + cols_los + ['Winner', 'Loser']], 

                         df_vic, df_mscores],1)

col_pairs = col_pairs + [('Winner', 'Loser'), ('vicW', 'vicL'), ('WMRank', 'LMRank'),

                         ('WMPts', 'LMPts'), ('WMsets', 'LMsets'), ('WMPS', 'LMPS'), 

                         ('WMB365', 'LMB365')]

# randomize the winner, loser data to player1, player2 data

player_data, idx1 = randomize_data(player_data, col_pairs)
def categorical_features_encoding(cat_features):

    """

    Categorical features encoding.

    Simple one-hot encoding.

    """



    ohe = OneHotEncoder()

    cat_features_encoded = ohe.fit_transform(cat_features)

    columns = ohe.get_feature_names(list(cat_features.columns))

    cat_features = pd.DataFrame(cat_features_encoded.todense(), columns=columns)



    return cat_features
def features_players_encoding(data):

    """

    Encoding of the players . 

    The players are not encoded like the other categorical features because for 

    each match we encode both players at the same time (we put a 1 in each row 

    corresponding to the players playing the match for each match).

    """

    player1 = data.player1

    player2 = data.player2

    le = LabelEncoder()

    le.fit(list(player1)+list(player2))

    player1 = le.transform(player1)

    player2 = le.transform(player2)

    encod = np.zeros([len(data), len(le.classes_)])

    for i in range(len(data)):

        encod[i,player1[i]] += 1

    for i in range(len(data)):

        encod[i,player2[i]] += 1

    columns = ["player_"+el for el in le.classes_]

    players_encoded = pd.DataFrame(encod, columns=columns)

    

    return players_encoded
def num_features_transform(features_numerical):

    """

    Handling missing values and then standardization.

    """



    features_numerical = features_numerical.fillna(0)

    scaler = StandardScaler()

    features_numerical_scaled = scaler.fit_transform(features_numerical)

    columns = features_numerical.columns

    features_numerical = pd.DataFrame(features_numerical_scaled, columns=columns)



    return features_numerical
# categorical features except the players

col_cat = ['Best of', 'Court', 'Round', 'Series', 'Surface', 'Tournament']

df_cat = df_atp2[col_cat]

# onehot encode categorical features

features_categorical_encoded = categorical_features_encoding(df_cat)

# onehot encode players

players_encoded = features_players_encoding(player_data)

# concat all the categorical features

features_onehot = pd.concat([features_categorical_encoded, players_encoded],1)



# numerical features

features_numerical = pd.concat([player_data.drop(['player1', 'player2'],axis=1), df_diff],1)
# concat all the features (categorical and numerical)

X = pd.concat([features_onehot, features_numerical], 1)



# create "label" data, which is what we want to predict

y = pd.Series(np.ones(len(df_atp)), name='label')

y.iloc[idx1] = 0 # if player1 wins, the label is 0
def get_indices(df, begin_date, end_date):

    """

    Calculate the index of observations between beginning date and ending date.

    """



    indices = df[(df.Date >= begin_date) & (df.Date < end_date)].index



    return indices



def crop_data(data, indices):

    """

    Return the data for corresponding indices.

    """



    if data.ndim > 1:

        cropped_data = data.iloc[indices,:].reset_index(drop=True)

    else:

        cropped_data = data.iloc[indices].reset_index(drop=True)



    return cropped_data
# train set, data from 2010 to 2017 

#begin_train = df_atp2.Date.iloc[0]

begin_train = datetime.datetime(2010,1,1) 

end_train = datetime.datetime(2017,1,1) 

indices_train = get_indices(df_atp2, begin_train, end_train)



X_train = crop_data(X, indices_train)

y_train = crop_data(y, indices_train)



# test set, data from 2017 to 2018

begin_test = datetime.datetime(2017,1,1)

end_test = datetime.datetime(2018,1,1)

indices_test = get_indices(df_atp2, begin_test, end_test)



X_test = crop_data(X, indices_test)

y_test = crop_data(y, indices_test)
# XGB classifier with default parameters

t = time.time()

xgb_clf = xgb.XGBClassifier()

xgb_clf.fit(X_train, y_train)

print('Elapsed: %s' % (time.time() - t))
# use the model to make predictions with the test data

y_pred = xgb_clf.predict(X_test)
# check model performance

count_misclassified = (y_test != y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}%'.format(100*accuracy))



print(classification_report(y_test, y_pred))
score = 'accuracy'

t = time.time()





print("# Tuning hyper-parameters for %s" % score)

print()



xgb_gs = xgb.XGBClassifier()

param_grid = {'max_depth': [2, 3], 

              'learning_rate': [0.1, 0.3], 

              'n_estimators': [50, 100]}

tscv = TimeSeriesSplit(n_splits=3).split(X_train)

clf = GridSearchCV(xgb_gs, param_grid, scoring=score, cv=tscv)

clf.fit(X_train, y_train)



print("Best parameters set found on development set:")

print()

print(clf.best_params_)

print()

print("Grid scores on development set:")

print()

means = clf.cv_results_['mean_test_score']

stds = clf.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, clf.cv_results_['params']):

    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

print()



print("Detailed classification report:")

print()

print("The model is trained on the full development set.")

print("The scores are computed on the full evaluation set.")

print()

y_true, y_pred = y_test, clf.predict(X_test)

print(classification_report(y_true, y_pred))

print()



print('Elapsed: %s' % (time.time() - t))
# save the best parameters

clf_bestP = clf.best_params_
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=0)

dtrain=xgb.DMatrix(X_train, label=y_train)

dval=xgb.DMatrix(X_val, label=y_val)



eval_set = [(dtrain, 'train'), (dval, 'val')]

progress = {}



params = {'eval_metric':['error', 'auc', 'logloss'], 'objective':'binary:logistic'}

params.update(clf_bestP)



t = time.time()



model=xgb.train(params, dtrain, num_boost_round=params['n_estimators'], 

                evals=eval_set, evals_result=progress,

                verbose_eval=False, early_stopping_rounds=7)



print('Elapsed: %s' % (time.time() - t))
train_loss = progress['train']['logloss']

val_loss = progress['val']['logloss']



plt.figure(figsize=(9,6))

plt.plot(np.array(range(1,len(train_loss)+1)), train_loss, label='train_loss')

plt.plot(np.array(range(1,len(val_loss)+1)), val_loss, label='val_loss')

plt.legend()

plt.xlabel('number of iteration')

plt.ylabel('logloss')

#plt.ylim([0,1])





train_error = progress['train']['error']

val_error = progress['val']['error']



plt.figure(figsize=(9,6))

plt.plot(np.array(range(1,len(train_loss)+1)), train_error, label='train_error')

plt.plot(np.array(range(1,len(val_loss)+1)), val_error, label='val_error')

plt.legend()

plt.xlabel('number of iteration')

plt.ylabel('error')

#plt.ylim([0,1])





train_auc = progress['train']['auc']

val_auc = progress['val']['auc']



plt.figure(figsize=(9,6))

plt.plot(np.array(range(1,len(train_loss)+1)), train_auc, label='train_auc')

plt.plot(np.array(range(1,len(val_loss)+1)), val_auc, label='val_auc')

plt.legend()

plt.xlabel('number of iteration')

plt.ylabel('auc')

#plt.ylim([0,1])



plt.show()



print("Best iteration number: %d" % (len(train_loss)+1))
# The probability given by the model to each outcome of each match :

pred_test = model.predict(xgb.DMatrix(X_test)) 

pred_test = pred_test  > 0.5  

pred_test = pred_test.astype(int)



print(classification_report(y_test, pred_test))

print('Accuracy: {:.2f}%'.format(100*accuracy_score(y_test, pred_test)))
cm = confusion_matrix(y_test, pred_test)

df_cm = pd.DataFrame(cm, index = [1, 0], columns = [1, 0])



# plot

fig = plt.figure(figsize=(9, 6))

sn.set(font_scale=1.4) # label size

sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt="d", cmap="YlGnBu") # font size

plt.xlabel('actual')

plt.ylabel('prediction')

plt.title('Confusion Matrix')



#fig.savefig(fig_dir+'cm.png')

plt.show()
# plotting decision trees within our trained XGBoost classifier

# to get insight into the gradient boosting process

ax = xgb.plot_tree(model,num_trees=0)

ax.figure.set_size_inches(13, 9)

ax.set(title='XGB classifier boosting process')



#ax.figure.savefig(fig_dir+'tree.png')

plt.show()
# use XGBoost library's built-in function to plot features ordered by their importance

ax = xgb.plot_importance(model)

ax.figure.set_size_inches(13, 9)

ax.figure.tight_layout()



#ax.figure.savefig(fig_dir+'fi.png')

plt.show()
def rankings(df, player):

    """

    For a given player, return his rank and date for each match.

    """

    

    idx_w = df_atp[(df_atp.Winner == player)].index

    idx_l = df_atp[(df_atp.Loser == player)].index

    dates = df_atp.Date.iloc[idx_w.append(idx_l)].sort_index()

    rankings = df_atp['WRank'].iloc[idx_w].append(df_atp['LRank'].iloc[idx_l]).sort_index()



    return dates, rankings







fig = plt.figure(figsize=(13, 9))



x, y = rankings(df_atp, 'Federer R.')

plt.plot_date(x, y, '-.')



x, y = rankings(df_atp, 'Nadal R.')

plt.plot_date(x, y, '-+')



x, y = rankings(df_atp, 'Djokovic N.')

plt.plot_date(x, y, '-v')



plt.xlabel('Year', fontsize=18)

plt.ylabel('ATP Entry Rank', fontsize=18)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.legend(['Federer R.', 'Nadal R.', 'Djokovic N.'], fontsize=14)

plt.title('ATP entry rank evolution of the three biggest players')

#plt.ylim([0, 50])



#fig.savefig(fig_dir+'ranking.png')

plt.show()
# number of Grand Slam victories by player (first 15 players)

df_gs = df_atp[df_atp.Series == 'Grand Slam'].groupby(['Winner']).agg({

    'ATP':'count'}).sort_values(by='ATP', ascending=False).iloc[0:15]



fig, ax = plt.subplots(figsize=(13,9))

ax = sn.barplot(x=df_gs.index, y=df_gs.ATP)

ax.set_xticklabels(labels=df_gs.index, rotation=90)

ax.set(title='Number of Grand Slam victories since 2000')

ax.figure.tight_layout()



#fig.savefig(fig_dir+"grand_slam.png")

plt.show()
df_gs2 = df_atp[df_atp.Series == 'Grand Slam'].groupby(['Winner', 'Tournament']).agg({

    'ATP':'count'}).sort_values(by='ATP', ascending=False).iloc[0:35]

df_gs2 = df_gs2.reset_index()



ax = sn.catplot(x='Winner', y='ATP', hue='Tournament', data=df_gs2, kind="bar",

                height=6, aspect=2)

ax.set_xticklabels(labels=df_gs2.Winner.unique(), rotation=90)

ax.set(title='Number of Grand Slam victories by tournament since 2000')

mpl.rcParams['figure.figsize'] = (13,9)



#ax.savefig(fig_dir+"grand_slam_tournament.png")

plt.show()
df_sur = df_atp.groupby(['Winner', 'Surface']).agg({'ATP':'count'})

df_FR = df_sur.loc['Federer R.']

df_NR = df_sur.loc['Nadal R.']

df_DN = df_sur.loc['Djokovic N.']





labels = df_FR.index

angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)

angles = np.concatenate((angles,[angles[0]]))

FR = np.concatenate((df_FR.ATP,[df_FR.ATP[0]]))

NR = np.concatenate((df_NR.ATP,[df_NR.ATP[0]]))

DN = np.concatenate((df_DN.ATP,[df_DN.ATP[0]]))



fig = plt.figure(figsize=(13,9))

ax = fig.add_subplot(111, polar=True)

ax.plot(angles, FR, 'o-', linewidth=2, label='Federer R.')

ax.plot(angles, NR, 'o-', linewidth=2, label='Nadal R.')

ax.plot(angles, DN, 'o-', linewidth=2, label='Djokovic N.')

ax.fill(angles, FR, alpha=0.25)

ax.set_thetagrids(angles * 180/np.pi, labels)

ax.grid(True)

plt.legend(loc='upper right', bbox_to_anchor=(1.2,1))



#fig.savefig(fig_dir+"surface_effect.png")

plt.show()