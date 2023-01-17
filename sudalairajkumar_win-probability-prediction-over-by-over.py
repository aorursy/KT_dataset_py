import operator

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import xgboost as xgb

import seaborn as sns

%matplotlib inline



pd.set_option('display.max_columns', 50)
data_path = "../input/"

score_df = pd.read_csv(data_path+"deliveries.csv")

match_df = pd.read_csv(data_path+"matches.csv")

score_df.head()
# Let us take only the matches played in 2016 for this analysis #

match_df = match_df.ix[match_df.season==2016,:]

match_df = match_df.ix[match_df.dl_applied == 0,:]

match_df.head()
# runs and wickets per over #

score_df = pd.merge(score_df, match_df[['id','season', 'winner', 'result', 'dl_applied', 'team1', 'team2']], left_on='match_id', right_on='id')

score_df.player_dismissed.fillna(0, inplace=True)

score_df['player_dismissed'].ix[score_df['player_dismissed'] != 0] = 1

train_df = score_df.groupby(['match_id', 'inning', 'over', 'team1', 'team2', 'batting_team', 'winner'])[['total_runs', 'player_dismissed']].agg(['sum']).reset_index()

train_df.columns = train_df.columns.get_level_values(0)



# innings score and wickets #

train_df['innings_wickets'] = train_df.groupby(['match_id', 'inning'])['player_dismissed'].cumsum()

train_df['innings_score'] = train_df.groupby(['match_id', 'inning'])['total_runs'].cumsum()

train_df.head()



# Get the target column #

temp_df = train_df.groupby(['match_id', 'inning'])['total_runs'].sum().reset_index()

temp_df = temp_df.ix[temp_df['inning']==1,:]

temp_df['inning'] = 2

temp_df.columns = ['match_id', 'inning', 'score_target']

train_df = train_df.merge(temp_df, how='left', on = ['match_id', 'inning'])

train_df['score_target'].fillna(-1, inplace=True)



# get the remaining target #

def get_remaining_target(row):

    if row['score_target'] == -1.:

        return -1

    else:

        return row['score_target'] - row['innings_score']



train_df['remaining_target'] = train_df.apply(lambda row: get_remaining_target(row),axis=1)



# get the run rate #

train_df['run_rate'] = train_df['innings_score'] / train_df['over']



# get the remaining run rate #

def get_required_rr(row):

    if row['remaining_target'] == -1:

        return -1.

    elif row['over'] == 20:

        return 99

    else:

        return row['remaining_target'] / (20-row['over'])

    

train_df['required_run_rate'] = train_df.apply(lambda row: get_required_rr(row), axis=1)



def get_rr_diff(row):

    if row['inning'] == 1:

        return -1

    else:

        return row['run_rate'] - row['required_run_rate']

    

train_df['runrate_diff'] = train_df.apply(lambda row: get_rr_diff(row), axis=1)

train_df['is_batting_team'] = (train_df['team1'] == train_df['batting_team']).astype('int')

train_df['target'] = (train_df['team1'] == train_df['winner']).astype('int')



train_df.head()
x_cols = ['inning', 'over', 'total_runs', 'player_dismissed', 'innings_wickets', 'innings_score', 'score_target', 'remaining_target', 'run_rate', 'required_run_rate', 'runrate_diff', 'is_batting_team']



# let us take all the matches but for the final as development sample and final as val sample #

val_df = train_df.ix[train_df.match_id == 577,:]

dev_df = train_df.ix[train_df.match_id != 577,:]



# create the input and target variables #

dev_X = np.array(dev_df[x_cols[:]])

dev_y = np.array(dev_df['target'])

val_X = np.array(val_df[x_cols[:]])[:-1,:]

val_y = np.array(val_df['target'])[:-1]

print(dev_X.shape, dev_y.shape)

print(val_X.shape, val_y.shape)
# define the function to create the model #

def runXGB(train_X, train_y, seed_val=0):

    param = {}

    param['objective'] = 'binary:logistic'

    param['eta'] = 0.05

    param['max_depth'] = 8

    param['silent'] = 1

    param['eval_metric'] = "auc"

    param['min_child_weight'] = 1

    param['subsample'] = 0.7

    param['colsample_bytree'] = 0.7

    param['seed'] = seed_val

    num_rounds = 100



    plst = list(param.items())

    xgtrain = xgb.DMatrix(train_X, label=train_y)

    model = xgb.train(plst, xgtrain, num_rounds)

    return model
# let us build the model and get predcition for the final match #

model = runXGB(dev_X, dev_y)

xgtest = xgb.DMatrix(val_X)

preds = model.predict(xgtest)
def create_feature_map(features):

    outfile = open('xgb.fmap', 'w')

    for i, feat in enumerate(features):

        outfile.write('{0}\t{1}\tq\n'.format(i,feat))

    outfile.close()



create_feature_map(x_cols)

importance = model.get_fscore(fmap='xgb.fmap')

importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)

imp_df = pd.DataFrame(importance, columns=['feature','fscore'])

imp_df['fscore'] = imp_df['fscore'] / imp_df['fscore'].sum()



# create a function for labeling #

def autolabel(rects):

    for rect in rects:

        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,

                '%f' % float(height),

                ha='center', va='bottom')

        

labels = np.array(imp_df.feature.values)

ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots(figsize=(12,6))

rects = ax.bar(ind, np.array(imp_df.fscore.values), width=width, color='y')

ax.set_xticks(ind+((width)/2.))

ax.set_xticklabels(labels, rotation='vertical')

ax.set_ylabel("Importance score")

ax.set_title("Variable importance")

autolabel(rects)

plt.show()
out_df = pd.DataFrame({'Team1':val_df.team1.values})

out_df['is_batting_team'] = val_df.is_batting_team.values

out_df['innings_over'] = np.array(val_df.apply(lambda row: str(row['inning']) + "_" + str(row['over']), axis=1))

out_df['innings_score'] = val_df.innings_score.values

out_df['innings_wickets'] = val_df.innings_wickets.values

out_df['score_target'] = val_df.score_target.values

out_df['total_runs'] = val_df.total_runs.values

out_df['predictions'] = list(preds)+[1]



fig, ax1 = plt.subplots(figsize=(12,6))

ax2 = ax1.twinx()

labels = np.array(out_df['innings_over'])

ind = np.arange(len(labels))

width = 0.7

rects = ax1.bar(ind, np.array(out_df['innings_score']), width=width, color=['yellow']*20 + ['green']*20)

ax1.set_xticks(ind+((width)/2.))

ax1.set_xticklabels(labels, rotation='vertical')

ax1.set_ylabel("Innings score")

ax1.set_xlabel("Innings and over")

ax1.set_title("Win percentage prediction for Sunrisers Hyderabad - over by over")



ax2.plot(ind+0.35, np.array(out_df['predictions']), color='b', marker='o')

ax2.plot(ind+0.35, np.array([0.5]*40), color='red', marker='o')

ax2.set_ylabel("Win percentage", color='b')

ax2.set_ylim([0,1])

ax2.grid(b=False)

plt.show()
fig, ax1 = plt.subplots(figsize=(12,6))

ax2 = ax1.twinx()

labels = np.array(out_df['innings_over'])

ind = np.arange(len(labels))

width = 0.7

rects = ax1.bar(ind, np.array(out_df['total_runs']), width=width, color=['yellow']*20 + ['green']*20)

ax1.set_xticks(ind+((width)/2.))

ax1.set_xticklabels(labels, rotation='vertical')

ax1.set_ylabel("Runs in the given over")

ax1.set_xlabel("Innings and over")

ax1.set_title("Win percentage prediction for Sunrisers Hyderabad - over by over")



ax2.plot(ind+0.35, np.array(out_df['predictions']), color='b', marker='o')

ax2.plot(ind+0.35, np.array([0.5]*40), color='red', marker='o')

ax2.set_ylabel("Win percentage", color='b')

ax2.set_ylim([0,1])

ax2.grid(b=False)

plt.show()