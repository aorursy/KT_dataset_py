import pandas as pd

import numpy as np

from sklearn import ensemble 

from sklearn import metrics



# this is meant to be a simple example so only matches and players are used

matches = pd.read_csv('../input/match.csv', index_col=0)

players = pd.read_csv('../input/players.csv')



test_labels = pd.read_csv('../input/test_labels.csv', index_col=0)

test_players = pd.read_csv('../input/test_player.csv')



train_labels = matches['radiant_win'].astype(int)
# take a look at the match data

matches.head()
# since this is a simple example I will use very basic features which are probably not very good.

feature_columns = players.iloc[:3,4:17].columns.tolist()

feature_columns
player_groups = players.groupby('account_id')



# These are just a the mean of the above values, one for each account

feature_components = player_groups[feature_columns].mean()
# the account_id 0 is included even though it represents more then one account 

# its average stats for players who hide their account ids 

feature_components.head()
# now to construct match_level features from the components

# account_id is needed to join with feature_components

train_ids = players[['match_id','account_id']]

test_ids = test_players[['match_id','account_id']]
# add player component data to full match and player data

# note if a player is not in the train set but appears in the test set they will have 

# nan values inserted



train_feat_comp = pd.merge(train_ids, feature_components,

                           how='left', left_on='account_id' ,

                           right_index=True)



test_feat_comp = pd.merge(test_ids, feature_components, 

                          how='left', left_on='account_id',

                          right_index=True)
# this is no longer needed now that the join is done 

train_feat_comp.drop(['account_id'], axis=1, inplace=True)

test_feat_comp.drop(['account_id'], axis=1, inplace=True)



# this basically flattens an entire match, removes the redundent match_ids, and then 

# drops the unneaded multi-index

# is there a better way to do this?

def unstack_simplify(df):

    return df.unstack().iloc[10:].reset_index(drop=True)
# group by match then combine data for all players in a match into one row

test_feat_group = test_feat_comp.groupby('match_id')

test_feats = test_feat_group.apply(unstack_simplify)
train_feat_group = train_feat_comp.groupby('match_id')

train_feats = train_feat_group.apply(unstack_simplify)
test_feats.head()
for i in range(0,40, 10):

    print(test_feats.iloc[0,i:i+10],'\n')
row_nans = test_feats.isnull().sum(axis=1)

nan_counts = row_nans.value_counts()

nan_counts = nan_counts.reset_index()



nan_counts.columns = ['num_missing_players','count']

nan_counts.loc[:, 'num_missing_players'] =(nan_counts.loc[:,'num_missing_players']/12).astype(int)

nan_counts



# counting how many players are missing from match because they didn't exist in 

# the train set
rf = ensemble.RandomForestClassifier(n_estimators=150, n_jobs=-1)

rf.fit(train_feats,train_labels) 





# this is a bad way to deal with missing values 

test_feats.replace(np.nan, 0, inplace=True)



test_probs = rf.predict_proba(test_feats)

test_preds = rf.predict(test_feats)
metrics.log_loss(test_labels.values.ravel(), test_probs[:,1])
metrics.roc_auc_score(test_labels.values, test_probs[:,1])
print(metrics.classification_report(test_labels.values, test_preds))