# Importing the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Importing the models
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
# Importing datasets
train_X = pd.read_csv('../input/mlcourse-dota2-win-prediction/train_features.csv', index_col = 0)
train_y = pd.read_csv('../input/mlcourse-dota2-win-prediction/train_targets.csv', index_col = 0)
test_X = pd.read_csv('../input/mlcourse-dota2-win-prediction/test_features.csv', index_col = 0)
traindf = pd.merge(train_X, train_y, left_index = True, right_index = True)

traindf.head()
traindf.dtypes
traindf.select_dtypes(exclude='number')
# Convert radiant_win into numerical data
traindf['radiant_win'] = (traindf['radiant_win'])*1 
traindf['radiant_win'].dtypes

# Convert next_roshan_team into numerical data
traindf['next_roshan_team_radiant'] = (traindf['next_roshan_team'] == 'Radiant')*1
traindf['next_roshan_team_dire'] = (traindf['next_roshan_team'] == 'Dire')*1
traindf = traindf.drop('next_roshan_team', axis=1)

# Checking Dtype again
tempdf = traindf.filter(['radiant_win','next_roshan_team_radiant','next_roshan_team_dire'],axis=1)
tempdf.dtypes
# Creating Radiant Team columns
traindf['radiant_gold'] = traindf['r1_gold'] + traindf['r2_gold'] + traindf['r3_gold'] + traindf['r4_gold'] + traindf['r5_gold']
traindf['radiant_xp'] = traindf['r1_xp'] + traindf['r2_xp'] + traindf['r3_xp'] + traindf['r4_xp'] + traindf['r5_xp']
traindf['radiant_gold_xp_ratio'] = traindf['radiant_gold']/traindf['radiant_xp']
traindf['radiant_vision'] = traindf['r1_obs_placed'] + traindf['r2_obs_placed'] + traindf['r3_obs_placed'] + traindf['r4_obs_placed'] + traindf['r5_obs_placed'] + traindf['r1_sen_placed'] + traindf['r2_sen_placed'] + traindf['r3_sen_placed'] + traindf['r4_sen_placed'] + traindf['r5_sen_placed']
traindf['radiant_sen'] = traindf['r1_sen_placed'] + traindf['r2_sen_placed'] + traindf['r3_sen_placed'] + traindf['r4_sen_placed'] + traindf['r5_sen_placed']
traindf['radiant_towers_killed'] = traindf['r1_towers_killed'] + traindf['r2_towers_killed'] + traindf['r3_towers_killed'] + traindf['r4_towers_killed'] + traindf['r5_towers_killed']
traindf['radiant_stun'] = traindf['r1_stuns'] + traindf['r2_stuns'] + traindf['r3_stuns'] + traindf['r4_stuns'] + traindf['r5_stuns']
# Creating Dire Team columns
traindf['dire_gold'] = traindf['d1_gold'] + traindf['d2_gold'] + traindf['d3_gold'] + traindf['d4_gold'] + traindf['d5_gold']
traindf['dire_xp'] = traindf['d1_xp'] + traindf['d2_xp'] + traindf['d3_xp'] + traindf['d4_xp'] + traindf['d5_xp']
traindf['dire_gold_xp_ratio'] = traindf['dire_gold']/traindf['dire_xp']
traindf['dire_vision'] = traindf['d1_obs_placed'] + traindf['d2_obs_placed'] + traindf['d3_obs_placed'] + traindf['d4_obs_placed'] + traindf['d5_obs_placed'] + traindf['d1_sen_placed'] + traindf['d2_sen_placed'] + traindf['d3_sen_placed'] + traindf['d4_sen_placed'] + traindf['d5_sen_placed']
traindf['dire_sen'] = traindf['d1_sen_placed'] + traindf['d2_sen_placed'] + traindf['d3_sen_placed'] + traindf['d4_sen_placed'] + traindf['d5_sen_placed']
traindf['dire_towers_killed'] = traindf['d1_towers_killed'] + traindf['d2_towers_killed'] + traindf['d3_towers_killed'] + traindf['d4_towers_killed'] + traindf['d5_towers_killed']
traindf['dire_stun'] = traindf['d1_stuns'] + traindf['d2_stuns'] + traindf['d3_stuns'] + traindf['d4_stuns'] + traindf['d5_stuns']

# Creating Team Variance Columns (From Radiance POV)
traindf['var_gold'] = traindf['radiant_gold'] - traindf['dire_gold']
traindf['var_xp'] = traindf['radiant_xp'] - traindf['dire_xp']
traindf['var_gold_xp_ratio'] = traindf['radiant_gold_xp_ratio']/traindf['dire_gold_xp_ratio']
traindf['var_vision'] = traindf['radiant_vision'] - traindf['dire_vision']
traindf['var_sen'] = traindf['radiant_sen'] - traindf['dire_sen']
traindf['var_towers_killed'] = traindf['radiant_towers_killed'] - traindf['dire_towers_killed']
traindf = traindf.replace([np.inf, -np.inf], np.nan).fillna(0)

# Print top 10 correlated features to 'radiant_win'
cor = traindf.corr()
cor_target = abs(cor['radiant_win'])
cor_target = cor_target.sort_values(ascending=False)
print(cor_target.head(10))

# Plot top 10 correlated features into heatmap
selected_col = cor_target.head(10).index 
tempdf = traindf[selected_col]
f, ax = plt.subplots(figsize=(12, 9))
cor2 = tempdf.corr()
sns.heatmap(cor2, vmax=.8, annot=True, square=True, cmap="YlGnBu");
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5 )
plt.show()
# Keeping only relevant columns
cols_to_keep = ['var_xp','var_gold','var_towers_killed','game_time_x']
traindf_X = traindf.filter(cols_to_keep,axis=1)
traindf_y = traindf[['radiant_win']]

traindf_X.head()
traindf_y.head()
# Comparing scatterplots among key features
sns.set()
sns.pairplot(traindf[cols_to_keep], height = 2.5)
plt.show();
# Dropping outliers where game_time_x == 0
traindf = traindf_X
traindf['radiant_win'] = traindf_y
traindf = traindf[traindf.game_time_x > 0]
traindf_y = traindf[['radiant_win']]
traindf_X = traindf.drop(['radiant_win'],axis=1)
# Modelling with Logistics Regression
model = LogisticRegression(solver='lbfgs')

# calcuate ROC-AUC for each split
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=8)
cv_score_mean = (cross_val_score(model, traindf_X, traindf_y.values.ravel(), cv=cv, scoring='roc_auc')).mean()
print(cv_score_mean)
# prep test_X data
test_X['radiant_gold'] = test_X['r1_gold'] + test_X['r2_gold'] + test_X['r3_gold'] + test_X['r4_gold'] + test_X['r5_gold']
test_X['radiant_xp'] = test_X['r1_xp'] + test_X['r2_xp'] + test_X['r3_xp'] + test_X['r4_xp'] + test_X['r5_xp']
test_X['radiant_towers_killed'] = test_X['r1_towers_killed'] + test_X['r2_towers_killed'] + test_X['r3_towers_killed'] + test_X['r4_towers_killed'] + test_X['r5_towers_killed']
test_X['dire_gold'] = test_X['d1_gold'] + test_X['d2_gold'] + test_X['d3_gold'] + test_X['d4_gold'] + test_X['d5_gold']
test_X['dire_xp'] = test_X['d1_xp'] + test_X['d2_xp'] + test_X['d3_xp'] + test_X['d4_xp'] + test_X['d5_xp']
test_X['dire_towers_killed'] = test_X['d1_towers_killed'] + test_X['d2_towers_killed'] + test_X['d3_towers_killed'] + test_X['d4_towers_killed'] + test_X['d5_towers_killed']
test_X['var_gold'] = test_X['radiant_gold'] - test_X['dire_gold']
test_X['var_xp'] = test_X['radiant_xp'] - test_X['dire_xp']
test_X['var_towers_killed'] = test_X['radiant_towers_killed'] - test_X['dire_towers_killed']

# replace game_time_x with game_time for test_X
cols_to_keep = ['var_xp','var_gold','var_towers_killed','game_time']
test_X = test_X.filter(cols_to_keep,axis=1)

# Make predictions and create submission file
model.fit(traindf_X, traindf_y.values.ravel())
submission = pd.DataFrame()
submission['match_id_hash'] = test_X.index
submission['radiant_win_prob']= model.predict_proba(test_X)[:, 1]
submission.to_csv('submission.csv', index=False)
