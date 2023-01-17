import numpy as np 
import pandas as pd 
from datetime import datetime
nba_df = pd.read_csv("/kaggle/input/nba-games-stats-from-2014-to-2018/nba.games.stats.csv")
print(nba_df.shape)
nba_df.sample(5)
# Drop irrelevant column:

nba_df=nba_df.drop('Unnamed: 0', axis=1)
# create defence rebounds field for team and opponent:

nba_df['DefRebounds'] = nba_df['TotalRebounds'] - nba_df['OffRebounds']
nba_df['Opp.DefRebounds'] = nba_df['Opp.TotalRebounds'] - nba_df['Opp.OffRebounds']
nba_df.shape
# Cleaning texts for easier handling:

nba_df.columns = nba_df.columns.str.replace('Opp.', 'Opp')
nba_df.columns = nba_df.columns.str.replace('Opp3PointShots', 'OppX3PointShots' )
# 2-class columns to Boolean:

nba_df['Home'] = nba_df['Home'].replace(['Home', 'Away'], [1,0]).astype(str).astype(int)
# 2-class columns to Boolean - Leave Column as such for ML:

nba_df['WINorLOSS'] = nba_df['WINorLOSS'].replace(['W', 'L'], [1,0]).astype(str).astype(int)
PIR = ((nba_df['TeamPoints'] + nba_df['TotalRebounds'] + nba_df['Assists'] 
     + nba_df['Steals'] + nba_df['Blocks'] + nba_df['OppTotalFouls']) 
       
       # Missed Field Goals:
    - ((nba_df['FieldGoalsAttempted']- nba_df['FieldGoals'])
       
       # Missed Free Throws:
    +(nba_df['FreeThrowsAttempted'] - nba_df['FreeThrows']) 
    + nba_df['Turnovers'] + nba_df['OppBlocks'] + nba_df['TotalFouls']))


OppPIR = ((nba_df['OppnentPoints'] + nba_df['OppTotalRebounds'] + nba_df['OppAssists'] 
     + nba_df['OppSteals'] + nba_df['OppBlocks'] + nba_df['TotalFouls']) 
    - ((nba_df['OppFieldGoalsAttempted']- nba_df['OppFieldGoals'])
    +(nba_df['OppFreeThrowsAttempted'] - nba_df['OppFreeThrows']) 
    + nba_df['OppTurnovers'] + nba_df['Blocks'] + nba_df['OppTotalFouls']))       

nba_df['PIR'] = pd.Series(PIR)
nba_df['OppPIR'] = pd.Series(OppPIR)
nba_df['Month'] = [int(m[5:7]) for m in nba_df['Date']]
nba_df['Year'] = [int(y[:4]) for y in nba_df['Date']]
def seasons(d):
  m=d[5:7]
  y=d[:4] 
  if (y =='2014' and m in ('10','11','12')) or (y=='2015' and m in ('01','02','03','04')):
    s='one'
  elif (y=='2015' and m in ('10','11','12')) or (y=='2016' and m in ('01','02','03','04')):
    s='two'
  elif (y=='2016' and m in ('10','11','12')) or (y=='2017' and m in ('01','02','03','04')):
    s='three'
  else:
    s='four'
  return (s)

nba_df['Season']=nba_df['Date'].apply(seasons)

# Question: 1-4 are integers. Should we change to text?
# Categorizing months into halfs (February contains the AllStar break). It will follow by dummies:

def halfs(x):
    if x in ('10','11','12','01'):
        x = 'Pre_AllStar'
    else:
        x = 'Post_AllStar'
    return (x)

nba_df['Season_half']=nba_df['Month'].apply(halfs)
# Points Differance at the end of the game:

nba_df['diff_points']=abs(nba_df['TeamPoints']-nba_df['OppnentPoints'])
total_wins_season = pd.DataFrame(nba_df.groupby(['Team', 'Season'])['WINorLOSS'].sum())
total_wins_season.iloc[75:85]
team_rank = nba_df.merge(total_wins_season, left_on=['Team','Season'], right_index = True)
nba_df['team_rank'] = team_rank['WINorLOSS_y']
nba_df_for_ML = nba_df[['Game', 'Home','WINorLOSS', 'Season', 'Season_half']].copy()
# Team stats coulmns are averaged for the last 5 games. The first 5 games are group_aggregated.
# In some runs, Opponent Columns were averaged too, but it was not beneficial

def five_last_games_avg(col):
  first_games_avg_dict = {}

  for idx in nba_df.index:
    sum_points=0
    if nba_df.loc[idx]['Game'] ==1:
      first_games_avg = (nba_df.iloc[idx:idx+5][col].sum())/5
      first_games_avg_dict.update({idx:first_games_avg})
      for i in range(5):
        first_games_avg_dict.update({idx+i:first_games_avg})
    elif nba_df.loc[idx]['Game'] > 5:
      y = nba_df.loc[idx]['Game'] -5
      for i in range(y,nba_df.loc[idx]['Game']):
          sum_points+=nba_df.loc[i][col]
          first_games_avg_dict.update({idx:(sum_points/5)})

  nba_df_for_ML[f'avg_{col}'] = pd.Series(first_games_avg_dict)
# VERY VERY SLOW:
# %%timeit
# 1 loop, best of 3: 5min 57s per loop

nba_df_columns_for_avg = ['PIR', 'OppPIR','TeamPoints',
        'FieldGoalsAttempted', 'FieldGoals.',
        'X3PointShotsAttempted', 'X3PointShots.', 
       'FreeThrowsAttempted', 'FreeThrows.', 'OffRebounds', 
       'DefRebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers', 'TotalFouls','diff_points']

for a in nba_df_columns_for_avg:
  five_last_games_avg(a)
from datetime import timedelta
nba_df['Game_Date'] = pd.to_datetime(nba_df['Date'])
def scale_rest_days(i):
  x=nba_df.loc[i]['Game']
  if x>3:
    if (nba_df.loc[i]['Game_Date']- nba_df.loc[i-1]['Game_Date']).days==1:
      if (nba_df.loc[i]['Game_Date']- nba_df.loc[i-2]['Game_Date']).days==2:
        if (nba_df.loc[i]['Game_Date']- nba_df.loc[i-3]['Game_Date']).days==3:
          return -3
        else: 
          return -1
      elif (nba_df.loc[i]['Game_Date']- nba_df.loc[i-2]['Game_Date']).days==3:
        return 1
      else:
        return 2
    elif (nba_df.loc[i]['Game_Date']- nba_df.loc[i-1]['Game_Date']).days==2:
      if (nba_df.loc[i]['Game_Date']- nba_df.loc[i-2]['Game_Date']).days==3:
        return 3
      else:
        return 4
    elif (nba_df.loc[i]['Game_Date']- nba_df.loc[i-1]['Game_Date']).days==3:
      return 5
    else:
      return 6
  else:
    return 0
nba_df['ind']=nba_df.index.values
nba_df_for_ML['rest_days_scale']= nba_df['ind'].apply(scale_rest_days)
def sum_wins():
    streak_dict = {}
    sum_wins = 0
    for idx in nba_df_for_ML.index:
        if nba_df_for_ML.loc[idx]['Game'] == 1:
            sum_wins = nba_df_for_ML.loc[idx]['WINorLOSS']
            streak_dict.update({idx:sum_wins})
        else: 
            if nba_df_for_ML.loc[idx-1]['WINorLOSS'] == 0:
                sum_wins=0
                streak_dict.update({idx:sum_wins})
            else:
                sum_wins += nba_df_for_ML.loc[idx-1]['WINorLOSS']
                streak_dict.update({idx:sum_wins})            
    nba_df_for_ML['wins_streak'] = pd.Series(streak_dict)

sum_wins()
    
print(nba_df_for_ML.shape)
nba_df_for_ML.sample(5)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss, confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
# !pip install pydot
# import pydot
# from IPython.display import Image
# from sklearn.externals.six import StringIO
# from sklearn.tree import export_graphviz
# def visualize_tree(model, md=5, width=800):
#     dot_data = StringIO()  
#     export_graphviz(model, out_file=dot_data, feature_names=X_train.columns, max_depth=md)
#     graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]  
#     return Image(graph.create_png(), width=width)
# def print_dot_text(model, md=5):
#     """The output of this function can be copied to http://www.webgraphviz.com/"""
#     dot_data = StringIO()
#     export_graphviz(model, out_file=dot_data, feature_names=X_train.columns, max_depth=md)
#     dot_text = dot_data.getvalue()
#     print(dot_text)
nba_df_dumm = pd.get_dummies(nba_df_for_ML)
nba_df_for_ML.shape #(9840, 24)
nba_df_dumm.shape
X = nba_df_dumm.drop(['WINorLOSS'], axis=1)
y = nba_df_dumm.WINorLOSS

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=0.75, 
                                                    test_size=0.25,
                                                    shuffle=True, 
                                                    #stratify=nba_df_for_ML.WINorLOSS
                                                    )
dt_model_1 = DecisionTreeClassifier(min_samples_leaf=3, min_weight_fraction_leaf=0.01, max_leaf_nodes=40).fit(X_train, y_train)
# visualize_tree(dt_model_1, md=5, width=1200)
pd.Series(dt_model_1.feature_importances_, index=X_train.columns).sort_values().tail(20)\
    .plot.barh(title='Features importance');
features_importance = pd.Series(dt_model_1.feature_importances_, index=X_train.columns).sort_values().tail(20)
features_importance
# Cross Validation:

(-cross_val_score(dt_model_1, X_train, y_train, cv=9, scoring='neg_log_loss')).mean().round(3)

# f1 was also tested with worse results
y_test_pred = pd.DataFrame(dt_model_1.predict_proba(X_test), 
                        columns=dt_model_1.classes_)

log_loss(y_test, y_test_pred).mean().round(3)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
random_forest = RandomForestClassifier(n_estimators=100,max_leaf_nodes=100, min_weight_fraction_leaf= 0.01)
my_param_grid = [{"n_estimators": [30, 70 ,100],"max_leaf_nodes": [30, 70, 100], "min_samples_leaf" : [30, 70, 100]}]
k = 7
random_forest_gs = GridSearchCV(random_forest, my_param_grid, scoring='neg_log_loss',cv=k)
random_forest_gs.fit(X_train, y_train)
random_forest_gs.best_params_
random_forest_2 = random_forest_gs.best_estimator_
y_test_pred = pd.DataFrame(random_forest_2.predict_proba(X_test), 
                           columns=random_forest_2.classes_)
log_loss(y_true=y_test, y_pred=y_test_pred).mean().round(3)
(-cross_val_score(random_forest_2, X_train, y_train, cv=10, scoring='neg_log_loss')).mean().round(3)
y_test_pred = pd.DataFrame(dt_model_1.predict_proba(X_test), 
                        columns=dt_model_1.classes_)

log_loss(y_test, y_test_pred).mean().round(3)
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
clf_bagging = BaggingClassifier(base_estimator=random_forest_2, n_estimators=100, max_features=0.8)
clf_bagging.fit(X_train, y_train)

# done also with dt_model_1 (Decision Tree Classifier), with similar results
print(f"Begging classifier:\n \
    \ttrain accuracy: {clf_bagging.score(X_train, y_train):.3f}\n \
    \ttest accuracy: {clf_bagging.score(X_test, y_test):.3f}")
clf_adaboost = AdaBoostClassifier(base_estimator=random_forest_2,
                                  n_estimators=200,
                                  learning_rate=0.01)
clf_adaboost.fit(X_train, y_train)
print(f"DT ADA boosting classifier:\n \
    \ttrain accuracy: {clf_adaboost.score(X_train, y_train):.3f}\n \
    \ttest accuracy: {clf_adaboost.score(X_test, y_test):.3f}")
from sklearn.linear_model import LogisticRegression
nba_lr = LogisticRegression(max_iter=1000)
X = nba_df_dumm.drop(['WINorLOSS'], axis=1)
y = nba_df_dumm.WINorLOSS

nba_lr.fit(X, y)
accuracy_score(y_true=y_train,
               y_pred=nba_lr.predict(X_train)).mean().round(3)
(-cross_val_score(nba_lr, X_train, y_train, cv=9, scoring='neg_log_loss')).mean().round(3)
cross_val_score(nba_lr, X_train, y_train, cv=9, scoring='accuracy').mean().round(3)