from IPython.display import HTML



HTML('''<script>

code_show=true; 

function code_toggle() {

 if (code_show){

 $('div.input').hide();

 } else {

 $('div.input').show();

 }

 code_show = !code_show

} 

$( document ).ready(code_toggle);

</script>

<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')
# Import libraries



import numpy as np

import pandas as pd

from IPython.display import display

pd.options.display.max_columns = None # Displays all columns and when showing dataframes

import sqlite3

import warnings

warnings.filterwarnings("ignore") # Hide warnings

import matplotlib.pyplot as plt

%matplotlib inline

import datetime as dt

import math

import time

from scipy.stats import poisson

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, roc_curve, average_precision_score, auc

from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
# Import the data

t0 = time.time()

'''

#For running on local machine, use:

path = ''   





'''

# For Kaggle kernels, use: 

path = "../input/"



with sqlite3.connect(path + 'database.sqlite') as con:

    countries = pd.read_sql_query("SELECT * from Country", con)

    matches = pd.read_sql_query("SELECT * from Match", con)

    leagues = pd.read_sql_query("SELECT * from League", con)

    teams = pd.read_sql_query("SELECT * from Team", con)

    player = pd.read_sql_query("SELECT * from Player",con)

    player_attributes = pd.read_sql_query("SELECT * from Player_Attributes",con)

    sequence = pd.read_sql_query("SELECT * from sqlite_sequence",con)

    team_attributes = pd.read_sql_query("SELECT * from Team_Attributes",con)

    

t1=time.time()

print('Data imported - time taken to run this step %i minutes and %i seconds' % ((t1-t0)//60,(t1-t0)%60))
print('The number of matches included in the data is %i' % np.shape(matches)[0])
# Create a dataframe with all home team and away team names and the result

temp_df = matches.copy() #for test runs, just take a sample of match data at this line

temp_df_2 = teams.copy()

temp_df_2['home_team_api_id'] = temp_df_2['team_api_id']

temp_df = temp_df.merge(temp_df_2,on='home_team_api_id',how='outer')

temp_df['home_team'] = temp_df['team_long_name']

temp_df_2['away_team_api_id'] = temp_df_2['team_api_id']

temp_df = temp_df.merge(temp_df_2,on='away_team_api_id',how='outer')

temp_df['away_team'] = temp_df['team_long_name_y']

temp_df_3 = temp_df.merge(leagues,on='country_id',how='outer')

temp_df['league_name'] = temp_df_3['name']





def points(goals_scored, goals_conceded):

    ''' (int, int) --> int

    

    Returns 3 points for a win, 1 for a draw and 0 for a loss.

    

    Pre-condition: Goals scored and conceded must be non-negative.

    

    >>> points(3,1)

    3

    

    >>> points(0,0)

    1

    

    >>> points(-1,2)

    None

    

    '''

    

    if goals_scored < 0 or goals_conceded < 0:

        return None

    elif goals_scored > goals_conceded:

        return 3

    elif goals_scored == goals_conceded:

        return 1

    else:

        return 0



# Add the points and result 

temp_df['home_points'] = temp_df.apply(lambda x: (points(x['home_team_goal'],x['away_team_goal'])),axis=1)

temp_df['away_points'] = temp_df.apply(lambda x: (points(x['away_team_goal'],x['home_team_goal'])),axis=1)

temp_df['scoreline'] = temp_df.apply(lambda x: (str(x['home_team_goal'])+'-'+str(x['away_team_goal'])),axis=1)

temp_df['total_goals'] = temp_df.apply(lambda x: (x['home_team_goal']+x['away_team_goal']),axis=1)



def result(home_points, total_goals):

    ''' (int) --> str

    

    Returns the result, based on the points won by the home team and the total goals

     

    >>> result(3,1)

    'Home win'

    

    >>> points(1,0)

    'No score draw'

    

    >>> points(1,2)

    'Score draw'

    

    >>> points(0,2)

    'Home loss'

    

    '''

    

    if home_points == 3:

        return 'Home win'

    elif home_points == 0:

        return 'Home loss'

    else:

        if total_goals == 0:

            return 'No score draw'

        else:

            return 'Score draw'





temp_df['result'] = temp_df.apply(lambda x: (result(x['home_points'],x['total_goals'])),axis=1)

#temp_df

match_results = temp_df
# Need to create player database in order to determine profile of starting lineups

temp_df = matches[['home_player_1',

                  'home_player_2',

                  'home_player_3',

                  'home_player_4',

                  'home_player_5',

                  'home_player_6',

                  'home_player_7',

                  'home_player_8',

                  'home_player_9',

                  'home_player_10',

                  'home_player_11',

                  'away_player_1',

                  'away_player_2',

                  'away_player_3',

                  'away_player_4',

                  'away_player_5',

                  'away_player_6',

                  'away_player_7',

                  'away_player_8',

                  'away_player_9',

                  'away_player_10',

                  'away_player_11'

                  ]]



temp_df_2 = pd.DataFrame(temp_df.apply(pd.value_counts).fillna(0).sum(axis=1),columns=['appearances'])

temp_df = player.copy()

temp_df = temp_df.set_index('player_api_id')

temp_df['appearances'] = 0

temp_df['appearances'][temp_df_2.index] = temp_df_2['appearances']

player_data = temp_df[['player_name','birthday','height','weight','appearances']]
# Add team player data for each match

# This section takes a long time to run (c25 minutes)

t0 = time.time()

home_age_dict = {}

away_age_dict = {}

home_height_dict= {}

away_height_dict = {}

home_weight_dict = {}

away_weight_dict = {}



for match in match_results.itertuples():

    match_api_id = match[7]

    date = pd.to_datetime(match[6])

    home_matched = 0

    away_matched = 0

    home_total_age = 0

    away_total_age = 0

    home_total_height = 0

    away_total_height = 0

    home_total_weight = 0

    away_total_weight = 0

    for i in range(22):

        if match[56+i] > 0:

            player_id = match[56+i]

            if i < 12:

                home_matched += 1

                home_total_age += (date - pd.to_datetime(player_data.ix[player_id]['birthday'])).days / 365.25

                home_total_height += player_data.ix[player_id]['height']

                home_total_weight += player_data.ix[player_id]['weight']

            else:

                away_matched += 1

                away_total_age += (date - pd.to_datetime(player_data.ix[player_id]['birthday'])).days / 365.25

                away_total_height += player_data.ix[player_id]['height']

                away_total_weight += player_data.ix[player_id]['weight']

    if home_matched > 0:

        home_age_dict[match_api_id] = home_total_age / home_matched

        home_height_dict[match_api_id] = home_total_height / home_matched

        home_weight_dict[match_api_id] = home_total_weight / home_matched

    if away_matched > 0:

        away_age_dict[match_api_id] = away_total_age / away_matched

        away_height_dict[match_api_id] = away_total_height / away_matched

        away_weight_dict[match_api_id] = away_total_weight / away_matched



match_results = match_results.set_index('match_api_id')

match_results['home_age'] = match_results.index.map(home_age_dict)

match_results['away_age'] = match_results.index.map(away_age_dict)

match_results['home_height'] = match_results.index.map(home_height_dict)

match_results['away_height'] = match_results.index.map(away_height_dict)

match_results['home_weight'] = match_results.index.map(home_weight_dict)

match_results['away_weight'] = match_results.index.map(away_weight_dict)



# For missing players, populate age, weight and height with average values

avg_age = (match_results['home_age'].mean() + match_results['away_age'])/2

avg_height = (match_results['home_height'].mean() + match_results['away_height'])/2

avg_weight = (match_results['home_weight'].mean() + match_results['away_weight'])/2



match_results['home_age'] = match_results['home_age'].fillna(avg_age)

match_results['away_age'] = match_results['away_age'].fillna(avg_age)

match_results['home_height'] = match_results['home_height'].fillna(avg_height)

match_results['away_height'] = match_results['away_height'].fillna(avg_height)

match_results['home_weight'] = match_results['home_weight'].fillna(avg_weight)

match_results['away_weight'] = match_results['away_weight'].fillna(avg_weight)



t1=time.time()

print('Profile of starting 11 created - time taken to run this step %i minutes and %i seconds' % ((t1-t0)//60,(t1-t0)%60))
# Add recent form data for each match

# This section takes a very long time to run (c2 hours)



t0 = time.time()

home_points_dict = {}

away_points_dict = {}

home_goals_for_dict= {}

away_goals_for_dict = {}

home_goals_against_dict = {}

away_goals_against_dict = {}



seasons = match_results['season'].unique()

leagues = match_results['league_id'].unique()



for season in seasons:

    for league in leagues:

        match_results_temp = match_results[(match_results['season']==season) & (match_results['league_id']==league)]



        for match in match_results_temp.itertuples():

            match_api_id = match[0]

            date = pd.to_datetime(match[6])

            season = match[4]

            home_team = match[7]

            away_team = match[8]

            home_team_recent = match_results_temp[(match_results_temp['season']==season) & 

                                                  (match_results_temp['date'].apply(pd.to_datetime)<date) &

                                                  ((match_results_temp['home_team_api_id_x']==home_team) | (match_results_temp['away_team_api_id']==home_team))

                                                 ].sort_values(by='date',ascending=False).head(6)

            home_points = home_team_recent['home_points'][home_team_recent['home_team_api_id_x']==home_team].sum() + home_team_recent['away_points'][home_team_recent['away_team_api_id']==home_team].sum()

            home_goals_for = home_team_recent['home_team_goal'][home_team_recent['home_team_api_id_x']==home_team].sum() + home_team_recent['away_team_goal'][home_team_recent['away_team_api_id']==home_team].sum()

            home_goals_against = home_team_recent['away_team_goal'][home_team_recent['home_team_api_id_x']==home_team].sum() + home_team_recent['home_team_goal'][home_team_recent['away_team_api_id']==home_team].sum()

            away_team_recent = match_results_temp[(match_results_temp['season']==season) &

                                                  (match_results_temp['date'].apply(pd.to_datetime)<date) &

                                                  ((match_results_temp['home_team_api_id_x']==away_team) | (match_results_temp['away_team_api_id']==away_team))

                                                 ].sort_values(by='date',ascending=False).head(6)

            away_points = away_team_recent['home_points'][away_team_recent['home_team_api_id_x']==away_team].sum() + away_team_recent['away_points'][away_team_recent['away_team_api_id']==away_team].sum()

            away_goals_for = away_team_recent['home_team_goal'][away_team_recent['home_team_api_id_x']==away_team].sum() + away_team_recent['away_team_goal'][away_team_recent['away_team_api_id']==away_team].sum()

            away_goals_against = away_team_recent['away_team_goal'][away_team_recent['home_team_api_id_x']==away_team].sum() + away_team_recent['home_team_goal'][away_team_recent['away_team_api_id']==away_team].sum()

    

            if len(home_team_recent)>2: #Do not include form for the first two matches of the season

                n = len(home_team_recent)

                home_points_dict[match_api_id] = home_points / n

                home_goals_for_dict[match_api_id] = home_goals_for / n

                home_goals_against_dict[match_api_id] = home_goals_against / n

            if len(away_team_recent)>2: #Do not include form for the first three matches of the season

                n = len(away_team_recent)

                away_points_dict[match_api_id] = away_points / n

                away_goals_for_dict[match_api_id] = away_goals_for / n

                away_goals_against_dict[match_api_id] = away_goals_against / n

        

match_results['home_team_form'] = match_results.index.map(home_points_dict)

match_results['away_team_form'] = match_results.index.map(away_points_dict)

match_results['home_team_goals_for'] = match_results.index.map(home_goals_for_dict)

match_results['away_team_goals_for'] = match_results.index.map(away_goals_for_dict)

match_results['home_team_goals_against'] = match_results.index.map(home_goals_against_dict)

match_results['away_team_goals_against'] = match_results.index.map(away_goals_against_dict)

t1=time.time()

print('Team form statistics created - time taken to run this step %i minutes and %i seconds' % ((t1-t0)//60,(t1-t0)%60))
# Create some additional columns



match_results['age_difference'] = match_results['home_age'] - match_results['away_age']

match_results['height_difference'] = match_results['home_height'] - match_results['away_height']

match_results['weight_difference'] = match_results['home_weight'] - match_results['away_weight']

match_results['average_age'] = (match_results['home_age'] + match_results['away_age']) / 2

match_results['average_height'] = (match_results['home_height'] + match_results['away_height']) / 2

match_results['average_weight'] = (match_results['home_weight'] + match_results['away_weight']) / 2

match_results['home_team_recent_goals'] = match_results['home_team_goals_for'] + match_results['home_team_goals_against']

match_results['away_team_recent_goals'] = match_results['away_team_goals_for'] + match_results['away_team_goals_against']

match_results['combined_recent_goals'] = match_results['home_team_recent_goals'] + match_results['away_team_recent_goals']

match_results['combined_form'] = (match_results['home_team_form'] + match_results['away_team_form']) / 2





match_results['home_team_avg_position_X'] = (match_results['home_player_X1'] + 

                                          match_results['home_player_X2'] + 

                                          match_results['home_player_X3'] + 

                                          match_results['home_player_X4'] + 

                                          match_results['home_player_X5'] + 

                                          match_results['home_player_X6'] + 

                                          match_results['home_player_X7'] + 

                                          match_results['home_player_X8'] + 

                                          match_results['home_player_X9'] + 

                                          match_results['home_player_X10'] + 

                                          match_results['home_player_X11'] 

                                          ) / 11

match_results['home_team_avg_position_Y'] = (match_results['home_player_Y1'] + 

                                          match_results['home_player_Y2'] + 

                                          match_results['home_player_Y3'] + 

                                          match_results['home_player_Y4'] + 

                                          match_results['home_player_Y5'] + 

                                          match_results['home_player_Y6'] + 

                                          match_results['home_player_Y7'] + 

                                          match_results['home_player_Y8'] + 

                                          match_results['home_player_Y9'] + 

                                          match_results['home_player_Y10'] + 

                                          match_results['home_player_Y11'] 

                                          ) / 11

match_results['away_team_avg_position_X'] = (match_results['away_player_X1'] + 

                                          match_results['away_player_X2'] + 

                                          match_results['away_player_X3'] + 

                                          match_results['away_player_X4'] + 

                                          match_results['away_player_X5'] + 

                                          match_results['away_player_X6'] + 

                                          match_results['away_player_X7'] + 

                                          match_results['away_player_X8'] + 

                                          match_results['away_player_X9'] + 

                                          match_results['away_player_X10'] + 

                                          match_results['away_player_X11'] 

                                          ) / 11

match_results['away_team_avg_position_Y'] = (match_results['away_player_Y1'] + 

                                          match_results['away_player_Y2'] + 

                                          match_results['away_player_Y3'] + 

                                          match_results['away_player_Y4'] + 

                                          match_results['away_player_Y5'] + 

                                          match_results['away_player_Y6'] + 

                                          match_results['away_player_Y7'] + 

                                          match_results['away_player_Y8'] + 

                                          match_results['away_player_Y9'] + 

                                          match_results['away_player_Y10'] + 

                                          match_results['away_player_Y11'] 

                                          ) / 11
match_results['league_season'] = match_results['league_name'] + match_results['season']

stages = dict(match_results.groupby('league_season').max()['stage'])

match_results['stages_per_season'] = match_results['league_season'].map(stages)

match_results['last_day'] = match_results['stage']==match_results['stages_per_season']
#Add some extra features to the data

match_results['stage_8'] = match_results['stage']==8





# define variables to take into the model

# Some variables need to be redefined as binary variable and others need to be recoded as numeric values

# split the variables into those available well in advance of the fixture and those available only once the teamsheet has been announced



Advan_Var = ['league_name',

             'season',

             'stage',

             'stage_8',

             'home_team_form',

             'away_team_form',

             'home_team_goals_for',

             'away_team_goals_for',

             'home_team_goals_against',

             'away_team_goals_against',

             'home_team_recent_goals',

             'away_team_recent_goals',

             'combined_recent_goals',

             'last_day'

             ]



Tmsht_Var = ['home_age',

             'away_age',

             'home_height',

             'away_height',

             'home_weight',

             'away_weight',

             'age_difference',

             'height_difference',

             'weight_difference',

             'average_age',

             'average_height',

             'average_weight'

             ]



Full_Var = Advan_Var + Tmsht_Var
# Define X and y for algorithms and split into training, test and cross-validation data



X_full = match_results[Full_Var]



y_full = match_results['total_goals']



# encode string input values as integers

to_encode = ['league_name','season']

features=[]

for col in to_encode:

    feature = LabelEncoder().fit_transform(X_full[col])

    

X_full = X_full.drop(to_encode,axis=1)

X_full['league'] = feature[0]

X_full['season'] = feature[1]
# Shape of training data (num_rows, num_columns)

print('The shape of the data is:')

print(X_full.shape)

print('\n')



# Number of missing values in each column of training data

missing_val_count_by_column = (X_full.isnull().sum())

print('The columns with missing data are:')

print(missing_val_count_by_column[missing_val_count_by_column > 0])
rows_with_missing = X_full.index[X_full.isnull().any(axis=1)]

X_full.loc[rows_with_missing]['stage'].plot(kind='hist',bins=38,figsize=(15,5));

plt.title('Gameweeks with missing data');

plt.xlabel('Gameweek');



X_full = X_full.drop(rows_with_missing,axis=0)

# Shape of training data (num_rows, num_columns)

print('The revised shape of the data is:')

print(X_full.shape)

print('\n')
X_train, X_holdout, y_train, y_holdout = train_test_split(X_full,y_full,test_size=0.2,random_state=42)

X_cv, X_test, y_cv, y_test = train_test_split(X_holdout,y_holdout,test_size=0.5,random_state=42)



m = np.shape(X_train)[1]



print('There are %i matches in the training data' % len(X_train))

print('There are %i matches in the test data' % len(X_test))

print('There are %i matches in the cross-validation data' % len(X_cv))

print('The training data contains %i features' % m)
t0 = time.time()
# Set up the model

#clf = xgb.XGBClassifier(max_depth=7,

#                           min_child_weight=1,

#                           learning_rate=0.01,

#                           n_estimators=100,

#                           silent=True,

#                           objective='count:poisson',

#                           gamma=0,

#                           max_delta_step=0,

#                           subsample=1,

#                           colsample_bytree=1,

#                           colsample_bylevel=1,

#                           reg_alpha=0,

#                           reg_lambda=0,

#                           scale_pos_weight=1,

#                           seed=1,

#                           missing=None)



clf = DecisionTreeClassifier()



print('Model parameters set')
# Train the model



clf.fit(X_train,y_train);



print('Model trained')
t1 = time.time()

print('Model fitted. Time taken %i minutes and %3.1f seconds' % ((t1-t0)//60,(t1-t0)%60))
# These results will use the underlying probabilities from the model

test_results = y_test > 2.5



# This is the probability of more than 2 goals being scored in the match

probs = 1- np.sum(clf.predict_proba(X_test)[:,:3],axis=1) 



# This will be true if there is more than a 50% chance of 3 or more goals being scored in the match (even if the highest individual probability is for 2 or fewer goals)

preds = probs > 0.5 



accuracy = accuracy_score(test_results,preds)

fpr, tpr, threshold = roc_curve(test_results, preds)

roc_auc = auc(fpr, tpr)

precis = average_precision_score(test_results,preds)

print('The accuracy score is %3.2f%%' % (accuracy*100))

print('The AUC score is %3.2f' % roc_auc)

print('The average precision score is %3.2f' % precis)

min_prob = min(probs)

max_prob = max(probs)

pos_count = sum(preds)

count = len(preds)

print('The probabilities range from %3.2f%% to %3.2f%% and %i matches are assumed to have more than 2.5 goals (out of %i)' % (min_prob*100,max_prob*100,pos_count,count))
plt.figure(figsize=(8, 6))

plt.title('Receiver Operating Characteristic');

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc);

plt.legend(loc = 'lower right');

plt.plot([0, 1], [0, 1],'r--');

plt.xlim([0, 1]);

plt.ylim([0, 1]);

plt.ylabel('True Positive Rate');

plt.xlabel('False Positive Rate');
# This will be true if the most likely outcome is for 3 or more goals being scored in the match

preds2 = clf.predict(X_test) > 2.5



accuracy = accuracy_score(test_results,preds2)

fpr, tpr, threshold = roc_curve(test_results, preds2)

roc_auc = auc(fpr, tpr)

precis = average_precision_score(test_results,preds2)

print('The accuracy score is %3.2f%%' % (accuracy*100))

print('The AUC score is %3.2f' % roc_auc)

print('The average precision score is %3.2f' % precis)

pos_count = sum(preds2)

count = len(preds2)

print('%i matches are assumed to have more than 2.5 goals (out of %i)' % (pos_count,count))
plt.figure(figsize=(8, 6))

plt.title('Receiver Operating Characteristic');

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc);

plt.legend(loc = 'lower right');

plt.plot([0, 1], [0, 1],'r--');

plt.xlim([0, 1]);

plt.ylim([0, 1]);

plt.ylabel('True Positive Rate');

plt.xlabel('False Positive Rate');
plt.figure(figsize=(8, 6))

sns.distplot(probs[y_test>2.5],color='red',label='Matches with more than 2.5 goals')

sns.distplot(probs[y_test<2.5],color='green',label='Matches with less than 2.5 goals')

plt.legend();

plt.title('Comparing the prediction probabilities for different outcomes')

plt.xlabel('Probability of more than 2.5 goals');
goal_probs = np.sum(clf.predict_proba(X_test),axis=0)

#lambda_hat = np.mean(np.matmul(clf.predict_proba(X_test),np.array(range(len(goal_probs))).reshape(len(goal_probs),1)))

#lambda_hat_2 = np.mean(clf.predict(X_test))

#poisson_est = poisson.pmf(range(len(goal_probs)),lambda_hat) * count

#poisson_est_2 = poisson.pmf(range(len(goal_probs)),lambda_hat_2) * count





fig, ax = plt.subplots(1, 2, figsize=(16, 5));

ax[0].bar(range(len(goal_probs)),goal_probs)

#poisson_est = poisson.pmf(range(len(goal_probs)),lambda_hat) * count

ax[0].plot(y_test.value_counts().sort_index(),linewidth=3,color='gold');

ax[0].set_title('Expected frequency of scorelines (actual results overlaid)');

ax[0].set_xlabel('Number of goals per match');

ax[0].set_ylabel('Number of matches');



ax[1].hist(clf.predict(X_test),bins=len(goal_probs));

ax[1].plot(y_test.value_counts().sort_index(),linewidth=3,color='gold');

ax[1].set_title('Predicted frequency of scorelines (actual results overlaid)');

ax[1].set_xlabel('Number of goals per match');

ax[1].set_ylabel('Number of matches');



ax2 = plt.bar

#plt.x_lab('Goals per match')