import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
original_df = pd.read_csv('../input/Speed Dating Data.csv', encoding="ISO-8859-1")

original_df.head()
original_df.size
# remove some fields we dont want to make data more realistic for us
gj_df = original_df.drop(['id', 'gender', 'idg', 'condtn', 'wave', 'round', 'position', 'positin1', 'order', 'partner', 'int_corr', 'samerace', 'age_o', 'race_o', 'pf_o_att','pf_o_sin','pf_o_int','pf_o_fun','pf_o_amb','pf_o_sha','dec_o','attr_o','sinc_o','intel_o','fun_o','amb_o','shar_o','like_o','prob_o','met_o', 'field', 'field_cd', 'mn_sat', 'undergra', 'income', 'tuition', 'race', 'imprace', 'imprelig', 'zipcode', 'goal','date', 'go_out', 'career', 'career_c'], axis=1)

# remove big chunk of unwanted fields at the end 
gj_df = gj_df.loc[:, :'dining']

# and preview for first user
gj_df.head()
# lets tidy up the data and make it closer to what we want

# remove NaNs
gj_df = gj_df.fillna(0)

# convert number fields to integers
convert_to_int_ignore = ['id', 'match', 'from', 'income']
for column in gj_df:  
    if (column not in convert_to_int_ignore):
        gj_df[column] = gj_df[column].astype(int)
        
# convert 'rating' fields to categoric, it shouldnt affect the results
def get_category(val, num_cats):
    return val % num_cats
    
def rating_to_categoric(column, num_cats):
    return column.apply(lambda row_val: rating_to_categoric(row_val, num_cats))

# give one column 5 categories (represents game) and the rest 4
gj_df['sports'] = gj_df['sports'].apply(lambda value: get_category(value, 5))
gj_df['tvsports'] = gj_df['tvsports'].apply(lambda value: get_category(value, 4))
gj_df['exercise'] = gj_df['exercise'].apply(lambda value: get_category(value, 4))
gj_df['dining'] = gj_df['dining'].apply(lambda value: get_category(value, 4))
        
gj_df.head()
# rename the columns for our uses
gj_df.rename(columns={'iid': 'uid', 'from': 'cityID','sports': 'gameID', 'tvsports': 'competitive', 'exercise': 'personality', 'dining': 'skill'}, inplace=True)

gj_df.head()
# create mock matches df for our matches db

matches_db_df = gj_df.loc[:, :'match']

matches_db_df.head()
# create a mock user df representing what we'd have on the front end

user_df = gj_df.drop(['pid', 'match'], axis=1)
user_df.drop_duplicates('uid', inplace=True)
user_df.reset_index(drop=True, inplace=True)

user_df.head()
# assuming we have just 2 locations to start (so each user should have around a min of 5 tested matches), 
# set locationIDs randomly...
import random

user_df['cityID'] = user_df['cityID'].apply(lambda value: random.randint(1,2))
# take a look at final user db for seeding

user_df.head(15)
# and the final matches db

matches_db_df.head(15)
# export these two for seeding databases

matches_db_df.to_csv('matches_db_seed.csv')
user_df.to_csv('user_db_seed.csv')
# add a basic 'filter' matching algorithm 

# define parameters
age_range = 10
matching_interests = 3

def is_basic_match(user1, user2):
    
    # location match
    if user1['cityID'] != user2['cityID']:
        return False
    
    # age match
    if abs(user1['age'] - user2['age']) > age_range:
        return False
    
    # interests match
    match_count = 0
    for cat_column in user1['gameID':].index:
        if user1[cat_column] == user2[cat_column]:
            match_count += 1
            
    if match_count < matching_interests:
        return False
    
    return True

def get_basic_matches_for_user_by_id(user_id):
    user1 = user_df.loc[user_id]
    match_mask = user_df.apply(lambda user2: is_basic_match(user1, user2), axis=1)
    return user_df[match_mask]

get_basic_matches_for_user_by_id(2)
# let's create a mockup 'matches' df from the gj_df now we have users to match
# and add some extra featutes

matches_df = matches_db_df
# create matches data frame from the iid, pid and match rating of the users
# NaN means they haven't been matches together yet
matches_df = matches_df.pivot(index='pid', columns='uid', values='match')

# preview the matches df for the first 20 users
matches_df.loc[0:20, 0:20]
# transform our categorical variables to dummy variables

user_df_dummies = pd.get_dummies(
    user_df.loc[:, 'gameID':], 
    prefix=['gameID','comp','pers','skill'],
    columns=['gameID', 'competitive', 'personality', 'skill']
)

user_df_dummies.head(10)
# replace categoric columns with dummy vars

user_df = user_df.drop(['gameID', 'competitive', 'personality', 'skill'], axis=1)
user_df = user_df.join(user_df_dummies)

user_df.head()
# now append the user attributes for each pid

train_df = user_df.join(matches_df, on='uid')

train_df.head()
# rearrange columns moving user df columns to the end

move_to_end = user_df.columns.tolist()
move_to_end.remove('uid')

cols = train_df.columns.tolist()
cols = [col for col in cols if col not in move_to_end]
cols = cols + move_to_end
train_df = train_df[cols]

# and remove non-categoric variables for now, we'll just assume we had enough data to filter on these
train_df = train_df.drop(['age','cityID'], axis=1)

train_df.head(10)
def get_train_df_for_user_by_id(id):
    train_cols = train_df.columns.tolist()
    user_cols = user_df.columns.tolist()
    cols = [col for col in train_cols if col in user_cols or col == id]
    user_train_df = train_df[cols]
    user_train_df = user_train_df.dropna()
    user_train_df = user_train_df.rename(columns={id: 'match'})
    user_train_df = user_train_df.set_index('uid')
    return user_train_df
# this is what we end up with to train a model for user with iid 206
# 206 is a nice example because we have 20 reted matches with a good mix of 1 and 0

get_train_df_for_user_by_id(208)
# train a logistic regression model 
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics

user_train_df = get_train_df_for_user_by_id(206)
Y = user_train_df.loc[:, 'match']
X = user_train_df.drop(['match'], axis=1)

user_train_X, user_test_X, user_train_Y, user_test_Y = model_selection.train_test_split(X, Y, test_size=0.5)


logistic = linear_model.LogisticRegression()
logistic.fit(user_train_X, user_train_Y)

predictions = logistic.predict(user_test_X)
print(predictions)
print(user_test_Y)
print(metrics.accuracy_score(user_test_Y, predictions))

