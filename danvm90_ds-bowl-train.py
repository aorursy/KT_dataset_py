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
train = pd.read_csv("../input/data-science-bowl-2019/train.csv")

test = pd.read_csv("/kaggle/input/data-science-bowl-2019/test.csv")
train.shape
# If quick set to true, only uses 5% of training data. 

quick = False
# To save time, we will only use 5% of the installation_ids for preparing the model.



import random



if quick:

    print(quick)

    # Grab all unique item_nbr from items file

    f = train['installation_id'].unique()



    # Count the lines

    num_lines = f.size

    



    # Sample size - in this case ~5% of items

    size = int(num_lines / 20)



    # Grab a random subset of size size from f

    skip_idx = random.sample(list(f), size)

    print(len(skip_idx))



    # Filter to only include training data for the subset of items we want

    train = train[train['installation_id'].isin(skip_idx)]
train.shape
train['timestamp'] = pd.to_datetime(train['timestamp'])

test['timestamp'] = pd.to_datetime(test['timestamp'])
# Store the installation_ids separately so that we can extract the train and test data from our features DataFrame 

train_ids = train.installation_id

test_ids = test.installation_id
# Because the Bird Measurer assessment uses a distinct event code, let's separate it from the data

bird_measure_assess = train[train['title'] == "Bird Measurer (Assessment)"]



# Capture bird_meas assessment attempts

bird_measure_assess = bird_measure_assess[bird_measure_assess['event_code'] == 4110]
# Grab all assessment attempts that are not for bird measurer

train_assess = train[train['title'] != "Bird Measurer (Assessment)"]



train_assess = train_assess[train_assess['event_code'] == 4100]



# Some non-assessment activities have a 4100 code, ignore those

train_assess = train_assess[train_assess['type'] == 'Assessment']



# Append bird measure assessment attempts to have a log of all assessment attempts

train_assess = train_assess.append(bird_measure_assess)
# To get each child's performance on their last assessment, extract only the last assessment per installation_id

last_assessments = train_assess.sort_values(by="timestamp").drop_duplicates(subset=["installation_id"], keep="last")



# Grab the game_sessions of the last assessment events

last_game_sessions = last_assessments.game_session.unique()



# Extract all assessment completion attempts in the final session

last_assessment_sessions = train_assess[train_assess['game_session'].isin(last_game_sessions)]
# We want to grab all the last game_sessions of the test data. This will allow us to create features on all the test data except for the last assessment attempt

test_assess = test[test.type == 'Assessment']



last_test_assess = test_assess.sort_values(by='timestamp').drop_duplicates(subset=['installation_id'], keep='last')
last_test_assess.head()
# Add columns that store if the attempt was a Pass or a Fail

last_assessment_sessions['Pass'] = last_assessment_sessions.apply(lambda row: '"correct":true' in row.event_data, axis = 1)



last_assessment_sessions['Fail'] = last_assessment_sessions.apply(lambda row: '"correct":false' in row.event_data, axis = 1)



# Create a pivot table that logs pass/fail for each installation_id for the last assessment

pass_fail_log = last_assessment_sessions.pivot_table(['Pass', 'Fail'], index='installation_id', aggfunc = 'sum')
# Using the rules provided in the Data tab of the competition, create a function that outputs the accuracy group

def accuracy_group_calculator(row):

    

    if row.Pass == 0:

        #0: the assessment was never solved

        return 0.0

    

    elif row.Fail == 0:

        # 3: the assessment was solved on the first attempt

        return 3.0

    

    elif row.Fail == 1:

        # 2: the assessment was solved on the second attempt

        return 2.0

    

    else:

        # 1: the assessment was solved after 3 or more attempts

        return 1.0
# We can finally calculate the target variable for our training data set

pass_fail_log['Accuracy_Group'] = pass_fail_log.apply(lambda row: accuracy_group_calculator(row), axis = 1)



train_targets = pass_fail_log.drop(['Fail', 'Pass'], axis=1)



train_targets.head()
# Combine the training and test data so that we can build the same features on both

data = train.append(test)



# We will build the features of the train and test data at the same time

features = train_targets.index.values

features = np.concatenate([features, test.installation_id.unique()])

features = pd.DataFrame(features, index=features)

features.index.name = 'installation_id'
last_assessment_sessions.drop(['Pass', 'Fail'], axis=1, inplace=True)



last_assessments = last_assessment_sessions.append(last_test_assess)
last_game_sessions = last_assessments.game_session.unique()
# Join the game_session, title, and world to the features dataframe.

features = pd.merge(features, last_assessments[['installation_id', 'game_session', 'title', 'world']], on='installation_id', how='left')



# Remove any duplicate lines created by the merge.

features = features.drop_duplicates(subset=['installation_id'], keep="first")
train_targets['Accuracy_Group'].value_counts()
train_targets['Accuracy_Group'].value_counts().sort_index()
train_targets.head()
train_targets.shape
features.head()
features.rename(columns = {'game_session':'last_assess_game_session', 'title': 'last_assess_title', 'world':'last_assess_world'}, inplace = True) 
features.drop([0], axis=1, inplace=True)
# Join the game_session, title, and world to the features dataframe.

train_targets = pd.merge(train_targets, features[['installation_id', 'last_assess_game_session', 'last_assess_title', 'last_assess_world']], on='installation_id', how='left')
assessments = train_targets.last_assess_title.unique()

worlds = train_targets.last_assess_world.unique()
worlds
assessments
import matplotlib.pyplot as plt





for i in range(1, 5):

    if i == 1:

        counts = train_targets['Accuracy_Group'].value_counts().sort_index()

        plt.subplot(2, 2, i).set_title("All Worlds")

    

    if i == 2:

        counts = train_targets[train_targets['last_assess_world'] == 'MAGMAPEAK']['Accuracy_Group'].value_counts().sort_index()

        plt.subplot(2, 2, i).set_title("MAGMAPEAK")

        

    if i == 3:

        counts = train_targets[train_targets['last_assess_world'] == 'TREETOPCITY']['Accuracy_Group'].value_counts().sort_index()

        plt.subplot(2, 2, i).set_title("TREETOPCITY")

    

    if i == 4:

        counts = train_targets[train_targets['last_assess_world'] == 'CRYSTALCAVES']['Accuracy_Group'].value_counts().sort_index()

        plt.subplot(2, 2, i).set_title("CRYSTALCAVES")

    

    plt.bar(counts.index,counts)

    plt.tight_layout()
train_targets.pivot_table(['last_assess_game_session'], index='last_assess_title', columns = ['last_assess_world'], aggfunc = 'count')
for i in range(1, 7):

    if i == 1:

        counts = train_targets['Accuracy_Group'].value_counts().sort_index()

        plt.subplot(3, 2, i).set_title("All Assessments")

    

    if i == 2:

        counts = train_targets[train_targets['last_assess_title'] == 'Mushroom Sorter (Assessment)']['Accuracy_Group'].value_counts().sort_index()

        plt.subplot(3, 2, i).set_title('Mushroom Sorter (Assessment)')

               

    if i == 3:

        counts = train_targets[train_targets['last_assess_title'] == 'Bird Measurer (Assessment)']['Accuracy_Group'].value_counts().sort_index()

        plt.subplot(3, 2, i).set_title('Bird Measurer (Assessment)')

              

    if i == 4:

        counts = train_targets[train_targets['last_assess_title'] == 'Cauldron Filler (Assessment)']['Accuracy_Group'].value_counts().sort_index()

        plt.subplot(3, 2, i).set_title('Cauldron Filler (Assessment)')

     

    if i == 5:

        counts = train_targets[train_targets['last_assess_title'] == 'Cart Balancer (Assessment)']['Accuracy_Group'].value_counts().sort_index()

        plt.subplot(3, 2, i).set_title('Cart Balancer (Assessment)')

        

    if i == 6:

        counts = train_targets[train_targets['last_assess_title'] == 'Chest Sorter (Assessment)']['Accuracy_Group'].value_counts().sort_index()

        plt.subplot(3, 2, i).set_title('Chest Sorter (Assessment)')

    

    

    plt.bar(counts.index,counts)

    plt.tight_layout()
# Because the Bird Measurer assessment uses a distinct event code, let's separate it from the data

bird_measure_assess = data[data['title'] == "Bird Measurer (Assessment)"]



# Capture bird_meas assessment attempts

bird_measure_assess = bird_measure_assess[bird_measure_assess['event_code'] == 4110]
# Grab all assessment attempts that are not for bird measurer

prior_assessments = data[data['title'] != "Bird Measurer (Assessment)"]



prior_assessments = prior_assessments[prior_assessments['event_code'] == 4100]



# Some non-assessment activities have a 4100 code, ignore those

prior_assessments = prior_assessments[prior_assessments['type'] == 'Assessment']



# Append bird measure assessment attempts to have a log of all assessment attempts

prior_assessments = prior_assessments.append(bird_measure_assess)
# Remove all assessment sessions that were part of the child's last assessment. That way, our features don't leak information about the target variable. 

prior_assessments = prior_assessments[~prior_assessments['game_session'].isin(last_game_sessions)]
prior_assessments.shape
# Create a Pass and Fail column that will allow us to count each user's performance history

prior_assessments['Pass'] = prior_assessments.apply(lambda row: '"correct":true' in row.event_data, axis = 1)



prior_assessments['Fail'] = prior_assessments.apply(lambda row: '"correct":false' in row.event_data, axis = 1)
prior_assessments.head()
# Create a pivot table that logs pass/fail for each installation_id for the last assessment

prior_pass_fail_log = prior_assessments.pivot_table(['Pass', 'Fail'], index=['installation_id','title',], aggfunc = 'sum')
prior_pass_fail_log.head(10)
titles = prior_assessments.title.unique()
# For each assessment title, filter on that title and pull each user's history into the features dataframe we're constructing.

for title in titles:

    features = pd.merge(features, prior_pass_fail_log.filter(like=title, axis=0), on="installation_id", how='left')

    features.rename(columns = {'Fail': title + "_Fail", 'Pass': title + "_Pass"}, inplace = True) 
# If the user has no attempt history for an assessment, enter a 0 for both pass and fail

features.fillna(value = 0, inplace=True)
assessments = data[data['type'] == 'Assessment']
assessments.title.unique()
# Because the Bird Measurer assessment uses a distinct event code, let's separate it from the data

bird_measure_assess = data[data['title'] == "Bird Measurer (Assessment)"]



measure_assess = data[data['title'] != "Bird Measurer (Assessment)"]

measure_assess = measure_assess[measure_assess.type == 'Assessment']



# Capture assessment attempts

game_sessions_start = bird_measure_assess[bird_measure_assess['event_code'] == 2000].game_session.unique()



game_sessions_start = np.concatenate([game_sessions_start, measure_assess[measure_assess['event_code'] == 2000].game_session.unique()])



# Capture completed assessment game sessions

game_sessions_complete = bird_measure_assess[bird_measure_assess.event_code == 4110].game_session.unique()



game_sessions_complete = np.concatenate([game_sessions_complete, measure_assess[measure_assess['event_code'] == 4100].game_session.unique()])



# Capture all game_sessions where an assessment is initiated but never completed

incomplete_gs = [gs for gs in game_sessions_start if gs not in game_sessions_complete]
# Capture the train data for the game_sessions that have incomplete assessments

incomplete_sessions = data[data.game_session.isin(incomplete_gs)]

incomplete_sessions = incomplete_sessions[incomplete_sessions.event_code == 2000]
incomplete_sessions.head()
incomplete_sessions_table = incomplete_sessions.pivot_table(['event_count'],columns=incomplete_sessions.title, index='installation_id', aggfunc = 'count', fill_value = 0)
incomplete_sessions_table.columns = [str(col) + '_incomplete_attempts' for col in incomplete_sessions_table.columns]
incomplete_sessions_table.tail()
features = pd.merge(features, incomplete_sessions_table, on='installation_id', how='left')

features.head()
# To get each child's performance on their last assessment, extract only the last assessment per installation_id

last_event_per_game_session = data.sort_values(by="timestamp").drop_duplicates(subset=["game_session"], keep="last")
last_event_per_game_session1 = last_event_per_game_session.copy()
last_event_per_game_session = last_event_per_game_session[last_event_per_game_session.type != "Assessment"]
time_per_activity = last_event_per_game_session.groupby(['installation_id','world'])[['game_time']].sum()
time_per_activity
worlds = data.world.unique()
worlds.shape
features1 = features.copy()
# For each assessment title, filter on that title and pull each user's history into the features dataframe we're constructing.

for world in worlds:

    features1 = pd.merge(features1, time_per_activity.filter(like=world, axis=0), on="installation_id", how='left')

    #print(features1.columns)

    features1.rename(columns = {'game_time': world + "_game_time"}, inplace = True) 
features1
features1.fillna(value = 0, inplace=True)
columns_to_drop = []

for col in features1.columns:

    if features1[col].unique().size == 1:

        columns_to_drop.append(col)
columns_to_drop
features1.drop(columns_to_drop, axis=1, inplace=True)
features = features1.copy()
features.head()
features = pd.concat([features, pd.get_dummies(features['last_assess_title'],prefix='last_title', drop_first=True)], axis=1)
features.drop(labels=["last_assess_game_session", "last_assess_title", "last_assess_world"], inplace=True, axis=1)
train_targets.head()
train_targets = train_targets.iloc[:, :2]
# Join the game_session, title, and world to the features dataframe.

train_targets = pd.merge(train_targets, features, on='installation_id', how='left')
sample_sub = pd.read_csv("/kaggle/input/data-science-bowl-2019/sample_submission.csv")
# Join the game_session, title, and world to the features dataframe.

test_X = pd.merge(sample_sub, features, on='installation_id', how='left')
test_X
train_X = train_targets.iloc[:, 2:]
train_X.head()
train_y = train_targets.iloc[:, 1]
train_y.head()
test_X_input =  test_X.iloc[:, 2:]
test_X_input.head()
train_X.fillna(0, inplace=True, axis=1)

test_X_input.fillna(0, inplace=True, axis=1)
from sklearn import linear_model, metrics 

from sklearn.naive_bayes import MultinomialNB

   

# defining feature matrix(X) and response vector(y) 

X = train_X

y = train_y

  

# splitting X and y into training and testing sets 

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 

                                                    random_state=1) 

   

# create logistic regression object 

#reg = linear_model.LogisticRegression() 

nb = MultinomialNB()  

# train the model using the training sets 

#reg.fit(X_train, y_train) 

nb.fit(X_train, y_train)

  

# making predictions on the testing set 

#y_pred = reg.predict(X_test) 

y_pred = nb.predict(X_test)

   

# comparing actual response values (y_test) with predicted response values (y_pred) 

print("Logistic Regression model accuracy(in %):",  

metrics.accuracy_score(y_test, y_pred)*100) 
sample_sub.drop(['accuracy_group'], axis=1, inplace=True)
#test_y = reg.predict(test_X_input)

test_y = nb.predict(test_X_input)
test_y_df = pd.DataFrame(test_y)
test_y_df.rename(columns={0: 'accuracy_group'}, inplace=True)
test_y_df.head()
sample_sub
output = pd.merge(sample_sub, test_y_df, left_index=True, right_index=True)
output
output['accuracy_group'] = output['accuracy_group'].astype('int32')
output.to_csv("submission.csv", index=False)