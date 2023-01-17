# Initial Imports

import pandas as pd

pd.set_option('display.max_columns', None)

import numpy as np

import seaborn as sns

import matplotlib as plt

%matplotlib inline

import calendar

import warnings

warnings.filterwarnings("ignore")

import datetime

from time import time

from tqdm import tqdm_notebook as tqdm

from collections import Counter

from scipy import stats



from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn import linear_model

from sklearn.linear_model import RidgeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import cohen_kappa_score, make_scorer

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import AdaBoostClassifier
%%time

# load the data

train = pd.read_csv('../input/data-science-bowl-2019/train.csv')

train_labels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')

test = pd.read_csv('../input/data-science-bowl-2019/test.csv')

specs = pd.read_csv('../input/data-science-bowl-2019/specs.csv')

sample_submission = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')
train.head()
train.info()
train.describe()
#  From Erik Bruin on Kaggle

# filter to only include installation_ids that have taken an assessment

keep_id = train[train.type == 'Assessment'][['installation_id']].drop_duplicates()

train = pd.merge(train, keep_id, on='installation_id', how='inner')

del keep_id

train.shape
train['installation_id'].nunique()
# reassign the column with a conversion of its value to dateitme

train['timestamp'] = pd.to_datetime(train['timestamp'])
train.info()
train['event_data'].head()
train['event_data'][0]
train['event_data'][2]
train['event_data'][3]
train['event_data'][4]
train['title'].value_counts()
train['type'].value_counts()
train['event_code'].value_counts()
train['timestamp'].plot()
train['world'].value_counts()
train_labels.head()
train_labels.info()
train_labels['installation_id'].nunique()
# create a list of unique installation ids in train_labels

unique_ids_train_labels = list(train_labels['installation_id'].unique())

# filter the train dataset for values whose installation_id appears in train_labels

train = train[train['installation_id'].isin(unique_ids_train_labels)]

# delete unique_ids_train_labels to save memory

del unique_ids_train_labels

# check the number of unique installation_ids in train

train['installation_id'].nunique()
train.shape
train_labels['game_session'].value_counts()
train_labels['installation_id'].value_counts()
installation_id_attempts = train_labels['installation_id'].value_counts()

installation_id_attempts.plot();
train_labels.describe()
train_labels['accuracy_group'].value_counts()
test.head()
test.info()
test.describe()
# deal with timestamp

test['timestamp'] = pd.to_datetime(test['timestamp'])

test.info()
test['installation_id'].nunique()
test[test.type == 'Assessment'][['installation_id']].nunique()
sample_submission.info()
# how many installation_ids are unique

sample_submission['installation_id'].nunique()
# are the installation_ids in test and sample_submission the same

len(set.intersection(set(sample_submission['installation_id']), set(test['installation_id'])))
# value_counts of accuracy_group

sample_submission['accuracy_group'].value_counts()
sample_submission.head()
specs.head()
specs.info()
specs['info'][0]
specs['info'][1]
specs['args'][0]
specs['args'][1]
# delete specs

del specs
# From Erik Bruin

#Credits go to Massoud Hosseinali



# encode title

# make a list with all the unique 'titles' from the train and test set

list_of_user_activities = list(set(train['title'].value_counts().index).union(set(test['title'].value_counts().index)))

# make a list with all the unique 'event_code' from the train and test set

list_of_event_code = list(set(train['event_code'].value_counts().index).union(set(test['event_code'].value_counts().index)))

# create a dictionary numerating the titles

activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))

activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))



# replace the text titles withing the number titles from the dict

train['title'] = train['title'].map(activities_map)

test['title'] = test['title'].map(activities_map)

train_labels['title'] = train_labels['title'].map(activities_map)



# I didnt undestud why, but this one makes a dict where the value of each element is 4100 

win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))

# then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest

win_code[activities_map['Bird Measurer (Assessment)']] = 4110



# this is the function that convert the raw data into processed features

def get_data(user_sample, test_set=False):

    '''

    The user_sample is a DataFrame from train or test where the only one 

    installation_id is filtered

    And the test_set parameter is related with the labels processing, that is only requered

    if test_set=False

    '''

    # Constants and parameters declaration

    last_activity = 0

    user_activities_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}

    

    # news features: time spent in each activity

    time_spent_each_act = {actv: 0 for actv in list_of_user_activities}

    event_code_count = {eve: 0 for eve in list_of_event_code}

    last_session_time_sec = 0

    

    accuracy_groups = {0:0, 1:0, 2:0, 3:0}

    all_assessments = []

    accumulated_accuracy_group = 0

    accumulated_accuracy=0

    accumulated_correct_attempts = 0 

    accumulated_uncorrect_attempts = 0 

    accumulated_actions = 0

    counter = 0

    time_first_activity = float(user_sample['timestamp'].values[0])

    durations = []

    

    # itarates through each session of one instalation_id

    for i, session in user_sample.groupby('game_session', sort=False):

        # i = game_session_id

        # session is a DataFrame that contain only one game_session

        

        # get some sessions information

        session_type = session['type'].iloc[0]

        session_title = session['title'].iloc[0]

        

        # get current session time in seconds

        if session_type != 'Assessment':

            time_spent = int(session['game_time'].iloc[-1] / 1000)

            time_spent_each_act[activities_labels[session_title]] += time_spent

        

        # for each assessment, and only this kind off session, the features below are processed

        # and a register are generated

        if (session_type == 'Assessment') & (test_set or len(session)>1):

            # search for event_code 4100, that represents the assessments trial

            all_attempts = session.query(f'event_code == {win_code[session_title]}')

            # then, check the numbers of wins and the number of losses

            true_attempts = all_attempts['event_data'].str.contains('true').sum()

            false_attempts = all_attempts['event_data'].str.contains('false').sum()

            # copy a dict to use as feature template, it's initialized with some itens: 

            # {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}

            features = user_activities_count.copy()

            features.update(time_spent_each_act.copy())

            features.update(event_code_count.copy())

            # add title as feature, remembering that title represents the name of the game

            features['session_title'] = session['title'].iloc[0] 

            # the 4 lines below add the feature of the history of the trials of this player

            # this is based on the all time attempts so far, at the moment of this assessment

            features['accumulated_correct_attempts'] = accumulated_correct_attempts

            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts

            accumulated_correct_attempts += true_attempts 

            accumulated_uncorrect_attempts += false_attempts

            # the time spent in the app so far

            if durations == []:

                features['duration_mean'] = 0

            else:

                features['duration_mean'] = np.mean(durations)

            durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)

            # the accurace is the all time wins divided by the all time attempts

            features['accumulated_accuracy'] = accumulated_accuracy/counter if counter > 0 else 0

            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0

            accumulated_accuracy += accuracy

            # a feature of the current accuracy categorized

            # it is a counter of how many times this player was in each accuracy group

            if accuracy == 0:

                features['accuracy_group'] = 0

            elif accuracy == 1:

                features['accuracy_group'] = 3

            elif accuracy == 0.5:

                features['accuracy_group'] = 2

            else:

                features['accuracy_group'] = 1

            features.update(accuracy_groups)

            accuracy_groups[features['accuracy_group']] += 1

            # mean of the all accuracy groups of this player

            features['accumulated_accuracy_group'] = accumulated_accuracy_group/counter if counter > 0 else 0

            accumulated_accuracy_group += features['accuracy_group']

            # how many actions the player has done so far, it is initialized as 0 and updated some lines below

            features['accumulated_actions'] = accumulated_actions

            

            # there are some conditions to allow this features to be inserted in the datasets

            # if it's a test set, all sessions belong to the final dataset

            # it it's a train, needs to be passed throught this clausule: session.query(f'event_code == {win_code[session_title]}')

            # that means, must exist an event_code 4100 or 4110

            if test_set:

                all_assessments.append(features)

            elif true_attempts+false_attempts > 0:

                all_assessments.append(features)

                

            counter += 1

        

        # this piece counts how many actions was made in each event_code so far

        n_of_event_codes = Counter(session['event_code'])

        

        for key in n_of_event_codes.keys():

            event_code_count[key] += n_of_event_codes[key]



        # counts how many actions the player has done so far, used in the feature of the same name

        accumulated_actions += len(session)

        if last_activity != session_type:

            user_activities_count[session_type] += 1

            last_activitiy = session_type

    # if it't the test_set, only the last assessment must be predicted, the previous are scraped

    if test_set:

        return all_assessments[-1]

    # in the train_set, all assessments goes to the dataset

    return all_assessments
# From Erik Bruin

#Credits go to Massoud Hosseinali

# here the get_data function is applyed to each installation_id and added to the compile_data list

compiled_data = []

# tqdm is the library that draws the status bar below

for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort=False)), total=train.installation_id.nunique()):

    # user_sample is a DataFrame that contains only one installation_id

    compiled_data += get_data(user_sample)
# From Erik Bruin

#Credits go to Massoud Hosseinali



# the compiled_data is converted to DataFrame and deleted to save memmory

new_train = pd.DataFrame(compiled_data)



del compiled_data

new_train.shape
# From Erik Bruin

# modified by rahim

new_test = []



for ins_id, user_sample in tqdm(test.groupby('installation_id', sort=False), total=1000):

    a = get_data(user_sample, test_set=True)

    new_test.append(a)



    

new_test = pd.DataFrame(new_test)
new_train.head()
new_test.head()
list(new_train.columns.values)
# create a list of the features

features = list(new_train.columns.values)

features.remove('accuracy_group')

len(features)
# removes accuracy_group from the train data

X_train = new_train[features]

# create a variable to contain just the accuracy_group label of the train data

y_train = new_train['accuracy_group']

# remove accuracy_group from the test data

X_test = new_test[features]
X_train.head()
X_test.head()
y_train.head()
clf_gbc = GradientBoostingClassifier(random_state=42, n_estimators=100)

clf_gbc.fit(X_train, y_train)

y_pred = clf_gbc.predict(X_test)
y_pred
type(y_pred)
submission = pd.DataFrame(sample_submission['installation_id'])

y_pred = pd.DataFrame({'accuracy_group':y_pred[:]})
submission = submission.join(y_pred)

submission['accuracy_group'] = submission['accuracy_group'].astype(int)

submission.head()
submission.to_csv('submission.csv', index=False)