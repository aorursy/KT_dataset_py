# Initial Imports

import pandas as pd

import numpy as np

import datetime



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
# read data

nrows = 50000

train = pd.read_csv('../input/data-science-bowl-2019/train.csv', nrows=nrows)

train_labels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')

test = pd.read_csv('../input/data-science-bowl-2019/test.csv', nrows=nrows)

sample_submission = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')
# select installation_ids that have taken an assessment

keep_id = train[train.type == 'Assessment'][['installation_id']].drop_duplicates()

train = pd.merge(train, keep_id, on='installation_id', how='inner')

del keep_id



# convert timestamp to datetime

train['timestamp'] = pd.to_datetime(train['timestamp'])



# filter the train dataset for values whose installation_id appears in train_labels

train = train[train['installation_id'].isin(list(train_labels['installation_id'].unique()))]
# convert timestamp to datetime

test['timestamp'] = pd.to_datetime(test['timestamp'])
# make a list with all the unique 'titles' from the train and test set

unique_train_title = set(train['title'].value_counts().index)

unique_test_title = set(test['title'].value_counts().index)

title_list = list(unique_train_title.union(unique_test_title))



del unique_train_title

del unique_test_title



# make a list with all the unique 'event_code' from the train and test set

unique_train_event_code = set(train['event_code'].value_counts().index)

unique_test_event_code = set(test['event_code'].value_counts().index)

event_code_list = list(unique_train_event_code.union(unique_test_event_code))



del unique_train_event_code

del unique_test_event_code



# encode titles

title_map = dict(zip(title_list, np.arange(len(title_list))))

title_labels = dict(zip(np.arange(len(title_list)), title_list))



train['title'] = train['title'].map(title_map)

test['title'] = test['title'].map(title_map)

#train_labels['title'] = train_labels['title'].map(title_map)



# write codes which mean win for each title

win_code = dict(zip(title_map.values(), (4100*np.ones(len(title_map))).astype('int')))



# then, it set one element, the 'Bird Measurer (Assessment)' as 4110

win_code[title_map['Bird Measurer (Assessment)']] = 4110



# function that convert the raw data into processed features

def get_data(user_sample, test_set=False):

    '''

    The user_sample is a DataFrame from train or test with the only one installation_id

    

    The test_set parameter is related with the labels processing, that is only requered

    if test_set=False

    '''

    # Constants and parameters declaration

    last_activity = 0

    user_activities_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}

    

    # new features: time spent in each activity

    time_spent_each_act = {actv: 0 for actv in title_list}

    event_code_count = {eve: 0 for eve in event_code_list}

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

        

        # get type of current session

        session_type = session['type'].iloc[0]

        # get type of activity of current session

        session_title = session['title'].iloc[0]

        

        # get current session time in seconds

        if session_type != 'Assessment':

            #how much time spent on current session

            time_spent = int(session['game_time'].iloc[-1] / 1000)

            #how much time spent on each activity

            time_spent_each_act[title_labels[session_title]] += time_spent

        

        # for each assessment, and only this kind of session, the features below are processed

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
# get_data function is applyed to each installation_id and added to the new_train list

new_train=[]



for i, (ins_id, user_sample) in enumerate(train.groupby('installation_id', sort=False)):

    # user_sample is a DataFrame that contains only one installation_id

    new_train+=get_data(user_sample)

    

new_train = pd.DataFrame(new_train)
#same for test dataset

new_test = []



for ins_id, user_sample in test.groupby('installation_id', sort=False):

    new_test.append(get_data(user_sample, test_set=True))

    

new_test = pd.DataFrame(new_test)
# create a list of the features

features = list(new_train.columns.values)

features.remove('accuracy_group')
# removes accuracy_group from the train data

X_train = new_train[features]

# create a variable to contain just the accuracy_group label of the train data

y_train = new_train['accuracy_group']

# remove accuracy_group from the test data

X_test = new_test[features]
#KNN = KNeighborsClassifier(n_neighbors=7)

#KNN.fit(X_train, y_train)

#y_pred = KNN.predict(X_test)
clf_gbc = GradientBoostingClassifier(random_state=42, n_estimators=100)

clf_gbc.fit(X_train, y_train)

y_pred = clf_gbc.predict(X_test)
submission = pd.DataFrame(sample_submission['installation_id'])

y_pred = pd.DataFrame({'accuracy_group':y_pred[:]})
submission = submission.join(y_pred)

print(submission)

submission['accuracy_group'] = submission['accuracy_group'].astype(int)

submission.head()
submission.to_csv('submission.csv', index=False)