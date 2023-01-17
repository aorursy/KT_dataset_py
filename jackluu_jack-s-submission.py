import numpy as np

import pandas as pd

import datetime

from catboost import CatBoostClassifier

from time import time

from tqdm import tqdm_notebook as tqdm

from collections import Counter

from scipy import stats
from sklearn.metrics import confusion_matrix

# this function is the quadratic weighted kappa (the metric used for the competition submission)

def qwk(act,pred,n=4,hist_range=(0,3)):

    

    # Calculate the percent each class was tagged each label

    O = confusion_matrix(act,pred)

    # normalize to sum 1

    O = np.divide(O,np.sum(O))

    

    # create a new matrix of zeroes that match the size of the confusion matrix

    # this matriz looks as a weight matrix that give more weight to the corrects

    W = np.zeros((n,n))

    for i in range(n):

        for j in range(n):

            # makes a weird matrix that is bigger in the corners top-right and botton-left (= 1)

            W[i][j] = ((i-j)**2)/((n-1)**2)

            

    # make two histograms of the categories real X prediction

    act_hist = np.histogram(act,bins=n,range=hist_range)[0]

    prd_hist = np.histogram(pred,bins=n,range=hist_range)[0]

    

    # multiply the two histograms using outer product

    E = np.outer(act_hist,prd_hist)

    E = np.divide(E,np.sum(E)) # normalize to sum 1

    

    # apply the weights to the confusion matrix

    num = np.sum(np.multiply(W,O))

    # apply the weights to the histograms

    den = np.sum(np.multiply(W,E))

    

    return 1-np.divide(num,den)

    
train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')

train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')

specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')

test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')

submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')
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
# convert text into datetime

train['timestamp'] = pd.to_datetime(train['timestamp'])

test['timestamp'] = pd.to_datetime(test['timestamp'])
train.head()
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
# here the get_data function is applyed to each installation_id and added to the compile_data list

compiled_data = []

# tqdm is the library that draws the status bar below

for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort=False)), total=17000):

    # user_sample is a DataFrame that contains only one installation_id

    compiled_data += get_data(user_sample)
# the compiled_data is converted to DataFrame and deleted to save memmory

new_train = pd.DataFrame(compiled_data)

del compiled_data

new_train.shape
pd.set_option('display.max_columns', None)

new_train[:10]
# this list comprehension create the list of features that will be used on the input dataset X

# all but accuracy_group, that is the label y

all_features = [x for x in new_train.columns if x not in ['accuracy_group']]

# this cat_feature must be declared to pass later as parameter to fit the model

cat_features = ['session_title']

# here the dataset select the features and split the input ant the labels

X, y = new_train[all_features], new_train['accuracy_group']

del train

X.shape
# this function makes the model and sets the parameters

# https://catboost.ai/docs/concepts/python-reference_catboostclassifier.html

def make_classifier():

    clf = CatBoostClassifier(

                                 

                            loss_function='MultiClass',

                            eval_metric="WKappa",

                            task_type="GPU",

                            thread_count=-1,

                            od_type="Iter",

                            early_stopping_rounds=500,

                            random_seed=42,

                            border_count=110,

                            l2_leaf_reg=7,

                            iterations=1800,

                            learning_rate=0.2,

                            depth=5

    )

        

    return clf

# CV

from sklearn.model_selection import KFold

oof = np.zeros(len(X))

NFOLDS = 7

folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=2019)

training_start_time = time()

models = []

for fold, (trn_idx, test_idx) in enumerate(folds.split(X, y)):

    start_time = time()

    print(f'Training on fold {fold+1}')

    clf = make_classifier()

    clf.fit(X.loc[trn_idx, all_features], y.loc[trn_idx], eval_set=(X.loc[test_idx, all_features], y.loc[test_idx]),

                          use_best_model=True, verbose=500, cat_features=cat_features)

    

    oof[test_idx] = clf.predict(X.loc[test_idx, all_features]).reshape(len(test_idx))

    models.append(clf)

    print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))

    print('____________________________________________________________________________________________\n')

    #break

    

print('-' * 30)

print('OOF QWK:', qwk(y, oof))

print('-' * 30)
# train model on all data once

#clf = make_classifier()

#clf.fit(X, y, verbose=500, cat_features=cat_features)



del X, y
new_test = []

for ins_id, user_sample in tqdm(test.groupby('installation_id', sort=False), total=1000):

    a = get_data(user_sample, test_set=True)

    new_test.append(a)

    

X_test = pd.DataFrame(new_test)

del test
predictions = []

for model in models:

    predictions.append(model.predict(X_test))

predictions = np.concatenate(predictions, axis=1)

print(predictions.shape)

predictions = stats.mode(predictions, axis=1)[0].reshape(-1)

print(predictions.shape)

submission['accuracy_group'] = np.round(predictions).astype('int')

submission.to_csv('submission.csv', index=None)



submission.head()
from IPython.display import FileLink

FileLink('submission.csv')
submission['accuracy_group'].plot(kind='hist')
train_labels['accuracy_group'].plot(kind='hist')
pd.Series(oof).plot(kind='hist')