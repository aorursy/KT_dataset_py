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



os.chdir('/kaggle/input/data-science-bowl-2019/')

os.getcwd()



# Any results you write to the current directory are saved as output.
import os

import pandas as pd

pd.set_option('display.max_columns', None)

import numpy as np

import seaborn as sns

import matplotlib.style as style

style.use('fivethirtyeight')

import matplotlib.pylab as plt

import calendar

import warnings

warnings.filterwarnings("ignore")

import datetime

from time import time

from tqdm import tqdm_notebook as tqdm

from collections import Counter

from scipy import stats

import random

seed = 1234

random.seed(seed)

np.random.seed(seed)

import collections

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

plt.style.use('seaborn')

sns.set(font_scale=1.8)

import random

from sklearn import metrics

import os

import warnings

warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score, cross_validate

from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from statistics import mean
# read datasets

raw_train = pd.read_csv("train.csv")

specs = pd.read_csv("specs.csv")

raw_labels = pd.read_csv("train_labels.csv")

test = pd.read_csv("test.csv")

submission = pd.read_csv("sample_submission.csv")
raw_train.head()
raw_labels.head()
raw_train.info()
raw_labels.info()
correct_count = raw_labels['num_correct'].sum()

incorrect_count = raw_labels['num_incorrect'].sum()

print(f'According to the count of correct and incorrect attempts within the labels dataset, the number of assessment records within the training set should be {correct_count + incorrect_count}')
raw_train1 = raw_train[((raw_train['event_code']==4100) | (raw_train['event_code'] == 4110)) & (raw_train['type'] == 'Assessment')]

raw_train2 = raw_train1[((raw_train1['event_code'] == 4100) & (raw_train1['title'] != 'Bird Measurer (Assessment)')) | (raw_train1['event_code'] == 4110)]

raw_train2.info()
train_correct = raw_train2[raw_train2['event_data'].str.contains("true")].shape[0]

train_incorrect = raw_train2[raw_train2['event_data'].str.contains("false")].shape[0]

print (f'The number of correct attempts in the training set is {train_correct}. The number of correct attempts in the labeling dataset is {correct_count}.')

print (f'The number of correct attempts in the training set is {train_incorrect}. The number of correct attempts in the labeling dataset is {incorrect_count}.')
train_4 = raw_train[((raw_train['event_code'] == 4100) & (raw_train['title'] != 'Bird Measurer (Assessment)')) | (raw_train['event_code'] != 4100)]

train = train_4[train_4.installation_id.isin(raw_labels.installation_id.unique())]
train['timestamp'] = pd.to_datetime(train['timestamp'])

test['timestamp'] = pd.to_datetime(test['timestamp'])
train_labels = raw_labels
def encode_title(train, test, train_labels):

    # encode title

    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))

    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))

    all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))

    # make a list with all the unique 'titles' from the train and test set

    list_of_user_activities = list(set(train['title'].unique()).union(set(test['title'].unique())))

    # make a list with all the unique 'event_code' from the train and test set

    list_of_event_code = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))

    list_of_event_id = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))

    # make a list with all the unique worlds from the train and test set

    list_of_worlds = list(set(train['world'].unique()).union(set(test['world'].unique())))

    # create a dictionary numerating the titles

    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))

    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))

    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))

    assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(set(test[test['type'] == 'Assessment']['title'].value_counts().index)))

    # replace the text titles with the number titles from the dict

    train['title'] = train['title'].map(activities_map)

    test['title'] = test['title'].map(activities_map)

    train['world'] = train['world'].map(activities_world)

    test['world'] = test['world'].map(activities_world)

    train_labels['title'] = train_labels['title'].map(activities_map)

    win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))

    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest

    win_code[activities_map['Bird Measurer (Assessment)']] = 4110

    # convert text into datetime

    train['timestamp'] = pd.to_datetime(train['timestamp'])

    test['timestamp'] = pd.to_datetime(test['timestamp'])

    

    

    return train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code



# get usefull dict with maping encode

train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code = encode_title(train, test, train_labels)



categoricals = ['session_title']
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

        session_title_text = activities_labels[session_title] #from Andrew

        

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

            # get installation_id for aggregated features

            features['installation_id'] = session['installation_id'].iloc[-1] #from Andrew

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

    # if test_set=True, only the last assessment must be predicted, the previous are scraped

    if test_set:

        return all_assessments[-1]

    # in train_set, all assessments are kept

    return all_assessments
compiled_data = []

for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort=False)), total=train.installation_id.nunique(), desc='Installation_id', position=0):

    compiled_data += get_data(user_sample)
new_train = pd.DataFrame(compiled_data)

del compiled_data

new_train.shape
reduced_train = new_train[['Activity','Assessment','Clip','Game','installation_id','session_title','accumulated_correct_attempts','accumulated_uncorrect_attempts','duration_mean','accumulated_accuracy','accuracy_group','accumulated_accuracy_group','accumulated_actions']]

reduced_train.shape
new_test = []

for ins_id, user_sample in tqdm(test.groupby('installation_id', sort=False), total=test.installation_id.nunique(), desc='Installation_id', position=0):

    a = get_data(user_sample, test_set=True)

    new_test.append(a)

    

reduce_test = pd.DataFrame(new_test)
reduced_test = reduce_test[['Activity','Assessment','Clip','Game','installation_id','session_title','accumulated_correct_attempts','accumulated_uncorrect_attempts','duration_mean','accumulated_accuracy','accuracy_group','accumulated_accuracy_group','accumulated_actions']]

reduced_test.shape
trainf1 = reduced_train.head(16000)

trainf1_input = trainf1[['Activity','Assessment','Clip','Game','session_title','accumulated_correct_attempts','accumulated_uncorrect_attempts','duration_mean','accumulated_actions']]

trainf1_target = trainf1[['accuracy_group']]

trainf1_input.info()
trainf2 = reduced_train.tail(1690)

trainf2_input = trainf2[['Activity','Assessment','Clip','Game','session_title','accumulated_correct_attempts','accumulated_uncorrect_attempts','duration_mean','accumulated_actions']]

trainf2_target = trainf2[['accuracy_group']]
test_input = reduced_test[['Activity','Assessment','Clip','Game','session_title','accumulated_correct_attempts','accumulated_uncorrect_attempts','duration_mean','accumulated_actions']]

test_target = reduced_test[['accuracy_group']]
#Logistic Regression

lr_c=LogisticRegression(random_state=0)

lr_c.fit(trainf1_input,trainf1_target)

lr_pred=lr_c.predict(trainf2_input)

lr_cm=confusion_matrix(trainf2_target,lr_pred)

lr_ac=accuracy_score(trainf2_target, lr_pred)

print('LogisticRegression_accuracy:',lr_ac)

plt.figure(figsize=(10,5))

plt.title("LogisticRegression_cm")

sns.heatmap(lr_cm,annot=True,cmap="Oranges",fmt="d",cbar=False)
#Decision Tree

dtree_c=DecisionTreeClassifier(criterion='entropy',random_state=0)

dtree_c.fit(trainf1_input,trainf1_target)

dtree_pred=dtree_c.predict(trainf2_input)

dtree_cm=confusion_matrix(trainf2_target,dtree_pred)

dtree_ac=accuracy_score(dtree_pred,trainf2_target)

plt.figure(figsize=(10,5))

plt.title("dtree_cm")

sns.heatmap(dtree_cm,annot=True,fmt="d",cbar=False)

print('DecisionTree_Classifier_accuracy:',dtree_ac)
#SVM Model

svc_r=SVC(kernel='rbf')

svc_r.fit(trainf1_input,trainf1_target)

svr_pred=svc_r.predict(trainf2_input)

svr_cm=confusion_matrix(trainf2_target,svr_pred)

svr_ac=accuracy_score(trainf2_target, svr_pred)

plt.figure(figsize=(10,5))

plt.title("svm_cm")

sns.heatmap(svr_cm,annot=True,cmap="Oranges",fmt="d",cbar=False)

print('SVM_regressor_accuracy:',svr_ac)
#RandomForest

rdf_c=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)

rdf_c.fit(trainf1_input,trainf1_target)

rdf_pred=rdf_c.predict(trainf2_input)

rdf_cm=confusion_matrix(trainf2_target,rdf_pred)

rdf_ac=accuracy_score(rdf_pred,trainf2_target)

plt.figure(figsize=(10,5))

plt.title("rdf_cm")

sns.heatmap(rdf_cm,annot=True,cmap="Blues",fmt="d",cbar=False)

print('RandomForest_accuracy:',rdf_ac)
#LGBM

lgbm_c=LGBMClassifier()

kfold = KFold(n_splits=5, random_state=1989)

lgbm_c.fit(trainf1_input,trainf1_target)

lgbm_pred = lgbm_c.predict(trainf2_input)

lgbm_cm=confusion_matrix(trainf2_target,lgbm_pred)

lgbm_ac=accuracy_score(lgbm_pred,trainf2_target)

plt.figure(figsize=(10,5))

plt.title("lgbm_cm")

sns.heatmap(lgbm_cm,annot=True,cmap="Blues",fmt="d",cbar=False)

print('LGBM_accuracy:',lgbm_ac)
model_accuracy = pd.Series(data=[lr_ac,dtree_ac,svr_ac,rdf_ac,lgbm_ac], 

        index=['Logistic_Regression','DecisionTree_Classifier','SVM_regressor_accuracy','RandomForest','LGBM'])

fig= plt.figure(figsize=(9,9))

model_accuracy.sort_values().plot.barh()

plt.title('Model Accracy')
lgbm_pred = lgbm_c.predict(test_input)

submission['accuracy_group'] = lgbm_pred

submission['accuracy_group'].plot(kind='hist')
os.chdir('/kaggle/working')
submission.to_csv('submission.csv', index=None)