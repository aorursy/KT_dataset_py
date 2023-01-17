# All the needed libraries

import numpy as np

import pandas as pd

import random

from random import choice

from collections import Counter



import lightgbm as lgb



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit, RepeatedStratifiedKFold

from sklearn.metrics import accuracy_score,f1_score,roc_auc_score, cohen_kappa_score, mean_squared_error

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

import tensorflow as tf



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import statsmodels.api as sm

import warnings

warnings.filterwarnings("ignore")

from tqdm import tqdm_notebook as tqdm



random.seed(42)

np.random.seed(42)
!ls -alh ../input/data-science-bowl-2019/
# Read in the data CSV files

training_data = pd.read_csv('../input/data-science-bowl-2019/train.csv')

testing_data = pd.read_csv('../input/data-science-bowl-2019/test.csv')
specs = pd.read_csv('../input/data-science-bowl-2019/specs.csv')

train_labels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')

sample_submission = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')
print("Train data has {} rows and the following {} columns.\n{}\n" \

      .format(training_data.shape[0],training_data.shape[1],training_data.columns.to_list()))

print("Test data has {} rows and the following {} columns.\n{}\n" \

      .format(testing_data.shape[0],testing_data.shape[1],testing_data.columns.to_list()))

print("The specs data has {} rows and the following {} columns.\n{}\n" \

      .format(specs.shape[0], specs.shape[1], specs.columns.to_list()))

print("The labels data has {} rows and the following {} columns.\n{}\n" \

      .format(train_labels.shape[0], train_labels.shape[1],train_labels.columns.to_list()))

print("The sample submission has {} rows and the following {} columns.\n{}\n" \

      .format(sample_submission.shape[0], sample_submission.shape[1],sample_submission.columns.to_list()))
reduced_training_data = training_data[training_data['installation_id'] \

                                      .isin(train_labels['installation_id'])]

print("{} ids out of {} in the training set are also in the labels." \

      .format(reduced_training_data.shape[0],training_data.shape[0]))

print("There are a total of {} unique ids in the reduced_training_data." \

      .format(len(reduced_training_data['installation_id'].unique())))

print("There are a total of {} unique ids in the train_labels." \

      .format(len(train_labels['installation_id'].unique())))

print("The number of common ids in the reduced set vs the labels is {}" \

     .format(len(list(set(reduced_training_data['installation_id'].unique()) \

                      .intersection(set(train_labels['installation_id'].unique()))))))
print(train_labels.shape)

train_labels.head()

print("There are a total of {} unique game_sessions in the train_labels out of total of {}" \

      .format(len(train_labels['game_session'].unique()), train_labels.shape[0]))

print(training_data.columns.to_list())
# make a list with all the unique categoricals from the training_data and testing_data set

list_of_all_titles = list(set(training_data['title'].unique()).union(set(testing_data['title'].unique())))

list_of_all_event_codes = list(set(training_data['event_code'].unique()).union(set(testing_data['event_code'].unique())))

list_of_event_id = list(set(training_data['event_id'].unique()).union(set(testing_data['event_id'].unique())))

list_of_worlds = list(set(training_data['world'].unique()).union(set(testing_data['world'].unique())))



# create a dictionary enumerating the titles

# enumerated_titles = dict(zip(sorted(list_of_all_titles), ["t_" + str(x) for x in range(len(list_of_all_titles))]))

enumerated_titles = dict(zip(sorted(list_of_all_titles), np.arange(len(list_of_all_titles))))

activities_labels = {value:key for key, value in enumerated_titles.items()}

enumerated_worlds = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))

assessment_titles = list(set(training_data[training_data['type'] == 'Assessment']['title'].value_counts().index).union(set(testing_data[testing_data['type'] == 'Assessment']['title'].value_counts().index)))



# Map the replace the text titles with the number titles from the dict

training_data['title'] = training_data['title'].map(enumerated_titles)

testing_data['title'] = testing_data['title'].map(enumerated_titles)

training_data['world'] = training_data['world'].map(enumerated_worlds)

testing_data['world'] = testing_data['world'].map(enumerated_worlds)

train_labels['title'] = train_labels['title'].map(enumerated_titles)



correct_answer_code = dict(zip(enumerated_titles.values(), (4100*np.ones(len(enumerated_titles))).astype('int')))

correct_answer_code[enumerated_titles['Bird Measurer (Assessment)']] = 4110



training_data['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), training_data['title'], training_data['event_code']))

testing_data['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), testing_data['title'], testing_data['event_code']))

all_title_event_code = list(set(training_data["title_event_code"].unique()).union(testing_data["title_event_code"].unique()))



# convert text into datetime

training_data['timestamp'] = pd.to_datetime(training_data['timestamp'])

testing_data['timestamp'] = pd.to_datetime(testing_data['timestamp'])
def get_qwk_score(y_true, y_pred):

    dist = Counter(reduce_train['accuracy_group'])

    for k in dist:

        dist[k] /= len(reduce_train)

    reduce_train['accuracy_group'].hist()

    

    acum = 0

    bound = {}

    for i in range(3):

        acum += dist[i]

        bound[i] = np.percentile(y_pred, acum * 100)



    def classify(x):

        if x <= bound[0]:

            return 0

        elif x <= bound[1]:

            return 1

        elif x <= bound[2]:

            return 2

        else:

            return 3



    y_pred = np.array(list(map(classify, y_pred))).reshape(y_true.shape)



    return 'cappa', cohen_kappa_score(y_true, y_pred, weights='quadratic'), True



def build_features(installation_group, test_set=False):

    # Constants and parameters declaration

    last_activity = 0

    count_of_tasks = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}

    

    # new features: time spent in each activity

    last_session_time_sec = 0

    accuracy_groups = {0:0, 1:0, 2:0, 3:0}

    all_assessments = []

    accumulated_accuracy_group = 0

    accumulated_accuracy = 0

    cumulative_true = 0 

    cumulative_false = 0

    accumulated_actions = 0

    counter = 0

    time_first_activity = float(installation_group['timestamp'].values[0])

    durations = []

    last_accuracy_title = {'acc_' + title: -1 for title in assessment_titles}

    event_code_count = {x: 0 for x in list_of_all_event_codes}

    event_id_count = {y: 0 for y in list_of_event_id}

    title_count = {z: 0 for z in enumerated_titles.values()} 

    title_event_code_count = {w: 0 for w in all_title_event_code}





    # iterates through each game_session of one instalation_id

    for i, game_session in installation_group.groupby('game_session', sort=False):

        game_session_type = game_session['type'].iloc[0]

        session_title = game_session['title'].iloc[0]

        session_title_text = activities_labels[session_title]

        features = []     

        # Collect additional information for each 'Assessment' type

        if (game_session_type == 'Assessment') & (test_set or len(game_session)>1):

            all_assessment_attempts = game_session.query(f'event_code == {correct_answer_code[session_title]}')

            # then, check the numbers of wins and the number of losses

            result_true = all_assessment_attempts['event_data'].str.contains('true').sum()

            result_false = all_assessment_attempts['event_data'].str.contains('false').sum()



            # Start building the features dictionary

            features = count_of_tasks.copy()

            features.update(last_accuracy_title.copy())

            features.update(event_code_count.copy())

            features.update(event_id_count.copy())

            features.update(title_count.copy())

            features.update(title_event_code_count.copy())

            features['installation_id'] = game_session['installation_id'].iloc[-1]

            # features['game_session_title'] = session_title_text

            features['session_title'] = session_title

            

            # the 4 lines below add the feature of the history of the trials of this player

            # this is based on the all time attempts so far, at the moment of this assessment

            features['cumulative_true'] = cumulative_true

            features['cumulative_false'] = cumulative_false

            cumulative_true += result_true 

            cumulative_false += result_false



            # the time spent in the app so far

            if durations == []:

                features['duration_mean'] = 0

            else:

                features['duration_mean'] = np.mean(durations)

            durations.append((game_session.iloc[-1, 2] - game_session.iloc[0, 2] ).seconds)



            # the accuracy is the all time wins divided by the all time attempts

            features['accumulated_accuracy'] = accumulated_accuracy/counter if counter > 0 else 0



            accuracy = result_true/(result_true+result_false) if (result_true+result_false) != 0 else 0

            accumulated_accuracy += accuracy

            last_accuracy_title['acc_' + session_title_text] = accuracy



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

            # if it's a testing_data set, all sessions belong to the final dataset

            # it it's a training_data, needs to be passed throught this clausule: game_session.query(f'event_code == {correct_answer_code[session_title]}')

            # that means, must exist an event_code 4100 or 4110

            if test_set:

                all_assessments.append(features)

            elif result_true+result_false > 0:

                all_assessments.append(features)

                

            counter += 1

        

        # this piece counts how many actions was made in each event_code so far

        def update_counters(counter: dict, col: str):

                num_of_session_count = Counter(game_session[col])

                for k in num_of_session_count.keys():

                    counter[k] += num_of_session_count[k]

                return counter

            

        event_code_count = update_counters(event_code_count, "event_code")

        event_id_count = update_counters(event_id_count, "event_id")

        title_count = update_counters(title_count, 'title')

        title_event_code_count = update_counters(title_event_code_count, 'title_event_code')



        # counts how many actions the player has done so far, used in the feature of the same name

        accumulated_actions += len(game_session)

        if last_activity != game_session_type:

            count_of_tasks[game_session_type] += 1

            last_activitiy = game_session_type 



    

    # if it is the test_set, only the last assessment must be predicted, the previous are scraped

    if test_set:

        return all_assessments[:-1]

    # in the train_set, all assessments goes to the dataset

    return all_assessments



def get_train_and_test(training_data, testing_data):

    compiled_train = []

    compiled_test = []

    for i, (ins_id, installation_group) in tqdm(enumerate(training_data.groupby('installation_id', sort = False)), total = 17000):

        compiled_train += build_features(installation_group)

    for ins_id, installation_group in tqdm(testing_data.groupby('installation_id', sort = False), total = 1000):

        compiled_test += build_features(installation_group, test_set = True)



    reduce_train = pd.DataFrame(compiled_train)

    reduce_test = pd.DataFrame(compiled_test)



    # categoricals = ['game_session_title']

    categoricals = ['session_title']

    return reduce_train, reduce_test, categoricals



reduce_train, reduce_test, categoricals = get_train_and_test(training_data, testing_data)
def stract_hists(feature, train=reduce_train, test=reduce_test, adjust=False, plot=False):

    n_bins = 10

    train_data = train[feature]

    test_data = test[feature]

    if adjust:

        test_data *= train_data.mean() / test_data.mean()

    perc_90 = np.percentile(train_data, 95)

    train_data = np.clip(train_data, 0, perc_90)

    test_data = np.clip(test_data, 0, perc_90)

    train_hist = np.histogram(train_data, bins=n_bins)[0] / len(train_data)

    test_hist = np.histogram(test_data, bins=n_bins)[0] / len(test_data)

    msre = mean_squared_error(train_hist, test_hist)

    if plot:

        print(msre)

        plt.bar(range(n_bins), train_hist, color='blue', alpha=0.5)

        plt.bar(range(n_bins), test_hist, color='red', alpha=0.5)

        plt.show()

    return msre



# stract_hists('Magma Peak - Level 1_2000', adjust=False, plot=True)
# Extract features from the training data. 

all_labels = reduce_train.columns

print("There a total of {} columns in the reduced training dataset".format(len(all_labels)))



# Remove the columns that appear in the sample_submission csv. 

features = [x for x in all_labels if x not in sample_submission.columns]

print("There a total of {} columns in the feature set".format(len(features)))



# Remove columns with only '0' in the values

features = reduce_train.loc[:,(reduce_train.sum(axis=0) != 0)].columns # delete useless columns

print("{} columns were removed that contained only '0' in the cell. {} remaining" \

      .format(len(reduce_train.columns) - len(features), len(features)))
print("The dtypes in the reduced training dataframe is {}".format(reduce_train.dtypes.unique()))
print("The features now have {} columns".format(len(features)))



# counter = 0

to_remove = []

# for feat_a in features:

#     for feat_b in features:

#         if feat_a != feat_b and feat_a not in to_remove and feat_b not in to_remove:

#             c = np.corrcoef(reduce_train[feat_a], reduce_train[feat_b])[0][1]

#             if c > 0.995:

#                 counter += 1

#                 to_remove.append(feat_b)

#                 print('{}: FEAT_A: {} FEAT_B: {} - Correlation: {}'.format(counter, feat_a, feat_b, c))



to_exclude = [] 

adjusted_test = reduce_test.copy()

for feature in adjusted_test.columns:

    #print(feature)

    if feature not in ['accuracy_group', 'installation_id', 'session_title']:

        data = reduce_train[feature]

        train_mean = data.mean()

        data = adjusted_test[feature] 

        test_mean = data.mean()

        try:

            error = stract_hists(feature, adjust=True)

            ajust_factor = train_mean / test_mean

            if ajust_factor > 10 or ajust_factor < 0.1:# or error > 0.01:

                to_exclude.append(feature)

                print(feature, train_mean, test_mean, error)

            else:

                adjusted_test[feature] *= ajust_factor

        except:

            to_exclude.append(feature)

            #print("Feature: {}, Train Mean: {}, Test Mean: {}.".format(feature, train_mean, test_mean))



features = [x for x in features if x not in to_exclude]

print("The features now have {} columns".format(len(features)))

features = [x for x in features if x not in to_remove]

print("The features now have {} columns".format(len(features)))



features = [x for x in features if x not in ['accuracy_group', 'installation_id']]

print("The features now have {} columns".format(len(features)))

class Base_Model(object):

    

    def __init__(self, train_df, test_df, features, categoricals=[], n_splits=5, verbose=True):

        self.train_df = train_df

        self.test_df = test_df

        self.features = features

        self.n_splits = n_splits

        self.categoricals = categoricals

        self.target = 'accuracy_group'

        self.cv = self.get_cv()

        self.verbose = verbose

        self.params = self.get_params()

        self.y_pred, self.score, self.model = self.fit()

        

    def train_model(self, train_set, val_set):

        raise NotImplementedError

        

    def get_params(self):

        raise NotImplementedError

        

    def convert_dataset(self, x_train, y_train, x_val, y_val):

        raise NotImplementedError

        

    def get_cv(self):

        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        return cv.split(self.train_df, self.train_df[self.target])

    

    def convert_x(self, x):

        return x

        

    def fit(self):

        oof_pred = np.zeros((len(reduce_train), ))

        y_pred = np.zeros((len(reduce_test), ))

        for fold, (train_idx, val_idx) in enumerate(self.cv):

            x_train, x_val = self.train_df[self.features].iloc[train_idx], self.train_df[self.features].iloc[val_idx]

            y_train, y_val = self.train_df[self.target][train_idx], self.train_df[self.target][val_idx]

            train_set, val_set = self.convert_dataset(x_train, y_train, x_val, y_val)

            model = self.train_model(train_set, val_set)

            conv_x_val = self.convert_x(x_val)

            oof_pred[val_idx] = model.predict(conv_x_val).reshape(oof_pred[val_idx].shape)

            x_test = self.convert_x(self.test_df[self.features])

            y_pred += model.predict(x_test).reshape(y_pred.shape) / self.n_splits

            print('Partial score of fold {} is: {}'.format(fold, get_qwk_score(y_val, oof_pred[val_idx])[1]))

        _, loss_score, _ = get_qwk_score(self.train_df[self.target], oof_pred)

        if self.verbose:

            print('Our oof cohen kappa score is: ', loss_score)

        return y_pred, loss_score, model





class LogisticRegression_Model(Base_Model):

    def train_model(self, train_set, val_set):

        verbosity = 100 if self.verbose else 0

        lr = LogisticRegression()

        lr.set_params(**self.params)

        return lr.fit(train_set, val_set)

        

    def convert_dataset(self, x_train, y_train, x_val, y_val):

        train_set = pd.concat([x_train, x_val])

        val_set = pd.concat([y_train, y_val])

        return train_set, val_set

        

    def get_params(self):

        params = {  'penalty': 'l2',

                    'solver': 'lbfgs',

                    'class_weight': 'balanced',

                    'random_state': 42,

                    'max_iter': 200,

                    'verbose': 100

                }

        return params

lr_model = LogisticRegression_Model(reduce_train, reduce_test, features, categoricals=categoricals)
class Lgb_Model(Base_Model):

    

    def train_model(self, train_set, val_set):

        verbosity = 100 if self.verbose else 0

        return lgb.train(self.params, train_set, valid_sets=[train_set, val_set], verbose_eval=verbosity)

        

    def convert_dataset(self, x_train, y_train, x_val, y_val):

        train_set = lgb.Dataset(x_train, y_train, categorical_feature=self.categoricals)

        val_set = lgb.Dataset(x_val, y_val, categorical_feature=self.categoricals)

        return train_set, val_set

        

    def get_params(self):

        params = {'n_estimators':5000,

                    'boosting_type': 'gbdt',

                    'objective': 'regression',

                    'metric': 'rmse',

                    'subsample': 0.75,

                    'subsample_freq': 1,

                    'learning_rate': 0.01,

                    'feature_fraction': 0.9,

                    'max_depth': 15,

                    'lambda_l1': 1,  

                    'lambda_l2': 1,

                    'early_stopping_rounds': 100

                    }

        return params



lgb_model = Lgb_Model(reduce_train, reduce_test, features, categoricals=categoricals)
final_pred = (lgb_model.y_pred)

print("The length of final_pred is {}".format(len(final_pred)))



dist = Counter(reduce_train['accuracy_group'])

for k in dist:

    dist[k] /= len(reduce_train)

reduce_train['accuracy_group'].hist()



acum = 0

bound = {}

for i in range(3):

    acum += dist[i]

    bound[i] = np.percentile(final_pred, acum * 100)

# print(bound)



def classify(x):

    if x <= bound[0]:

        return 0

    elif x <= bound[1]:

        return 1

    elif x <= bound[2]:

        return 2

    else:

        return 3

    

final_pred = np.array(list(map(classify, final_pred)))

print("The final_pred is {}".format(final_pred))

print("The length of final_pred is {}".format(len(final_pred)))

sample_submission['accuracy_group'] = final_pred.astype(int)

sample_submission.to_csv('submission.csv', index=False)

sample_submission['accuracy_group'].value_counts(normalize=True)