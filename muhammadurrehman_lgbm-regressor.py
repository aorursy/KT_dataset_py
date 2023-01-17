from sklearn.model_selection import GroupKFold #CV purposes

import time #for time related tasks



from sklearn.model_selection import train_test_split

import lightgbm as lgb



import pandas as pd

import numpy as np

from tqdm import tqdm
import os

for dirname, _, filenames in os.walk('/kaggle/output'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



def read_data(folder_path):

#     print("reading Files ...")

    train = pd.read_csv(folder_path+"/train.csv")

    test = pd.read_csv(folder_path+"/test.csv")

    train_labels = pd.read_csv(folder_path+"/train_labels.csv")

    submission_sample = pd.read_csv(folder_path+"/sample_submission.csv")

    return train, test, train_labels, submission_sample
train, test, train_lables, submission_sample = read_data("/kaggle/input/data-science-bowl-2019")
def encode_values():

#     print("getting encoded Values")

    unique_event_id = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))

    unique_game_session = list(set(train['game_session'].unique()).union(set(test['game_session'].unique())))

    unique_installation_id = list(set(train['installation_id'].unique()).union(set(test['installation_id'].unique())))

    unique_event_code = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))

    unique_titles = list(set(train['title'].unique()).union(set(test['title'].unique())))

    unique_type = list(set(train['type'].unique()).union(set(test['type'].unique())))

    unique_world = list(set(train['world'].unique()).union(set(test['world'].unique())))

    

    # map titles by numbers

    titles_to_number = dict(zip(unique_titles, np.arange(len(unique_titles))))

    # map number by titles

    numbers_to_title = dict(zip(np.arange(len(unique_titles)), unique_titles))

    # map world by numbers

    world_to_number = dict(zip(unique_world, np.arange(len(unique_world))))

    # map world by numbers

    type_to_number = dict(zip(unique_type, np.arange(len(unique_type))))

    # get list of all the assessments titles

    assessment_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(set(test[test['type'] == 'Assessment']['title'].value_counts().index)))

    

    

    train['timestamp'] = pd.to_datetime(train['timestamp'])

    test['timestamp'] = pd.to_datetime(test['timestamp'])

    

    return unique_event_id, unique_game_session, unique_installation_id, unique_event_code, unique_titles, unique_type, unique_world, assessment_titles,world_to_number,  type_to_number, numbers_to_title, titles_to_number
unique_event_id, unique_game_session, unique_installation_id, unique_event_code, unique_titles, unique_type, unique_world, assessment_titles, world_to_number, type_to_number, numbers_to_title, titles_to_number = encode_values()

def parse_data(dataframe, test_dataset):

    records = []



    accuracy_groups = {0: 0, 1: 0, 2: 0, 3: 0}

    activites_count = {'Clip': 0, 'Game': 0, 'Activity': 0, 'Assessment': 0}

    every_assessments_count = {title: 0 for title in assessment_titles}

    every_assessments_accuracy = {title + "_accuracy": 0 for title in assessment_titles}  # keep record for last played



    accuracy_sum = 0

    num_correct = 0

    num_incorrect = 0

    count = 0

    durations = []

    game_sessions = dataframe.groupby("game_session")

    sessions_count = len(game_sessions)



    for i, game_session in game_sessions:

        event_count = game_session.iloc[len(game_session) - 1]['event_count']

        installation_id = game_session.iloc[len(game_session) - 1]['installation_id']

        session_type = game_session.iloc[len(game_session) - 1]['type']

        session_title = game_session.iloc[len(game_session) - 1]['title']

        # keeping the count of each type

        activites_count[session_type] += 1



        if (session_type == "Assessment") & (test_dataset or len(game_session) > 1):

            # keep record of how many time this assessments has been taken

            every_assessments_count[session_title] += 1



            if session_title == 'Bird Measurer (Assessment)':

                all_attempts = game_session[game_session['event_code'].isin([4110])]

                correct = all_attempts['event_data'].str.contains('true').sum()

                incorrect = all_attempts['event_data'].str.contains('false').sum()

#                 event_code = game_session['event_code']



            else:

                all_attempts = game_session[game_session['event_code'].isin([4100])]

                correct = all_attempts['event_data'].str.contains('true').sum()

                incorrect = all_attempts['event_data'].str.contains('false').sum()

#                 event_code = game_session['event_code']





            num_correct += correct

            num_incorrect += incorrect

            count += 1



            accuracy = correct / (correct + incorrect) if (correct + incorrect)>0 else 0

            accuracy_sum += accuracy

            mean_accuracy = accuracy_sum / count

            every_assessments_accuracy[session_title + "_accuracy"] = accuracy

            if accuracy == 0:

                accuracy_groups[0] += 1

                accuracy_group = 0

            elif accuracy == 1:

                accuracy_groups[3] += 1

                accuracy_group = 3

            elif accuracy == 0.5:

                accuracy_groups[2] += 1

                accuracy_group = 2

            else:

                accuracy_groups[1] += 1

                accuracy_group = 1

            durations.append((game_session.iloc[-1, 2] - game_session.iloc[0, 2]).seconds)

            duration_mean = np.mean(durations)



            generated_data = accuracy_groups.copy()

            generated_data.update(activites_count.copy())

            generated_data.update(every_assessments_count.copy())

            generated_data.update(every_assessments_accuracy.copy())

            generated_data['sessions_count'] = sessions_count

            generated_data['mean_time'] = duration_mean

            generated_data['event_count'] = event_count

#             generated_data['event_code'] = event_code

            generated_data['title'] = session_title

            generated_data['assessment_session_count'] = count

            generated_data['num_correct'] = num_correct

            generated_data['num_incorrect'] = num_incorrect

            generated_data['accuracy'] = accuracy

            generated_data['mean_accuracy'] = mean_accuracy

            generated_data['accuracy_group'] = accuracy_group

            generated_data['installation_id'] = installation_id

            

            records.append(generated_data)



    # if it't the test_set, only the last assessment must be predicted

    if test_dataset:

        return records[-1]

    # in the train_set, all assessments goes to the dataset

    return records

def get_preprocessed_data():

    trian_preprocessed_data =[]

    test_preprocessed_data =[]

    

    train_ID_groups = train.groupby('installation_id', sort = False)

    test_ID_groups = test.groupby('installation_id', sort = False)

    

    for i, (installation_id, history) in tqdm(enumerate(train_ID_groups), total = len(train_ID_groups)):

        trian_preprocessed_data += parse_data(history, False)

    

    for i, (installation_id, history) in tqdm(enumerate(test_ID_groups), total = len(test_ID_groups)):

        t = parse_data(history, True)

        test_preprocessed_data.append(t)

        

    t = pd.DataFrame(trian_preprocessed_data)

    ts = pd.DataFrame(test_preprocessed_data)

    return t, ts

    
p_train, p_test = get_preprocessed_data()
p_train['title'] = p_train['title'].map(titles_to_number)

p_test['title'] = p_test['title'].map(titles_to_number)
p_train.head()
def qwk_loss(a1, a2):

    max_rat = 3

    a1 = np.asarray(a1, dtype=int)

    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1, ))

    hist2 = np.zeros((max_rat + 1, ))

    o = 0

    for k in range(a1.shape[0]):

        i, j = a1[k], a2[k]

        hist1[i] += 1

        hist2[j] += 1

        o +=  (i - j) * (i - j)

    e = 0

    for i in range(max_rat + 1):

        for j in range(max_rat + 1):

            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e
def regr_resl_to_label(true_labels, preds_labels):

    preds_labels[preds_labels <= 1.12232214] = 0

    preds_labels[np.where(np.logical_and(preds_labels > 1.12232214, preds_labels <= 1.73925866))] = 1

    preds_labels[np.where(np.logical_and(preds_labels > 1.73925866, preds_labels <= 2.22506454))] = 2

    preds_labels[preds_labels > 2.22506454] = 3

    return 'cappa', qwk_loss(true_labels, preds_labels), True
scores = []

def train_model(X: pd.DataFrame,

                y,

            folds = None,

            params: dict = None,

            del_cols: list = None):



    """Basic parameters

        1. X: train_data

        2. y: ground truth labels

        3. params: lightGBM parameters

        4. del_cols: columns to be avoided while training like accuracy_group must not be a column! 

    """

    global scores

    eval_metric = regr_resl_to_label #custom metric as defined above

    columns = [col for col in X.columns.values if not col in del_cols] #features

    

    models = [] #save n_folds models

    n_target = 1 # number of targets

    oof = np.zeros((len(X), n_target)) # out of fold predictions



    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y, X['installation_id'])):

        

        print('Fold {} started at {}'.format(fold_n + 1,time.ctime()))

        X_train, X_valid = X.loc[train_index,columns], X.loc[valid_index,columns]

        y_train, y_valid = y.loc[train_index], y.loc[valid_index]

        print(X_train.shape)

        

        #Eval set preparation

        eval_set = [(X_train, y_train)]

        eval_names = ['train']

        eval_set.append((X_valid, y_valid))

        eval_names.append('valid')

        categorical_columns = 'auto'

        

        model = lgb.LGBMRegressor(**params)

        model.fit(X=X_train, y=y_train,

                       eval_set=eval_set, eval_names=eval_names, eval_metric=eval_metric,

                       verbose=params['verbose'], early_stopping_rounds=params['early_stopping_rounds'],

                       categorical_feature=categorical_columns)

        

        oof[valid_index] = model.predict(X_valid).reshape(-1, n_target)

        score = regr_resl_to_label(X.loc[valid_index,"accuracy_group"],oof[valid_index])

        scores.append(score)

        models.append(model)

    scores = [score[1][0] for score in scores]

    print(scores)

    return models
params = {'verbose': 100,

          'learning_rate': 0.010514633017309072,

          'metric': 'rmse',

          'bagging_freq': 3,

          'boosting_type': 'gbdt',

          'eval_metric': 'cappa',

          'lambda_l1': 4.8999704874480745,

          'colsample_bytree': 0.4236269531042225,

          'early_stopping_rounds': 100,

          'max_depth': 12,

          'lambda_l2': 0.054084652510602016,

          'bagging_fraction': 0.7931423220563563,

          'n_jobs': -1,

          'n_estimators': 2000,

          'objective': 'regression',

          'seed': 42}
p_train.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in p_train.columns]

p_test.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in p_test.columns]



# no need for these columns in training

cols_to_drop = ['installation_id','accuracy_group'] + [col for col in p_train.columns.values if "_time" in str(col)]#ground truth fact labels



y = p_train['accuracy_group']

n_fold = 5

folds = GroupKFold(n_splits=n_fold)

models = train_model(X = p_train, y = y,folds = folds, params = params, del_cols = cols_to_drop)
models
def predict(models, X_test, averaging: str = 'usual'):

    full_prediction = np.zeros((X_test.shape[0], 1))

    for i in range(len(models)):

        X_t = X_test.copy()

        if cols_to_drop is not None:

            del_cols = [col for col in cols_to_drop if col in X_t.columns.values]

            X_t = X_t.drop(del_cols, axis=1)

        y_pred = models[i].predict(X_t).reshape(-1, full_prediction.shape[1])

        if full_prediction.shape[0] != len(y_pred):

            full_prediction = np.zeros((y_pred.shape[0], 1))

        if averaging == 'usual':

            full_prediction += y_pred

        elif averaging == 'rank':

            full_prediction += pd.Series(y_pred).rank().values

    return full_prediction / len(models)
preds = predict(models, p_test)

    

coefficients = [1.12232214, 1.73925866, 2.22506454]

preds[preds <= coefficients[0]] = 0

preds[np.where(np.logical_and(preds > coefficients[0], preds <= coefficients[1]))] = 1

preds[np.where(np.logical_and(preds > coefficients[1], preds <= coefficients[2]))] = 2

preds[preds > coefficients[2]] = 3
preds = preds.astype(int)
submission_sample['accuracy_group'] = preds.astype(int)

submission_sample.to_csv('submission.csv', index=False)

submission_sample['accuracy_group'].value_counts(normalize=True)

submission_sample
submission_sample.to_csv('/kaggle/working/submission.csv', index = False)