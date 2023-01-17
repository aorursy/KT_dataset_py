# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import warnings

warnings.filterwarnings("ignore")
import os

import pandas as pd # Panads for Data Handling

import numpy as np  # Numpy for Maths

import seaborn as sns  # SeaBorn for Plotting Charts 

import matplotlib.pylab as plt # Matplotlib for Plotting
# Read in the data CSV files



df_train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')

df_train_lbls = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')

df_test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')

df_specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')



# This Data Frame will be used while formatting submission data

df_submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')
# Details of data



print('Train Data: ', df_train.shape)

df_train.info()

print("+++++++++++++++++++++++++++++++++++++")

print('Train Label Data: ', df_train_lbls.shape)

df_train_lbls.info()

print("+++++++++++++++++++++++++++++++++++++")

print('Test Data: ', df_test.shape)

df_test.info()

print("+++++++++++++++++++++++++++++++++++++")

print('Specification Data: ', df_specs.shape)

df_specs.info()
# Check the train data set.



df_train.head()
# As Train Label data set provides details about assesment of training data



df_test.head()
# Currently we focus only on "train_label" data. As it contains information to sort and manipulate the "train" data.



from IPython.display import display



display(df_train_lbls.columns, df_train_lbls.shape)
# Lets peek more into train label data



df_train_lbls.groupby('accuracy_group')['game_session'].count().plot(kind='barh', figsize=(10, 5), title='Target (accuracy group)')
# Reference: https://stackoverflow.com/questions/42592493/displaying-pair-plot-in-pandas-data-frame

sns.pairplot(df_train_lbls, hue='accuracy_group')
# Function for Modification in Train/Test Data Frame

def get_time(df):

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['date'] = df['timestamp'].dt.date

    df['month'] = df['timestamp'].dt.month

    df['hour'] = df['timestamp'].dt.hour

    df['week_day'] = df['timestamp'].dt.dayofweek

    return df



# Train Data Frame

df_train = get_time(df_train)

# Test Data Frame

df_test = get_time(df_test)
df_train.groupby('hour')['event_id'].agg('count').plot(figsize=(15, 3),title='Observations by Hour', color='red')

plt.show()



df_train.groupby('date')['event_id'].agg('count').plot(figsize=(15, 3),title='Observations by Date', color='blue')

plt.show()



df_train.groupby('week_day')['event_id'].agg('count').plot(figsize=(15, 3),title='Observations by Week',color='yellow')

plt.show()



df_train.groupby('month')['event_id'].agg('count').plot(figsize=(15, 3),title='Observations by Month',color='green')

plt.show()
df_train['log1p_game_time'] = df_train['game_time'].apply(np.log1p)



fig, ax = plt.subplots(figsize=(15, 5))

sns.catplot(x="type", y="log1p_game_time",data=df_train.sample(10000), alpha=0.5, ax=ax);

ax.set_title('Distribution of log1p(game_time) by Type')

plt.close()

plt.show()



fig, ax = plt.subplots(figsize=(15, 5))

sns.catplot(x="world", y="log1p_game_time",data=df_train.sample(10000), alpha=0.5, ax=ax);

ax.set_title('Distribution of log1p(game_time) by World')

plt.close()

plt.show()
#Encode Title



list_user_activities = list(set(df_train['title'].value_counts().index).union(set(df_test['title'].value_counts().index)))

activity_map = dict(zip(list_user_activities, np.arange(len(list_user_activities))))



df_train['title'] = df_train['title'].map(activity_map)

df_test['title'] = df_test['title'].map(activity_map)

df_train_lbls['title'] = df_train_lbls['title'].map(activity_map)
win_code = dict(zip(activity_map.values(), (4100*np.ones(len(activity_map))).astype('int')))

win_code[activity_map['Bird Measurer (Assessment)']] = 4110
def get_data(user_sample, test_set=False):

    last_activity = 0

    user_activities_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}

    accuracy_groups = {0:0, 1:0, 2:0, 3:0}

    all_assessments = []

    accumulated_accuracy_group = 0

    accumulated_accuracy=0

    accumulated_correct_attempts = 0 

    accumulated_uncorrect_attempts = 0 

    accumulated_actions = 0

    counter = 0

    durations = []

    for i, session in user_sample.groupby('game_session', sort=False):

        session_type = session['type'].iloc[0]

        session_title = session['title'].iloc[0]

        if test_set == True:

            second_condition = True

        else:

            if len(session)>1:

                second_condition = True

            else:

                second_condition= False

            

        if (session_type == 'Assessment') & (second_condition):

            all_attempts = session.query(f'event_code == {win_code[session_title]}')

            true_attempts = all_attempts['event_data'].str.contains('true').sum()

            false_attempts = all_attempts['event_data'].str.contains('false').sum()

            features = user_activities_count.copy()

            features['session_title'] = session['title'].iloc[0] 

            features['accumulated_correct_attempts'] = accumulated_correct_attempts

            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts

            accumulated_correct_attempts += true_attempts 

            accumulated_uncorrect_attempts += false_attempts

            if durations == []:

                features['duration_mean'] = 0

            else:

                features['duration_mean'] = np.mean(durations)

            durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)

            features['accumulated_accuracy'] = accumulated_accuracy/counter if counter > 0 else 0

            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0

            accumulated_accuracy += accuracy

            if accuracy == 0:

                features['accuracy_group'] = 0

            elif accuracy == 1:

                features['accuracy_group'] = 3

            elif accuracy == 0.5:

                features['accuracy_group'] = 2

            else:

                features['accuracy_group'] = 1



            features.update(accuracy_groups)

            features['accumulated_accuracy_group'] = accumulated_accuracy_group/counter if counter > 0 else 0

            features['accumulated_actions'] = accumulated_actions

            accumulated_accuracy_group += features['accuracy_group']

            accuracy_groups[features['accuracy_group']] += 1

            if test_set == True:

                all_assessments.append(features)

            else:

                if true_attempts+false_attempts > 0:

                    all_assessments.append(features)

                

            counter += 1



        # Accumulated actions on session.

        accumulated_actions += len(session)

        if last_activity != session_type:

            user_activities_count[session_type] += 1

            last_activitiy = session_type



    if test_set:

        return all_assessments[-1] 

    return all_assessments
from tqdm import tqdm_notebook as tqdm



def get_train_and_test(df_train, df_test):

    compiled_train = []

    compiled_test = []

    for i, (ins_id, user_sample) in tqdm(enumerate(df_train.groupby('installation_id', sort = False)), total = 17000):

        compiled_train += get_data(user_sample)

    for ins_id, user_sample in tqdm(df_test.groupby('installation_id', sort = False), total = 1000):

        test_data = get_data(user_sample, test_set = True)

        compiled_test.append(test_data)

    df_train_ = pd.DataFrame(compiled_train)

    df_test_ = pd.DataFrame(compiled_test)

    cat_features = ['session_title']

    return df_train_, df_test_, cat_features
import lightgbm as lgb

import gc

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit

from sklearn.preprocessing import LabelEncoder

from catboost import CatBoostClassifier

from time import time

import datetime

from sklearn.metrics import confusion_matrix
df_train_, df_test_, cat_features = get_train_and_test(df_train, df_test)



all_features = [x for x in df_train_.columns if x not in ['accuracy_group']]

X, y = df_train_[all_features], df_train_['accuracy_group']

del df_train
df_train_.head()
# QWK function, Reference: https://www.kaggle.com/mhviraf/a-new-baseline-for-dsb-2019-catboost-model/comments

def qwk(act,pred,n=4,hist_range=(0,3)):

    

    O = confusion_matrix(act,pred)

    O = np.divide(O,np.sum(O))

    

    W = np.zeros((n,n))

    for i in range(n):

        for j in range(n):

            W[i][j] = ((i-j)**2)/((n-1)**2)

            

    act_hist = np.histogram(act,bins=n,range=hist_range)[0]

    prd_hist = np.histogram(pred,bins=n,range=hist_range)[0]

    

    E = np.outer(act_hist,prd_hist)

    E = np.divide(E,np.sum(E))

    

    num = np.sum(np.multiply(W,O))

    den = np.sum(np.multiply(W,E))

        

    return 1-np.divide(num,den)
# https://catboost.ai/docs/concepts/python-reference_catboostclassifier.html

def make_CatBoostClassifier(iterations=6000):

    clf = CatBoostClassifier(

        loss_function='MultiClass',

        eval_metric="WKappa",

        task_type="CPU",

        #learning_rate=0.01,

        iterations=iterations,

        od_type="Iter",

        #depth=4,

        early_stopping_rounds=500,

        #l2_leaf_reg=10,

        #border_count=96,

        random_seed=45,

        #use_best_model=use_best_model,

        verbose=0

    )

    return clf
# CV

from sklearn.model_selection import KFold, StratifiedKFold



NFOLDS = 7 # More the number of splits/folds, less the test will be impacted by randomness

folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=7654)

training_start_time = time()

predictions_tr = []

models = []

for i in range(3):

    oof = np.zeros(len(X)) # oof is an zeroed array

    for fold, (train_idx, test_idx) in enumerate(folds.split(X, y)):

        # each iteration returns data.

        start_time = time()

        print(f'Training on fold {fold+1}')



        # fits the model

        clf = make_CatBoostClassifier()

        clf.fit(X.loc[train_idx, all_features], y.loc[train_idx], eval_set=(X.loc[test_idx, all_features], y.loc[test_idx]),

                                  use_best_model=True, verbose=0, cat_features=cat_features)

                

        # The predictions of each split is inserted into the oof array

        oof[test_idx] = clf.predict(X.loc[test_idx, all_features]).reshape(len(test_idx))

        models.append(clf)



    print('-' * 30)

    print('OOF QWK:', i, qwk(y, oof))

    print('-' * 30)
from scipy import stats



pred_catclf = []

for model in models:

    pred_catclf.append(model.predict(df_test_))

    #pred_catclf.append(model.predict(df_test_[cat_features]))

pred_catclf = np.concatenate(pred_catclf, axis=1)

print(pred_catclf.shape)
from sklearn.metrics import cohen_kappa_score



def run_lgb(df_train_, df_test_, usefull_features):

    kf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 42)

    target = 'accuracy_group'

    oof_pred = np.zeros((len(df_train_), 4))

    pred_lgb = np.zeros((len(df_test_), 4))

    for fold, (tr_ind, val_ind) in enumerate(kf.split(df_train_, df_train_[target])):

        print('Fold {}'.format(fold + 1))

        x_train, x_val = df_train_[usefull_features].iloc[tr_ind], df_train_[usefull_features].iloc[val_ind]

        y_train, y_val = df_train_[target][tr_ind], df_train_[target][val_ind]

        train_set = lgb.Dataset(x_train, y_train, categorical_feature=cat_features)

        val_set = lgb.Dataset(x_val, y_val, categorical_feature=cat_features)



        params = {

            'learning_rate': 0.01,

            'metric': 'multiclass',

            'objective': 'multiclass',

            'num_classes': 4,

            'feature_fraction': 0.75,

            'subsample': 0.75,

            'n_jobs': -1,

            'seed': 50,

            'max_depth': 10

        }



        model = lgb.train(params, train_set, num_boost_round = 1000000, early_stopping_rounds = 50, 

                          valid_sets=[train_set, val_set], verbose_eval = 100)

        oof_pred[val_ind] = model.predict(x_val)

        pred_lgb += model.predict(df_test_[usefull_features]) / 5

    loss_score = cohen_kappa_score(df_train_[target], np.argmax(oof_pred, axis = 1), weights = 'quadratic')

    result = pd.Series(np.argmax(oof_pred, axis = 1))

    print('Our oof cohen kappa score is: ', loss_score)

    print(result.value_counts(normalize = True))

    return pred_lgb
pred_lgbclf = run_lgb(df_train_, df_test_, all_features)


import xgboost as xgb



def run_XGBoostClassifier(X_train,y_train,final_test,n_splits=3):

    scores=[]

    pars = {

        'colsample_bytree': 0.8,                 

        'learning_rate': 0.08,

        'max_depth': 10,

        'subsample': 1,

        'objective':'multi:softprob',

        'num_class':4,

        'eval_metric':'mlogloss',

        'min_child_weight':3,

        'gamma':0.25,

        'n_estimators':500

    }



    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    y_pre=np.zeros((len(final_test),4),dtype=float)

    #y_pre=np.zeros((len(final_test), 4))

    final_test=xgb.DMatrix(final_test.drop('accuracy_group',axis=1))

    #final_test=xgb.DMatrix(final_test[usefull_features])





    for train_index, val_index in kf.split(X_train, y_train):

        train_X = X_train.iloc[train_index]

        val_X = X_train.iloc[val_index]

        train_y = y_train[train_index]

        val_y = y_train[val_index]

        xgb_train = xgb.DMatrix(train_X, train_y)

        xgb_eval = xgb.DMatrix(val_X, val_y)



        xgb_model = xgb.train(pars,

                      xgb_train,

                      num_boost_round=1000,

                      evals=[(xgb_train, 'train'), (xgb_eval, 'val')],

                      verbose_eval=False,

                      early_stopping_rounds=20

                     )



        val_X=xgb.DMatrix(val_X)

        pred_val=[np.argmax(x) for x in xgb_model.predict(val_X)]

        score=cohen_kappa_score(pred_val,val_y,weights='quadratic')

        scores.append(score)

        print('choen_kappa_score :',score)



        pred=xgb_model.predict(final_test)

        y_pre+=pred



    pred = np.asarray([np.argmax(line) for line in y_pre])

    print('Mean score:',np.mean(scores))

    

    return xgb_model,pred
xgb_model,pred_xgbclf=run_XGBoostClassifier(X, y, df_test_, 5)
print(pred_catclf.shape)

print(pred_xgbclf.shape)

print(pred_lgbclf.shape)



df_submission['accuracy_group'] = np.round(pred_catclf).astype('int')

sub_catclf = pd.DataFrame(df_submission['accuracy_group'])



df_submission['accuracy_group'] = np.round(pred_xgbclf).astype('int')

sub_xgbclf = pd.DataFrame(df_submission['accuracy_group'])



df_submission['accuracy_group'] = np.round(pred_lgbclf).astype('int')

sub_lgbclf = pd.DataFrame(df_submission['accuracy_group'])



sub_final = 0.6*sub_catclf + 0.3*sub_xgbclf + 0.1*sub_lgbclf



#df_submission['accuracy_group'] = prediction.argmax(1)

#df_submission.to_csv('submission.csv', index=False)

#df_submission.head()



sub_final.to_csv('submission.csv', index=False)

sub_final.head()



#df_submission['accuracy_group'] = np.round(pred_catclassifier).astype('int')

#df_submission.to_csv('submission.csv', index=None)

#df_submission.head()
#Final Submission



#df_submission['accuracy_group'].value_counts(normalize=True)

sub_final['accuracy_group'].value_counts(normalize=True)