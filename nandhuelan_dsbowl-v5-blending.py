from IPython.display import HTML

# Youtube

HTML('<iframe width="560" height="315" src="https://pbskids.org/apps/media/video/Seesaw_v6_subtitled_ccmix.ogv" width="700" height="240" frameborder="0" allowfullscreen></iframe>')
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.core.display import display, HTML

import plotly.offline as py

import plotly.graph_objs as go

import plotly.express as px

from plotly import tools, subplots

from tqdm import tqdm_notebook as tqdm

from functools import reduce

from sklearn.preprocessing import LabelEncoder,StandardScaler, MinMaxScaler, OneHotEncoder

from sklearn.model_selection import StratifiedKFold

from pandas.api.types import is_datetime64_any_dtype as is_datetime

from pandas.api.types import is_categorical_dtype

from sklearn.metrics import confusion_matrix,cohen_kappa_score, mean_squared_error

import lightgbm as lgb

from sklearn.model_selection import KFold, StratifiedKFold,GroupKFold

import shap,random

import seaborn as sns

import matplotlib.pyplot as plt

import gc

from collections import Counter

from catboost import CatBoostRegressor

import xgboost as xgb 

from sklearn.base import BaseEstimator, TransformerMixin

import copy

import tensorflow as tf



py.init_notebook_mode()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import warnings



# Any results you write to the current directory are saved as output.
path='../input/data-science-bowl-2019/'
sample_submission = pd.read_csv(path+'sample_submission.csv')

specs = pd.read_csv(path+'specs.csv')

test = pd.read_csv(path+'test.csv', parse_dates=["timestamp"])

train = pd.read_csv(path+'train.csv', parse_dates=["timestamp"])

train_labels = pd.read_csv(path+'train_labels.csv')
train.tail()
def reduce_mem_usage(df, use_float16=False):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        if is_datetime(df[col]) or is_categorical_dtype(df[col]):

            # skip datetime type or categorical type

            continue

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

    



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
train=reduce_mem_usage(train)

test=reduce_mem_usage(test)
train.head(2)
nrows = train.shape[0]

installids = train["installation_id"].nunique()

gameids = train["game_session"].nunique()

event_codes = train["event_code"].nunique()



min_date = train["timestamp"].min()

max_date = train["timestamp"].max()



display(HTML(f"""<br>Number of rows in the dataset: {nrows:,}</br>

             <br>Number of unique installation ids in the dataset: {installids:,}</br>

             <br>Number of unique game sessions in the dataset: {gameids:,}</br>

             <br>Number of unique event codes in the dataset: {event_codes:,}</br>

             <br>Min date value in train data is {min_date}</br>

             <br>Max date value in train data is {max_date}</br>

             """))



print(f"Unique installation ids in training data which has assessment data is: {train[train['type']=='Assessment']['installation_id'].nunique()}")



trainv1=train[train['installation_id'].isin(train[train['type']=='Assessment']['installation_id'].unique())]



print(f"No of unique rows after filtering:{len(trainv1)}")



del train
cnt_srs = trainv1["type"].value_counts().sort_index()



trace = go.Bar(

    x=cnt_srs.index,

    y=cnt_srs.values,

    marker=dict(

        color="#1E90FF",

    ),

)



layout = go.Layout(

    title=go.layout.Title(

        text="Type of activities",

        x=0.5

    ),

    font=dict(size=14),

    width=500,

    height=500,

)



data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Activity type")
def _generate_bar_plot_hor(df, col, title, color, w=None, h=None, lm=0):

    cnt_srs = df[col].value_counts().sort_values(ascending=False)

    

    trace = go.Bar(y=cnt_srs.index[::-1], x=cnt_srs.values[::-1], orientation = 'h',

        marker=dict(color=color))



    layout = dict(title=title, margin=dict(l=lm), width=w, height=h)

    data = [trace]

    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig, filename="World")

    

_generate_bar_plot_hor(trainv1,'world', "Count of world(section of appplication)", '#1E90FF', 600, 400)
sample=trainv1[trainv1['installation_id']==train_labels['installation_id'].unique()[1]]

train_labels[train_labels['installation_id']==sample['installation_id'].values[0]]
noofrows=len(sample[sample['game_session']=='197a373a77101924'])

noofuniquetype=sample[sample['game_session']=='197a373a77101924']['type'].nunique()

noofuniqueworld=sample[sample['game_session']=='197a373a77101924']['world'].nunique()

noofuniquetitle=sample[sample['game_session']=='197a373a77101924']['title'].nunique()



display(HTML(f"""<br>Number of rows for game session e6a6a262a8243ff7: {noofrows:,}</br>

             <br>Number of unique type for the session: {noofuniquetype:,}</br>

             <br>Number of unique world category for the session: {noofuniqueworld:,}</br>

             <br>Number of unique title for the session: {noofuniquetitle:,}</br>"""))
def time_feature(df):

    df['hour'] = df['timestamp'].dt.hour

    df['day'] = df['timestamp'].dt.day

    df['weekday'] =df['timestamp'].dt.weekday

    df['month'] = df['timestamp'].dt.month

    df['year'] = df['timestamp'].dt.year 

    df['date'] = df['timestamp'].dt.date 

    

    return df



time_train=time_feature(trainv1)
def scatter_plot(cnt_srs, color):

    trace = go.Scatter(

        x = cnt_srs.index,

        y = cnt_srs.values,

        showlegend = False,

        marker = dict(

            color = color,

        )

    )

    return trace



def get_time_plots(df):

    print('The dataset start on {} and ends on {}'.format(df['date'].min(), df['date'].max()))

    cnt_srs = df['date'].value_counts().sort_index()

    trace1 = scatter_plot(cnt_srs, 'red')

    cnt_srs = df['month'].value_counts().sort_index()

    trace2 = scatter_plot(cnt_srs, 'blue')

    cnt_srs = df['hour'].value_counts().sort_index()

    trace3 = scatter_plot(cnt_srs, 'green')

    cnt_srs = df['weekday'].value_counts().sort_index()

    trace4 = scatter_plot(cnt_srs, 'orange')

    

    subtitles = ['Date Frequency', 'Month Frequency', 'Hour Frequency', 'Day of Week Frequency']

    

    fig = subplots.make_subplots(rows = 4, cols = 1, vertical_spacing = 0.08, subplot_titles = subtitles)

    fig.append_trace(trace1, 1, 1)

    fig.append_trace(trace2, 2, 1)

    fig.append_trace(trace3, 3, 1)

    fig.append_trace(trace4, 4, 1)

    fig['layout'].update(height = 1200, width = 1000, paper_bgcolor = 'rgb(233, 233, 233)')

    py.iplot(fig, filename = 'time_plots')
get_time_plots(time_train)

del time_train
def plot_count(feature, title, df, size=5):

    f, ax = plt.subplots(1,1, figsize=(2*size,10))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:10], palette='Set2')

    g.set_title("Number and percentage of {}".format(title))

    

    if(size > 2):

        plt.xticks(rotation=90, size=8)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show() 
plot_count('title','Title(Activity) -distribution',trainv1[trainv1['type']=='Activity'])
plot_count('title','Title(Assessment) -distribution',trainv1[trainv1['type']=='Assessment'])
plot_count('title','Title(Clip) -distribution',trainv1[trainv1['type']=='Clip'])
trainv1 = trainv1[trainv1.installation_id.isin(train_labels.installation_id.unique())]

trainv1.shape
trainv1.drop(['hour','day','weekday','month','year'],axis=1,inplace=True)
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

trainv1, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code = encode_title(trainv1, test, train_labels)
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



for i, (ins_id, user_sample) in tqdm(enumerate(trainv1.groupby('installation_id', sort=False)), total=trainv1['installation_id'].nunique()):

    compiled_data += get_data(user_sample)

    

new_train = pd.DataFrame(compiled_data)
del compiled_data,trainv1

new_train.shape
new_train.head()
compiled_data = []

for ins_id, user_sample in tqdm(test.groupby('installation_id', sort=False), total=1000):

    a = get_data(user_sample, test_set=True)

    compiled_data.append(a)

    

new_test = pd.DataFrame(compiled_data)

del test,compiled_data
#cat_features = ['session_title']

new_test.head()
def preprocess(reduce_train, reduce_test):

    for df in [reduce_train, reduce_test]:

        df['installation_session_count'] = df.groupby(['installation_id'])['Clip'].transform('count')

        df['installation_duration_mean'] = df.groupby(['installation_id'])['duration_mean'].transform('mean')

        #df['installation_duration_std'] = df.groupby(['installation_id'])['duration_mean'].transform('std')

        df['installation_title_nunique'] = df.groupby(['installation_id'])['session_title'].transform('nunique')

        

        df['sum_event_code_count'] = df[[2050, 4100, 4230, 5000, 4235, 2060, 4110, 5010, 2070, 2075, 2080, 2081, 2083, 3110, 4010, 3120, 3121, 4020, 4021, 

                                        4022, 4025, 4030, 4031, 3010, 4035, 4040, 3020, 3021, 4045, 2000, 4050, 2010, 2020, 4070, 2025, 2030, 4080, 2035, 

                                        2040, 4090, 4220, 4095]].sum(axis = 1)

        

        df['installation_event_code_count_mean'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('mean')

     

    features = reduce_train.loc[(reduce_train.sum(axis=1) != 0), (reduce_train.sum(axis=0) != 0)].columns # delete useless columns

    

    return reduce_train, reduce_test, features



# call feature engineering function

reduce_train, reduce_test, features = preprocess(new_train, new_test)
cols_to_drop = ['installation_id']



features=[feat for feat in features if feat not in cols_to_drop ]



mytrain=reduce_train[features]

mytest=reduce_test[features]
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
def run_lgb(reduce_train, reduce_test):

    

    kf = StratifiedKFold(n_splits=10)

    features = [i for i in reduce_train.columns if i not in ['accuracy_group']]

    target = 'accuracy_group'

    oof_pred = np.zeros((len(reduce_train)))

    y_pred = np.zeros((len(reduce_test), 4))



    for fold, (tr_ind, val_ind) in enumerate(kf.split(reduce_train,reduce_train['accuracy_group'])):

        print('Fold {}'.format(fold + 1))

        x_train, x_val = reduce_train[features].iloc[tr_ind], reduce_train[features].iloc[val_ind]

        y_train, y_val = reduce_train[target][tr_ind], reduce_train[target][val_ind]

        train_set = lgb.Dataset(x_train, y_train)#, categorical_feature=cat_features)

        val_set = lgb.Dataset(x_val, y_val)#, categorical_feature=cat_features)



        params = {

            'learning_rate': 0.01,

            'metric': 'multiclass',

            'objective': 'multiclass',

            'num_classes': 4,

            'feature_fraction': 0.75,

            'subsample': 0.75,

            'n_estimators': 2000

        }

       

        model = lgb.train(params, train_set, num_boost_round = 10000, early_stopping_rounds = 100, 

                          valid_sets=[train_set, val_set], verbose_eval = 100)

        oof_pred[val_ind] = [np.argmax(i) for i in model.predict(x_val)]

        y_pred += model.predict(reduce_test[features]) / 10



        print('OOF QWK:', qwk(reduce_train[target], oof_pred))



    return y_pred,model

y_pred,modelobj = run_lgb(mytrain,mytest)
def run_xgb(reduce_train, reduce_test):

    

    kf = StratifiedKFold(n_splits=10)

    features = [i for i in reduce_train.columns if i not in ['accuracy_group']]

    target = 'accuracy_group'

    oof_pred = np.zeros((len(reduce_train)))

    y_pred = np.zeros((len(reduce_test),4))



    for fold, (tr_ind, val_ind) in enumerate(kf.split(reduce_train,reduce_train['accuracy_group'])):

        print('Fold {}'.format(fold + 1))

        x_train, x_val = reduce_train[features].iloc[tr_ind], reduce_train[features].iloc[val_ind]

        y_train, y_val = reduce_train[target][tr_ind], reduce_train[target][val_ind]

        

        train_set = xgb.DMatrix(x_train, y_train)

        val_set = xgb.DMatrix(x_val, y_val)

        test_set=xgb.DMatrix(reduce_test[features])

        val_ip = xgb.DMatrix(x_val)



        params = {

            'learning_rate': 0.5,

            'metric': 'mlogloss',

            'objective': 'multi:softprob',

            'feature_fraction': 0.75,

            'subsample': 1,

            'n_estimators': 2000,

            'num_class': 4

        }

       

        model = xgb.train(params, train_set, num_boost_round = 10000, early_stopping_rounds = 10, 

                          evals=[(train_set, 'train'), (val_set, 'val')], verbose_eval = True)

        oof_pred[val_ind] = [np.argmax(i) for i in model.predict(val_ip)]

        y_pred += model.predict(test_set) / 10



        print('OOF QWK:', qwk(reduce_train[target], oof_pred))



    return y_pred,model

y_pred_2,modelobj_2 = run_xgb(mytrain,mytest)
finalpred=y_pred+y_pred_2
all_features = [x for x in mytrain.columns if x not in ['accuracy_group']]



#select a random row

# row_to_show = random.randint(0,mytrain.shape[0])

# data_for_prediction = mytrain[all_features].iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired

# data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
explainer = shap.TreeExplainer(modelobj)

shap_values = explainer.shap_values(mytrain[all_features])

shap.initjs()

shap.force_plot(explainer.expected_value[0], shap_values[0][0].reshape(1,-1), mytrain[all_features].loc[7].values)
sns.set_style("whitegrid")

plt.title("Variable Importance Plot")

shap.summary_plot(shap_values, mytrain[all_features], plot_type="bar")
mytest['accuracy_group'] = finalpred.argmax(axis = 1)

sample_submission.drop('accuracy_group', inplace = True, axis = 1)

sample_submission = pd.concat([sample_submission,pd.DataFrame(mytest['accuracy_group'])],axis=1,ignore_index=True)

sample_submission.columns=['installation_id','accuracy_group']



sample_submission.to_csv('submission.csv', index = False)
plt.bar(sample_submission['accuracy_group'].value_counts().index,sample_submission['accuracy_group'].value_counts().values)