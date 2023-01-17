# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input', topdown = True):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly as py

import plotly.express as px

import plotly.graph_objs as pgo

from plotly.subplots import make_subplots

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from wordcloud import WordCloud



init_notebook_mode(connected=True) 



import warnings

warnings.filterwarnings('ignore')
import pandas as pd

path = "/kaggle/input/data-science-bowl-2019"

df_sample_submission = pd.read_csv(f"{path}/sample_submission.csv")

df_specs = pd.read_csv(f"{path}/specs.csv")

df_test = pd.read_csv(f"{path}/test.csv")

df_train = pd.read_csv(f"{path}/train.csv")

df_train_labels = pd.read_csv(f"{path}/train_labels.csv")
def data_description(data):

    return (data.describe())



def data_info(data):

    return (data.info())



def data_head(data):

    return (data.head())



def data_column(data):

    return (data.column)
data_description(df_train)

data_info(df_train)

data_head(df_train)
print ("Train Size: {0} \n Test Size: {1}" .format(df_train.shape, df_test.shape))
df_train_labels.head()

df_train_labels.shape
pd.set_option("max_colwidth", 150)

data_head(df_test)
pd.set_option("max_colwidth", 150)

data_head(df_specs)
pd.set_option("max_colwidth", 150)

data_head(df_train_labels)
print(f"df_train_installation_id: {df_train.installation_id.nunique()}")

print(f"df_train_labels_installation_id: {df_train_labels.installation_id.nunique()}")

print(f"df_test_installation_id: {df_test.installation_id.nunique()}")

print(f"test&submission_installation_id_check: {set(df_test.installation_id.unique()) == set(df_sample_submission.installation_id.unique())}")

print(f"train&test titles_check: {set(df_test.title.unique()) == set(df_train.title.unique())}")
def missing_data(data):

    total = data.isnull().sum()

    percent = (total/data.isnull().count()*100)

    missing_columns = pd.concat([total, percent], axis = 1, keys = ["Total", "Percent"])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    missing_columns["Types"] = types

    return(np.transpose(missing_columns))
missing_data(df_train)
missing_data(df_test)
missing_data(df_specs)
missing_data(df_train_labels)
print(f"Counting Rows in train set: {df_train.shape[0]}")

print(f"Counting Rows in train_labels set: {df_train_labels.shape[0]}")

print(f"Counting Rows in test set: {df_test.shape[0]}")

print(f"Counting Rows in specs set: {df_specs.shape[0]}")

for column in df_train.columns.values:

    print(f"Counts of Unique values in train_set ´{column}`: {df_train[column].nunique()}")
for column in df_train_labels.columns.values:

    print(f"Counts of Unique values in train_labels ´{column}`: {df_train_labels[column].nunique()}")
df_train_labels['title'].apply(lambda x: x if x != {} else 0).value_counts()
for column in df_test.columns.values:

    print(f"Counts of Unique values in test_set ´{column}`: {df_test[column].nunique()}")
for column in df_specs.columns.values:

    print(f"Counts of Unique values in Specs ´{column}`: {df_specs[column].nunique()}")
import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



trace = go.Pie(labels = df_train['title'].value_counts().index,

              values = df_train['title'].value_counts().values,

              domain = {'x':[0.20,1]})



data = [trace]

layout = go.Layout(title = 'PieChat Distribution of titles in train_labels')

fig = go.Figure(data = data, layout = layout)

py.iplot(fig)
trace = go.Pie(labels = df_train_labels['title'].value_counts().index,

              values = df_train_labels['title'].value_counts().values,

              domain = {'x':[0.20,1]})



data = [trace]

layout = go.Layout(title = 'PieChat Distribution of Titles in train_labels')

fig = go.Figure(data = data, layout = layout)

py.iplot(fig)
def plot_count(feature, title, df, size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:30], palette='Set3')

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
plot_count('title', 'title (first most frequent 30 values - train_labels)', df_train_labels, size=6)
trace = go.Pie(labels = df_train['type'].value_counts().index,

              values = df_train['type'].value_counts().values,

              domain = {'x':[0.20,1]})



data = [trace]

layout = go.Layout(title = 'PieChat Distribution of type in train')

fig = go.Figure(data = data, layout = layout)

py.iplot(fig)
trace = go.Pie(labels = df_test['type'].value_counts().index,

              values = df_test['type'].value_counts().values,

              domain = {'x':[0.20,1]})



data = [trace]

layout = go.Layout(title = 'PieChat Distribution of type in test')

fig = go.Figure(data = data, layout = layout)

py.iplot(fig)
trace = go.Pie(labels = df_train['world'].value_counts().index,

              values = df_train['world'].value_counts().values,

              domain = {'x':[0.20,1]})



data = [trace]

layout = go.Layout(title = 'PieChat Distribution of world in train')

fig = go.Figure(data = data, layout = layout)

py.iplot(fig)
trace = go.Pie(labels = df_test['world'].value_counts().index,

              values = df_test['world'].value_counts().values,

              domain = {'x':[0.20,1]})



data = [trace]

layout = go.Layout(title = 'PieChat Distribution of world in test')

fig = go.Figure(data = data, layout = layout)

py.iplot(fig)
extracted_samples_train = df_train.sample(100000)

data_head(extracted_samples_train)
extracted_samples_train.iloc[0].event_data

import json



extracted_event_data_samples_train = pd.io.json.json_normalize(extracted_samples_train.event_data.apply(json.loads))
print(f"extracted_event_data_shape: {extracted_event_data_samples_train.shape}")

data_head(extracted_event_data_samples_train)
missing_data(extracted_event_data_samples_train)
extracted_event_data_samples_train.columns

def existing_data(data):

    total = data.isnull().count() - data.isnull().sum()

    percent = 100 - (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    tt = pd.DataFrame(tt.reset_index())

    return(tt.sort_values(['Total'], ascending=False))
stat_extracted_event_data_samples_train = existing_data(extracted_event_data_samples_train)



plt.figure(figsize=(10, 10))

sns.set(style='whitegrid')

ax = sns.barplot(x='Percent', y='index', data=stat_extracted_event_data_samples_train.head(50), color='blue')

plt.title('Most frequent features in extracted_event_data_samples_train')

plt.ylabel('features')
df_specs.args[0]


extracted_specs_args = pd.DataFrame()

for i in range(0, df_specs.shape[0]): 

    for arg_item in json.loads(df_specs.args[i]) :

        new_df = pd.DataFrame({'event_id': df_specs['event_id'][i],\

                               'info':df_specs['info'][i],\

                               'args_name': arg_item['name'],\

                               'args_type': arg_item['type'],\

                               'args_info': arg_item['info']}, index=[i])

        extracted_specs_args = extracted_specs_args.append(new_df)
print(f"extracted args from specs: {extracted_specs_args.shape}")



data_head(extracted_specs_args)
tmp = extracted_specs_args.groupby(['event_id'])['info'].count()

df = pd.DataFrame({'event_id':tmp.index, 'count': tmp.values})

plt.figure(figsize=(6,4))

sns.set(style='whitegrid')

ax = sns.distplot(df['count'],kde=True,hist=False, bins=40)

plt.title('Distribution of number of arguments per event_id')

plt.xlabel('Number of arguments'); plt.ylabel('Density'); plt.show()
plot_count('args_name', 'args_name (first most frequent 30 values - specs)', extracted_specs_args, size=6)
from typing import Any

import re



def add_datepart(df: pd.DataFrame, field_name: str,

                 prefix: str = None, drop: bool = True, time: bool = True, date: bool = True):

    """

    Helper function that adds columns relevant to a date in the column `field_name` of `df`.

    from fastai: https://github.com/fastai/fastai/blob/master/fastai/tabular/transform.py#L55

    """

    field = df[field_name]

    prefix = ifnone(prefix, re.sub('[Dd]ate$', '', field_name))

    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Is_month_end', 'Is_month_start']

    if date:

        attr.append('Date')

    if time:

        attr = attr + ['Hour', 'Minute']

    for n in attr:

        df[prefix + n] = getattr(field.dt, n.lower())

    if drop:

        df.drop(field_name, axis=1, inplace=True)

    return df





def ifnone(a: Any, b: Any) -> Any:

    """`a` if `a` is not None, otherwise `b`.

    from fastai: https://github.com/fastai/fastai/blob/master/fastai/core.py#L92"""

    return b if a is None else a
from joblib import Parallel, delayed

from collections import defaultdict

import copy





class FeatureGenerator(object):

    def __init__(self, n_jobs=1, df=None, dataset: str = 'df_train'):

        self.n_jobs = n_jobs

        self.df = df

        self.dataset = dataset



    def read_chunks(self):

        for id, user_sample in self.df.groupby('installation_id', sort=False):

            yield id, user_sample



    def get_features(self, row):

        """

        Gets three groups of features: from original data and from read and imaginary parts of FFT.

        """

        return self.features(row)



    def features(self, id, user_sample):

        user_data = []



        accuracy_mapping = {0: 0, 1: 3, 0.5: 2}



        user_stats = defaultdict(int)

        user_stats['installation_id'] = user_sample['installation_id'].unique()[0]

        user_stats['world'] = user_sample['world'].unique()[0]

        user_stats['timestamp'] = user_sample['timestamp'].unique()[0]



        temp_dict = defaultdict(int)

        another_temp_dict = {}

        another_temp_dict['durations'] = []

        another_temp_dict['all_durations'] = []

        another_temp_dict['durations_with_triers'] = []

        another_temp_dict['mean_action_time'] = []

        title_data = defaultdict(dict)



        for i, session in user_sample.groupby('game_session', sort=False):

            user_stats['last_ass_session_game_time'] = another_temp_dict['durations'][-1] if len(another_temp_dict['durations']) > 0 else 0

            user_stats['last_session_game_time'] = another_temp_dict['all_durations'][-1] if len(another_temp_dict['all_durations']) > 0 else 0



            # calculate some user_stats and append data

            if session['trier'].sum() > 0 or self.dataset == 'df_test':

                user_stats['session_title'] = session['title'].values[0]

                accuracy = np.nan_to_num(session['correct'].sum() / session['trier'].sum())

                if accuracy in accuracy_mapping.keys():

                    user_stats['accuracy_group'] = accuracy_mapping[accuracy]

                else:

                    user_stats['accuracy_group'] = 1

                user_stats['accumulated_accuracy_group'] = temp_dict['accumulated_accuracy_group'] / user_stats['counter'] if user_stats['counter'] > 0 else 0

                temp_dict['accumulated_accuracy_group'] += user_stats['accuracy_group']

                user_data.append(copy.copy(user_stats))



            user_stats[session['type'].values[-1]] += 1

            user_stats['accumulated_correct_attempts'] += session['correct'].sum()

            user_stats['accumulated_uncorrect_attempts'] += session['trier'].sum() - session['correct'].sum()

            event_code_counts = session['event_code'].value_counts()

            for i, j in zip(event_code_counts.index, event_code_counts.values):

                user_stats[i] += j



            temp_dict['assessment_counter'] += 1

            if session['title'].values[-1] in title_data.keys():

                pass

            else:

                title_data[session['title'].values[-1]] = defaultdict(int)



            title_data[session['title'].values[-1]]['duration_all'] += session['game_time'].values[-1]

            title_data[session['title'].values[-1]]['counter_all'] += 1

            #user_stats['duration'] += (session['timestamp'].values[-1] - session['timestamp'].values[0]) / np.timedelta64(1, 's')



            user_stats['duration'] = (session.iloc[-1,2] - session.iloc[0,2]).seconds

            if session['type'].values[0] == 'Assessment' and (len(session) > 1 or self.dataset == 'df_test'):

                another_temp_dict['durations'].append(user_stats['duration'])

                accuracy = np.nan_to_num(session['correct'].sum() / session['trier'].sum())

                user_stats['accumulated_accuracy_'] += accuracy

                user_stats['counter'] += 1

                if user_stats['counter'] == 0:

                    user_stats['accumulated_accuracy'] = 0

                else:

                    user_stats['accumulated_accuracy'] = user_stats['accumulated_accuracy_'] / user_stats['counter']



                accuracy = np.nan_to_num(session['correct'].sum() / session['trier'].sum())



                if accuracy in accuracy_mapping.keys():

                    user_stats[accuracy_mapping[accuracy]] += 1

                else:

                    user_stats[1] += 1



                user_stats['accumulated_actions'] += len(session)



                if session['trier'].sum() > 0:

                    user_stats['sessions_with_trier'] += 1

                    another_temp_dict['durations_with_triers'].append(user_stats['duration'])



                if session['correct'].sum() > 0:

                    user_stats['sessions_with_correct_trier'] += 1

                    

                user_stats['title_duration'] = title_data[session['title'].values[-1]]['duration']

                user_stats['title_counter'] = title_data[session['title'].values[-1]]['counter']

                user_stats['title_mean_duration'] = user_stats['title_duration'] / user_stats['title_mean_duration']  if user_stats['title_mean_duration'] > 0 else 0



                user_stats['title_duration_all'] = title_data[session['title'].values[-1]]['duration_all']

                user_stats['title_counter_all'] = title_data[session['title'].values[-1]]['counter_all']

                user_stats['title_mean_duration_all'] = user_stats['title_duration_all'] / user_stats['title_mean_duration_all']  if user_stats['title_mean_duration_all'] > 0 else 0

                

                title_data[session['title'].values[-1]]['duration'] += session['game_time'].values[-1]

                title_data[session['title'].values[-1]]['counter'] += 1



            elif (len(session) > 1 or self.dataset == 'df_test'):

                another_temp_dict['all_durations'].append(user_stats['duration'])





            if user_stats['duration'] != 0:

                temp_dict['nonzero_duration_assessment_counter'] += 1

            #user_stats['duration_mean'] = user_stats['duration'] / max(temp_dict['nonzero_duration_assessment_counter'], 1)

            # stats from assessment sessions

            user_stats['duration_mean'] = np.mean(another_temp_dict['durations'])

            user_stats['duration_trier'] = np.mean(another_temp_dict['durations_with_triers'])



            # stats from all sessions

            user_stats['all_duration_mean'] = np.mean(another_temp_dict['all_durations'])

            user_stats['all_accumulated_actions'] += len(session)

            user_stats['mean_action_time'] = np.mean(another_temp_dict['mean_action_time'])

            another_temp_dict['mean_action_time'].append(session['game_time'].values[-1] / len(session))





        if self.dataset == 'df_test':

            user_data = [user_data[-1]]



        return user_data



    def generate(self):

        feature_list = []

#         res = Parallel(n_jobs=self.n_jobs, backend='threading')(delayed(self.features)(id, user_sample)

#                                                                 for id, user_sample in tqdm(self.read_chunks(),

#                                                                                             total=self.df[

#                                                                                                 'installation_id'].nunique()))

        res = Parallel(n_jobs=self.n_jobs, backend='threading')(delayed(self.features)(id, user_sample)

                                                                for id, user_sample in self.read_chunks())

        for r in res:

            for r1 in r:

                feature_list.append(r1)

        return pd.DataFrame(feature_list)


df_train["trier"] = 0

df_train.loc[(df_train["title"]=="Bird Measurer (Assessment)")&(df_train["event_code"]==4110),"trier"] = 1

df_train.loc[(df_train["type"]=="Assessment")&(df_train["title"]!="Bird Measurer (Assessment)")&(df_train["event_code"]==4100),"trier"] = 1
from category_encoders.ordinal import OrdinalEncoder



title_encode = OrdinalEncoder()

title_encode.fit(list(set(df_train['title'].unique()).union(set(df_test['title'].unique()))));

world_encode = OrdinalEncoder()

world_encode.fit(list(set(df_train['world'].unique()).union(set(df_test['world'].unique()))));
df_train['correct'] = None

df_train.loc[(df_train['trier'] == 1) & (df_train['event_data'].str.contains('"correct":true')), 'correct'] = True

df_train.loc[(df_train['trier'] == 1) & (df_train['event_data'].str.contains('"correct":false')), 'correct'] = False
df_train['title'] = title_encode.transform(df_train['title'].values)

df_train['world'] = world_encode.transform(df_train['world'].values)

df_train = df_train.loc[df_train['installation_id'].isin(df_train_labels['installation_id'].unique())]
%%time

FG = FeatureGenerator(n_jobs=2, df=df_train)

train_set = FG.generate()

train_set = train_set.fillna(0)
data_head(train_set)
df_train["timestamp"] = pd.to_datetime(df_train["timestamp"])



extracted_train_timestamp = add_datepart(df_train, "timestamp", drop = False)
data_head(extracted_train_timestamp)