import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

from time import time

from tqdm import tqdm_notebook as tqdm

from collections import Counter

from scipy import stats

import lightgbm as lgb

from sklearn.metrics import cohen_kappa_score

from sklearn.model_selection import KFold, StratifiedKFold

import gc

import json

pd.set_option('display.max_columns', 1000)

import matplotlib.pyplot as plt

from plotly.subplots import make_subplots

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go

import seaborn as sns

sns.set_style("whitegrid")


def existing_data(data):

    total = data.isnull().count() - data.isnull().sum()

    percent = 100 - (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    tt = pd.DataFrame(tt.reset_index())

    return(tt.sort_values(['Total'], ascending=False))
def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return(np.transpose(tt))
def read_data():

    print(f'Read data')

    train_df = pd.read_csv('../input/data-science-bowl-2019/train.csv')

    test_df = pd.read_csv('../input/data-science-bowl-2019/test.csv')

    train_labels_df = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')

    specs_df = pd.read_csv('../input/data-science-bowl-2019/specs.csv')

    sample_submission_df = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')

    print(f"train shape: {train_df.shape}")

    print(f"test shape: {test_df.shape}")

    print(f"train labels shape: {train_labels_df.shape}")

    print(f"specs shape: {specs_df.shape}")

    print(f"sample submission shape: {sample_submission_df.shape}")

    return train_df, test_df, train_labels_df, specs_df, sample_submission_df
train_df, test_df, train_labels_df, specs_df, sample_submission_df = read_data()
total = train_df.isnull().sum().sort_values(ascending = False)

percent = (train_df.isnull().sum()/train_df.isnull().count() * 100).sort_values(ascending = False)

missing_data = pd.concat([total, percent], keys = ['Total', 'Percent'], axis = 1)

missing_data
total = test_df.isnull().sum().sort_values(ascending = False)

precent = (train_df.isnull().sum()/train_df.isnull().count()*100).sort_values(ascending = False)

missing_data = pd.concat([total, percent], keys = ['Total', 'Percent'], axis = 1)

missing_data
total = train_labels_df.isnull().sum().sort_values(ascending = False)

percent = (train_labels_df.isnull().sum()/train_labels_df.isnull().count()* 100).sort_values(ascending = False)

missing_data = pd.concat([total, percent], keys = ['Total', 'Percent'], axis = 1)

missing_data
train_df.groupby('world')['game_session'].count().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,10), title = 'world distribution in train set', rot = 360)
train_df.groupby('type')['game_session'].count().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,10), rot = 360)
train_df.groupby('event_code')['game_session'].count().sort_values(ascending = True).plot(kind = 'barh', figsize = (20,10), title = 'Event_code distribution in train set')
train_types = train_df['type'].value_counts()

test_types = train_df['type'].value_counts()
train_types
fig = make_subplots(rows=1, cols =2, specs = [[{'type':'domain'}, {'type': 'domain'}]])



fig.add_trace(go.Pie(values = train_types, labels = train_types.index.tolist(), name = 'Train', hole = .3), 1,1)

fig.add_trace(go.Pie(values=test_types, labels=test_types.index.tolist(), name="Test" , hole=.3),1, 2)



fig.update_traces(hoverinfo = 'label + percent + value', textinfo = 'percent', textfont_size = 17, textposition = 'inside', marker = dict(colors = ['gold', 'mediumturquoise', 'darkorange', 'plum'], line = dict(color = '#000000', width = 3)))



fig.update_layout(title_text = 'Media Type of The Game or Video', height = 500, width = 800, annotations = [dict(text = 'Train', x = 0.18, y=0.5, font_size = 20, showarrow = False), dict(text='Test', x = 0.82, font_size = 20, showarrow = False)])

train_df.groupby('installation_id').count()['event_id'].plot(kind = 'hist', title = 'distribution of installation_id ', figsize = (15,5), bins = 40)
train_df.groupby('installation_id').count()['event_id'].apply(np.log1p).plot(kind = 'hist', title = 'distribution of installation_id in train set (log1p scale) ', bins = 40, figsize = (15,5))
train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])

train_df['date'] = train_df['timestamp'].dt.date

train_df['hour'] = train_df['timestamp'].dt.hour

train_df['weekday_name'] = train_df['timestamp'].dt.weekday_name
train_df
train_df.groupby('date')['event_id'].count().plot(figsize = (20,5), color = 'green')
train_df.groupby('hour')['event_id'].count().plot(figsize = (20,5), color = 'red') 
train_df.groupby('weekday_name')['event_id'].count().T[['Monday','Tuesday','Wednesday',

                     'Thursday','Friday','Saturday',

                     'Sunday']].plot(figsize = (15,5))
train_df['game_time'].plot(kind = 'hist', bins = 100, title = 'Game time', figsize = (15,5))
train_df['game_time'].apply(np.log1p).plot(kind = 'hist', bins = 100 ,figsize = (15,5) ,title = 'Log1p scale of Game time')
train_df.groupby('title')['event_id'].count().sort_values(ascending = True).plot(kind = 'barh', figsize = (15,15), title = 'Game/Vedeo title')
train_df.groupby('event_code').count()['event_id'].sort_values(ascending = False).plot(kind = 'bar', figsize = (15,5))
sample_train = train_df.sample(100000)
extracted_event_data = pd.io.json.json_normalize(sample_train.event_data.apply(json.loads))
extracted_event_data.head(15)
missing_data(extracted_event_data)
stat_event_data = existing_data(extracted_event_data)

stat_event_data
stat_event_data[['index','Percent']].head()
specs_df['args'][0]
specs_args_extracted = pd.DataFrame()

for i in range(0, specs_df.shape[0]):

    for arg_item in json.loads(specs_df.args[i]):

        new_df = pd.DataFrame({'event_id':specs_df['event_id'][i],

                              'info': specs_df['info'][i],

                              'args_name': arg_item['name'],

                              'args_type': arg_item['type'],

                              'args_info': arg_item['info']}, index = [i])

        specs_args_extracted = specs_args_extracted.append(new_df)
specs_args_extracted
tmp = specs_args_extracted.groupby(['event_id'])['info'].count()

df = pd.DataFrame({'event_id': tmp.index, 'count': tmp.values})

plt.figure(figsize = (6,4))

ax = sns.distplot(df['count'], kde = True, hist = False, bins = 40)

plt.title('Distribution of number of arguments per event_id')

plt.xlabel('Number of arguments'); plt.ylabel('Density')
train_df[(train_df.event_code == 4100) & (train_df.installation_id == '0006a69f') & (train_df.title == 'Bird Measurer (Assessment)')]['event_data'].tolist()
train_df[(train_df.installation_id == '0006a69f') & ((train_df.type == "Assessment") & (train_df.title == 'Bird Measurer (Assessment)') & (train_df.event_code == 4110) | (train_df.type == 'Assessment') & (train_df.title != ''))]