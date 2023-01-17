import numpy as np

import pandas as pd

import scipy.special

import matplotlib.pyplot as plt

import os

import json
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
path_in = '../input/data-science-bowl-2019/'

os.listdir(path_in)
train_data = pd.read_csv(path_in+'train.csv', parse_dates=['timestamp'])

train_labels = pd.read_csv(path_in+'train_labels.csv')

specs_data = pd.read_csv(path_in+'specs.csv')
def plot_bar(data, name, width, lenght):

    fig = plt.figure(figsize=(width, lenght))

    ax = fig.add_subplot(111)

    data_label = data[name].value_counts()

    dict_train = dict(zip(data_label.keys(), ((data_label.sort_index())).tolist()))

    names = list(dict_train.keys())

    values = list(dict_train.values())

    plt.bar(names, values)

    ax.set_xticklabels(names, rotation=45)

    plt.grid()

    plt.show()
print('# samples train_data:', len(train_data))

print('# samples train_labels:', len(train_labels))

print('# samples specs:', len(specs_data))
train_data.head()
train_labels.head()
specs_data.head()
cols_with_missing_train_data = [col for col in train_data.columns if train_data[col].isnull().any()]

cols_with_missing_train_labels = [col for col in train_labels.columns if train_labels[col].isnull().any()]

cols_with_missing_specs_data = [col for col in specs_data.columns if specs_data[col].isnull().any()]
print(cols_with_missing_train_data)

print(cols_with_missing_train_labels)

print(cols_with_missing_specs_data)
#train_data = train_data.loc[0: len(train_data.index)/233]
train_data.columns
train_data.dtypes
train_data['event_id'].value_counts()
train_data['game_session'].value_counts()
train_data['month'] = train_data['timestamp'].dt.month

train_data['day'] = train_data['timestamp'].dt.weekday

train_data['hour'] = train_data['timestamp'].dt.hour

train_data['weekend'] = np.where((train_data['day'] == 5) | (train_data['day'] == 6), 1, 0)
features_cyc = {'month' : 12, 'day' : 7, 'hour' : 24}

for feature in features_cyc.keys():

    train_data[feature+'_sin'] = np.sin((2*np.pi*train_data[feature])/features_cyc[feature])

    train_data[feature+'_cos'] = np.cos((2*np.pi*train_data[feature])/features_cyc[feature])

train_data = train_data.drop(features_cyc.keys(), axis=1)
encode_fields = ['description']

# steps = 233

# for i in range(steps):

#     print('work on step: ', i+1)

#     for encode_field in encode_fields:

#         slice_from = i*len(train_data.index)/steps

#         slice_to = (i+1)*len(train_data.index)/steps-1

#         train_data.loc[slice_from:slice_to, encode_field] = train_data.loc[slice_from:slice_to, 'event_data'].apply(json.loads).apply(pd.Series)[encode_field]

del train_data['event_data']
plot_bar(train_data, 'title', 30, 5)
map_train_title = dict(zip(train_data['title'].value_counts().sort_index().keys(),

                     range(1, len(train_data['title'].value_counts())+1)))
train_data['title'] = train_data['title'].replace(map_train_title)
plot_bar(train_data, 'type', 9, 5)
train_data = pd.get_dummies(train_data, columns=['type'])
plot_bar(train_data, 'world', 9, 5)
train_data = pd.get_dummies(train_data, columns=['world'])
train_labels.columns
plot_bar(train_labels, 'title', 9, 5)
map_label_title = dict(zip(train_labels['title'].value_counts().sort_index().keys(),

                     range(1, len(train_labels['title'].value_counts())+1)))
train_labels['title'] = train_labels['title'].replace(map_label_title)
train_labels['num_correct'].value_counts()
#train_labels['num_incorrect'].value_counts()
train_labels['accuracy'].describe()
plot_bar(train_labels, 'accuracy_group', 8, 4)
train_labels['accuracy_group'].value_counts().sort_index()
specs_data.columns
specs_data['event_id']
specs_data['info'].value_counts()
specs_data.loc[0, 'args']
train_data = pd.merge(train_data, train_labels,  how='right', on=['game_session','installation_id'])
no_features = ['accuracy_group', 'event_id', 'game_session', 'timestamp','installation_id',

              'accuracy', 'num_correct', 'num_incorrect']

X_train = train_data[train_data.columns.difference(no_features)].copy(deep=False)

y_train = train_data['accuracy_group']



del X_train['title_y']

X_train = X_train.rename(columns = {'title_x': 'title'})
len(X_train.index), len(train_data.index)
X_train.head()
del train_data
model = XGBClassifier(objective ='multi:softmax',

                      learning_rate = 0.2,

                      max_depth = 16,

                      n_estimators = 350,

                      random_state=2020,

                      num_class = 4)

model.fit(X_train,y_train)
del X_train, y_train
test_data = pd.read_csv(path_in+'test.csv', parse_dates=['timestamp'])

samp_subm = pd.read_csv(path_in+'sample_submission.csv')
""" Extract new features from timestamp """

test_data['month'] = test_data['timestamp'].dt.month

test_data['day'] = test_data['timestamp'].dt.weekday

test_data['hour'] = test_data['timestamp'].dt.hour

test_data['weekend'] = np.where((test_data['day'] == 5) | (test_data['day'] == 6), 1, 0)



""" Encode cyclic features """

features_cyc = {'month' : 12, 'day' : 7, 'hour' : 24}

for feature in features_cyc.keys():

    test_data[feature+'_sin'] = np.sin((2*np.pi*test_data[feature])/features_cyc[feature])

    test_data[feature+'_cos'] = np.cos((2*np.pi*test_data[feature])/features_cyc[feature])

test_data = test_data.drop(features_cyc.keys(), axis=1)



""" Encode feature title """

test_data['title'] = test_data['title'].replace(map_train_title)



""" Encode feature type """

test_data = pd.get_dummies(test_data, columns=['type'])



""" Encode feature world """

test_data = pd.get_dummies(test_data, columns=['world'])



""" Delete feature event_data """

del test_data['event_data']
X_test = test_data[test_data.columns.difference(no_features)].copy(deep=False)
y_test = model.predict(X_test)
y_temp = pd.DataFrame(y_test, index=test_data['installation_id'], columns=['accuracy_group'])
y_temp_grouped = y_temp.groupby(y_temp.index).agg(lambda x:x.value_counts().index[0])
output = pd.DataFrame({'installation_id': y_temp_grouped.index,

                       'accuracy_group': y_temp_grouped['accuracy_group']})

output.index = samp_subm.index

output.to_csv('submission.csv', index=False)
output.head()
output['accuracy_group'].value_counts().sort_index()