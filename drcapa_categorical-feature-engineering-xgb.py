import numpy as np

import pandas as pd

import scipy.special

import matplotlib.pyplot as plt

import os
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier
path_in = '../input/cat-in-the-dat/'

print(os.listdir(path_in))
train_data = pd.read_csv(path_in+'train.csv', index_col=0)

test_data = pd.read_csv(path_in+'test.csv', index_col=0)

samp_subm = pd.read_csv(path_in+'sample_submission.csv', index_col=0)
def plot_bar(data, name):

    data_label = data[name].value_counts()

    dict_train = dict(zip(data_label.keys(), ((data_label.sort_index())).tolist()))

    names = list(dict_train.keys())

    values = list(dict_train.values())

    plt.bar(names, values)

    plt.grid()

    plt.show()
def plot_bar_compare(train, test, name, rot=False):

    """ Compare the distribution between train and test data """

    fig, axs = plt.subplots(1, 2, figsize=(9, 3), sharey=True)

    

    train_label = train[name].value_counts().sort_index()

    dict_train = dict(zip(train_label.keys(), ((100*(train_label)/len(train.index)).tolist())))

    train_names = list(dict_train.keys())

    train_values = list(dict_train.values())

    

    test_label = test[name].value_counts().sort_index()

    dict_test = dict(zip(test_label.keys(), ((100*(test_label)/len(test.index)).tolist())))

    test_names = list(dict_test.keys())

    test_values = list(dict_test.values())

    

    axs[0].bar(train_names, train_values, color='yellowgreen')

    axs[1].bar(test_names, test_values, color = 'sandybrown')

    axs[0].grid()

    axs[1].grid()

    axs[0].set_title('Train data')

    axs[1].set_title('Test data')

    axs[0].set_ylabel('%')

    if(rot==True):

        axs[0].set_xticklabels(train_names, rotation=45)

        axs[1].set_xticklabels(test_names, rotation=45)

    plt.show()
print('# samples train:', len(train_data))

print('# samples test:', len(test_data))
cols_with_missing_train_data = [col for col in train_data.columns if train_data[col].isnull().any()]

cols_with_missing_test_data = [col for col in test_data.columns if test_data[col].isnull().any()]

print('train cols with missing data:', cols_with_missing_train_data)

print('test cols with missing data:', cols_with_missing_test_data)
train_data.columns
train_data.head()
plot_bar(train_data, 'target')
features_bin = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']

features_cat = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']

features_hex = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']

features_ord = ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']

features_cyc = ['day', 'month']
map_ord_1 = {'Novice':1, 'Contributor':2, 'Expert':3, 'Master':4, 'Grandmaster':5}

map_ord_2 = {'Freezing': 1, 'Cold':2, 'Warm':3, 'Hot':4, 'Boiling Hot': 5, 'Lava Hot':6}

map_ord_3 = dict(zip(train_data['ord_3'].value_counts().sort_index().keys(),

                     range(1, len(train_data['ord_3'].value_counts())+1)))

map_ord_4 = dict(zip(train_data['ord_4'].value_counts().sort_index().keys(),

                     range(1, len(train_data['ord_4'].value_counts())+1)))
temp_ord_5 = pd.DataFrame(train_data['ord_5'].value_counts().sort_index().keys(), columns=['ord_5'])

temp_ord_5['First'] = temp_ord_5['ord_5'].astype(str).str[0].str.upper()

temp_ord_5['Second'] = temp_ord_5['ord_5'].astype(str).str[1].str.upper()

temp_ord_5['First'] = temp_ord_5['First'].replace(map_ord_4)

temp_ord_5['Second'] = temp_ord_5['Second'].replace(map_ord_4)

temp_ord_5['Add'] = temp_ord_5['First']+temp_ord_5['Second']

temp_ord_5['Mul'] = temp_ord_5['First']*temp_ord_5['Second']

map_ord_5 = dict(zip(temp_ord_5['ord_5'],

                     temp_ord_5['Mul']))
plot_bar_compare(train_data, test_data, 'nom_0')
train_data['rgb'] = np.where(train_data['nom_0'] == 'Green', 0, 1)

test_data['rgb'] = np.where(test_data['nom_0'] == 'Green', 0, 1)
plot_bar_compare(train_data, test_data, 'nom_1', rot=True)
train_data['round'] = np.where(train_data['nom_1'] == 'Circle', 1, 0)

test_data['round'] = np.where(test_data['nom_1'] == 'Circle', 1, 0)
plot_bar_compare(train_data, test_data, 'nom_2', rot=True)
train_data['feet'] = np.where(train_data['nom_2'] == 'Snake', 0, 1)

test_data['feet'] = np.where(test_data['nom_2'] == 'Snake', 0, 1)
plot_bar_compare(train_data, test_data, 'nom_3', rot=True)
train_data['monarchy'] = np.where(train_data['nom_3'] == 'Canada', 1, 0)

test_data['monarchy'] = np.where(test_data['nom_3'] == 'Canada', 1, 0)
plot_bar_compare(train_data, test_data, 'nom_4', rot=True)
train_data['electro'] = np.where(train_data['nom_4'] == 'Theremin', 1, 0)

test_data['electro'] = np.where(test_data['nom_4'] == 'Theremin', 1, 0)
y_train = train_data['target']

del train_data['target']
X_train = train_data.copy()

X_test = test_data.copy()
le = LabelEncoder()

for col in features_bin:

    le.fit(X_train[col])

    X_train[col] = le.transform(X_train[col])

    X_test[col] = le.transform(X_test[col])
le = LabelEncoder()

for col in features_cat:

    le.fit(X_train[col])

    X_train[col] = le.transform(X_train[col])

    X_test[col] = le.transform(X_test[col])
le = LabelEncoder()

for col in features_hex:

    le.fit(X_train[col].append(X_test[col]))

    X_train[col] = le.transform(X_train[col])

    X_test[col] = le.transform(X_test[col])
X_train['ord_1'] = X_train['ord_1'].replace(map_ord_1)

X_train['ord_2'] = X_train['ord_2'].replace(map_ord_2)

X_train['ord_3'] = X_train['ord_3'].replace(map_ord_3)

X_train['ord_4'] = X_train['ord_4'].replace(map_ord_4)

X_train['ord_5'] = X_train['ord_5'].replace(map_ord_5)

X_test['ord_1'] = X_test['ord_1'].replace(map_ord_1)

X_test['ord_2'] = X_test['ord_2'].replace(map_ord_2)

X_test['ord_3'] = X_test['ord_3'].replace(map_ord_3)

X_test['ord_4'] = X_test['ord_4'].replace(map_ord_4)

X_test['ord_5'] = X_test['ord_5'].replace(map_ord_5)
for feature in features_cyc:

    X_train[feature+'_sin'] = np.sin((2*np.pi*X_train[feature])/max(X_train[feature]))

    X_train[feature+'_cos'] = np.cos((2*np.pi*X_train[feature])/max(X_train[feature]))

    X_test[feature+'_sin'] = np.sin((2*np.pi*X_test[feature])/max(X_test[feature]))

    X_test[feature+'_cos'] = np.cos((2*np.pi*X_test[feature])/max(X_test[feature]))

X_train = X_train.drop(features_cyc, axis=1)

X_test = X_test.drop(features_cyc, axis=1)
mean = X_train[features_hex].mean(axis=0)

X_train[features_hex] = X_train[features_hex].astype('float32')

X_train[features_hex] -= X_train[features_hex].mean(axis=0)

std = X_train[features_hex].std(axis=0)

X_train[features_hex] /= X_train[features_hex].std(axis=0)

X_test[features_hex] = X_test[features_hex].astype('float32')

X_test[features_hex] -= mean

X_test[features_hex] /= std
mean = X_train[features_ord].mean(axis=0)

X_train[features_ord] = X_train[features_ord].astype('float32')

X_train[features_ord] -= X_train[features_ord].mean(axis=0)

std = X_train[features_ord].std(axis=0)

X_train[features_ord] /= X_train[features_ord].std(axis=0)

X_test[features_ord] = X_test[features_ord].astype('float32')

X_test[features_ord] -= mean

X_test[features_ord] /= std
mean = X_train[features_cat].mean(axis=0)

X_train[features_cat] = X_train[features_cat].astype('float32')

X_train[features_cat] -= X_train[features_cat].mean(axis=0)

std = X_train[features_cat].std(axis=0)

X_train[features_cat] /= X_train[features_cat].std(axis=0)

X_test[features_cat] = X_test[features_cat].astype('float32')

X_test[features_cat] -= mean

X_test[features_cat] /= std
X_train.columns
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.001, random_state=2020)
weight = float(len(y_train[y_train == 0]))/float(len(y_train[y_train == 1]))

w1 = np.array([1]*y_train.shape[0])

w1[y_train==1]=weight
X_train[features_cat]
model = XGBClassifier(objective ='binary:logistic',

                      colsample_bytree = 0,

                      learning_rate = 0.2,

                      max_depth = 15,

                      n_estimators = 1555,

                      scale_pos_weight=2,

                      random_state=2019,

                      subsample=0.8)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)#, sample_weight=w1)
preds_val = model.predict_proba(X_val)[:,1]
score = roc_auc_score(y_val ,preds_val)

print("score: %f" % (score))
y_test = model.predict_proba(X_test)[:,1]
num = samp_subm.index

output = pd.DataFrame({'id': num,

                       'target': y_test})

output.to_csv('submission.csv', index=False)