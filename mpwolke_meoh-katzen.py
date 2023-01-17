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



# Any results you write to the current directory are saved as output.
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier

import scipy.special

import matplotlib.pyplot as plt
path_in = '../input/cat-in-the-dat-ii/'

print(os.listdir(path_in))
train_data = pd.read_csv(path_in+'train.csv', index_col=0)

test_data = pd.read_csv(path_in+'test.csv', index_col=0)

samp_subm = pd.read_csv(path_in+'sample_submission.csv', index_col=0)
test_data.head()
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