import numpy as np # linear algebra

import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plot

%matplotlib inline

import seaborn as sns # plot

import pandas_profiling as pdp # data profiling

from collections import OrderedDict # data structure



# Input data files are available in the "../input/" directory.

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# load the data

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
train.head()
# show column names

train.columns
# a function which plots survival rate about a peculiar column

def survived_rate_plot(column, str_option=True):

    """a function which returns survived rate of a peculiar column"""

    def return_index(lst, item):

        for i in range(len(lst)):

            if lst[i] == item:

                return i

        return None

    

    # by_column : total count grouped by "column"

    by_column = np.array(train[column].value_counts())

    by_column_index =  train[column].value_counts().index

    by_column_dict = dict(zip(by_column_index, by_column))



    # survived_by_column : survival count grouped by "column"

    survived_by_column = list(train[train['Survived'] == 1][column].value_counts())

    survived_by_column_index =  train[train['Survived'] == 1][column].value_counts().index

    temp_dict = dict(zip(survived_by_column_index, survived_by_column))

    survived_by_column_dict = OrderedDict()

    for key in by_column_dict.keys():

        if key in survived_by_column_index:

            survived_by_column_dict[key] = temp_dict[key]

        else:

            survived_by_column_dict[key] = 0

    temp = survived_by_column_dict.items()

    survived_by_column = np.array([t[1] for t in temp])



    # dead_by_column: dead count grouped by "column"

    dead_by_column = by_column - survived_by_column

    

    # survival_rate

    survival_rate = ((survived_by_column / by_column) * 100).astype(int)



    # option: if str_option is true (default), it translates int to str

    if str_option:

        left = by_column_index.astype(str)

    else:

        left = by_column_index

    

    # plotting

    plt.figure(figsize=(12,6))

    plt.bar(left, survived_by_column, color="limegreen", label='alive')

    plt.bar(left, dead_by_column, bottom=survived_by_column, color="orangered", label='dead')

    plt.title(f'dead or alive grouped by {column}', fontsize=20)

    plt.legend()

    

    # write down numbers in bar plots

    for x, y in zip(left, survival_rate):

        try:

            plt.text(x, by_column_dict[x], y, ha='center', va='bottom')

        except:

            x = int(x)

            plt.text(x, by_column_dict[x], y, ha='center', va='bottom')

    
# ceil values of "Age" to get more concise in data plot

train['Age'] = np.ceil(train['Age'])

survived_rate_plot('Age', str_option=False)
survived_rate_plot('Pclass')
survived_rate_plot('Sex')
survived_rate_plot('SibSp')
survived_rate_plot('Parch')
# classify "Cabin" based on its capital letter

train['Cabin'] = [item[0] if type(item) == str else item for item in train['Cabin']]

survived_rate_plot('Cabin')
survived_rate_plot('Embarked')