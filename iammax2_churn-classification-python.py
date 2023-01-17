# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
churn = pd.read_csv('../input/Churn_Modelling.csv')



def filter_by_dtype(dataframe, data_type):

    """filter a dataframe by columns with a certain data_type"""

    col_names = dataframe.dtypes[dataframe.dtypes == data_type].index

    return dataframe[col_names]



churn_numerical = pd.concat([filter_by_dtype(churn, int), filter_by_dtype(churn, float)], axis=1)



### Plot Distibutions ###

fig = plt.figure(figsize = (15,20))

ax = fig.gca()

churn_numerical.hist(ax = ax, bins = 15)

plt.show()
def print_column_counts(dataframe, list_of_columns):

    """print the percentages of all the unique values in a column"""

    df_len = len(dataframe)

    for name in list_of_columns:

        counts = dataframe[name].value_counts() / df_len

        percents = list(counts); values = list(counts.index)

        values_and_percents = list(zip(values, percents))

        [print(name + ' value ' + str(value) + ' represents ' + str(round(percent * 100, 4)) + '% percent of the field \n') for value, percent in values_and_percents]

        

print_column_counts(churn, ['Exited','HasCrCard','Gender', 'Geography','IsActiveMember','NumOfProducts'])
fig = plt.figure(figsize = (18,18)); ax = fig.gca()

one_hot_churn = pd.get_dummies(churn, columns = ['Gender', 'Geography', 'HasCrCard', 'IsActiveMember'])

sns.heatmap(one_hot_churn.corr(), annot = True, vmin= -0.5, vmax = 0.5, ax=ax)
one_hot_churn.columns
np.random.seed(123)



keep_columns = ['CreditScore', 'Age', 'Tenure',

       'Balance', 'NumOfProducts', 'EstimatedSalary',

       'Gender_Female', 'Gender_Male', 'Geography_France', 'Geography_Germany',

       'Geography_Spain', 'HasCrCard_0', 'HasCrCard_1', 'IsActiveMember_0',

       'IsActiveMember_1']



label = ['Exited']



shuffled_churn = one_hot_churn.sample(frac=1).reset_index(drop=True)

churn_train = shuffled_churn[keep_columns]

churn_label = shuffled_churn[label]

split_num = int(len(shuffled_churn) * 0.8)



train_x = churn_train.iloc[:split_num,:]

train_y = churn_label[:split_num]

test_x = churn_train.iloc[split_num:,:]

test_y = churn_label[split_num:]
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit(train_x, train_y)

pred_y = neigh.predict(test_x)

print(neigh.score(train_x, train_y))

conf = confusion_matrix(test_y, pred_y)

conf
sns.heatmap(conf, annot = True, fmt='g')