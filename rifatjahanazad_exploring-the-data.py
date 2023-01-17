# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from pprint import pprint





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
def displayData(dataFrame):

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):

        pprint(dataFrame.describe())

        pprint(dataFrame.info())

        pprint(dataFrame.head())
displayData(train)
displayData(test)
%matplotlib notebook

total_vars = 200

for i in range(total_vars):

    figure_num = 0

    unique_val_train = train.sort_values(by="var_"+ str(i))['var_' + str(i)].unique()

    unique_val_test = test.sort_values(by="var_"+ str(i))['var_' + str(i)].unique()

    fig = plt.figure(i,figsize=(20,6))

    figure_num += 1

    left_fig = fig.add_subplot(1, 2, figure_num)

    sns_plot = sns.distplot(train["var_"+ str(i)],color='red',axlabel="var_"+ str(i) + " with all data")

    sns_plot = sns.distplot(test["var_"+ str(i)],color='blue',axlabel="var_"+ str(i) + " with all data")

    sns_plot = None

    

    figure_num += 1

    right_fig = fig.add_subplot(1, 2, figure_num)

    right_fig.table(cellText=np.array([unique_val_train[:15]]).transpose(),bbox=[.9,0,.1,1],colLabels=['train_unq'])

    right_fig.table(cellText=np.array([unique_val_test[:15]]).transpose(),bbox=[1,0,.1,1],colLabels=['test_unq'])

    sns_plot = sns.distplot(unique_val_train, color='red',axlabel="var_"+ str(i) + " with unique data only")

    sns_plot = sns.distplot(unique_val_test, color='blue',axlabel="var_"+ str(i) + " with unique data only")

    fig.savefig("output-" + str(i) + ".png")

    sns_plot = None

    