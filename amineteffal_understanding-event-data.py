# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/data-science-bowl-2019/train.csv')

train_labels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')

test = pd.read_csv('../input/data-science-bowl-2019/test.csv')

specs = pd.read_csv('../input/data-science-bowl-2019/specs.csv')

sample_submission = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')
# fist ten lines of train

train.head(10)

# unique values of every clolumn

unique_values = {}

unique_values_count = {}

train_cols = train.columns

for c in train_cols:

    unique_values[c] = train[c].unique()

    unique_values_count[c] = len(train[c].unique())



print(unique_values_count)
unique_values['event_code']
unique_values['title']
# dic where we will store kyes and there number

event_data_keys = {}



# store column in a variable

col_event_data = train['event_data']



# find patterns containing any string except : , { and }  followed by :

for l in col_event_data:

    temp = re.findall("[^,{}]*:", l[1:-1])  # don't take { and }

    for k in temp:

        if k in event_data_keys:

            event_data_keys[k] += 1

        else:

            event_data_keys[k] = 0

            

# Store the dic in a dataframe for futur uses

event_data_keys_df = pd.DataFrame()

event_data_keys_df['key'] = list(event_data_keys.keys())

event_data_keys_df['number'] = list(event_data_keys.values())



# sort by descending order of number column

event_data_keys_df = event_data_keys_df.sort_values(

    'number', ascending=True)



# display first lines

event_data_keys_df.head(5)
# display last lines

event_data_keys_df.tail(5)
# first, sort data

# sort by descending order of number column

event_data_keys_df = event_data_keys_df.sort_values('number', ascending=False)



sns.set(style="whitegrid")

# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(12, 30))

sns.set_color_codes("pastel")

sns.barplot(x="number", y="key", data=event_data_keys_df,

            label="Number of occuence of keys in event_data column", color="b")



ax.set(xlim=(0, 12000000), ylabel="Keys in event_data column",

       xlabel="Number of occuence of keys in event_data column")