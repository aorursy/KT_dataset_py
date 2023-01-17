# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



data_dir = '/kaggle/input/inf8460-sts-bA19/'



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Reading the training data

train_data = pd.read_csv(os.path.join(data_dir, 'sts-b_train.csv'))

print(train_data.shape)

train_data.head()
# Splitting by data type

relationship_data = train_data[train_data['score'] == 5.0]

print('Score 5')

for i, line in relationship_data.sample(5).iterrows():

    print('"{}"\t "{}"'.format(line['sentence1'], line['sentence2']))



print(relationship_data.shape)

relationship_data.head()
# Splitting by data type

relationship_data = train_data[(train_data['score'] > 2) & (train_data['score'] < 3) ]

print('Score between 2 and 3')

for i, line in relationship_data.sample(5).iterrows():

    print('"{}"\t "{}"'.format(line['sentence1'], line['sentence2']))



print(relationship_data.shape)

relationship_data.head()
# Splitting by data type

relationship_data = train_data[train_data['score'] == 0]

print('Score of 1')

for i, line in relationship_data.sample(5).iterrows():

    print('"{}"\t "{}"'.format(line['sentence1'], line['sentence2']))



print(relationship_data.shape)

relationship_data.head()