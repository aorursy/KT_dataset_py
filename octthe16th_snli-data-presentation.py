# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



data_dir = '/kaggle/input/inf8460-snliA19/'



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Reading the training data

train_data = pd.read_csv(os.path.join(data_dir, 'snli_train.csv'))

print(train_data.shape)

train_data.head()
# Splitting by data type

entailement_data = train_data[train_data['label1'] == 'entailment']

print('Entailement relationships')

for i, line in entailement_data.sample(5).iterrows():

    print('"{}"\t "{}"'.format(line['sentence1'], line['sentence2']))



print(entailement_data.shape)

entailement_data.head()
# Splitting by data type

contradiction_data = train_data[train_data['label1'] == 'contradiction']

print('Contradiction relationships')

for i, line in contradiction_data.sample(5).iterrows():

    print('"{}"\t "{}"'.format(line['sentence1'], line['sentence2']))



print(contradiction_data.shape)

contradiction_data.head()
# Splitting by data type

neutral_data = train_data[train_data['label1'] == 'neutral']

print('Neutral relationships')

for i, line in neutral_data.sample(5).iterrows():

    print('"{}"\t "{}"'.format(line['sentence1'], line['sentence2']))



print(neutral_data.shape)

neutral_data.head()