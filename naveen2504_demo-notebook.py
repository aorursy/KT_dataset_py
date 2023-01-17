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
df_train = pd.read_csv('/kaggle/input/mdss-datathon/train_data.csv')

print(df_train.shape)

df_train.head()
# Looking at the distribution of the labels

df_train['label'].value_counts()
df_test = pd.read_csv('/kaggle/input/mdss-datathon/test_data.csv')

print(df_test.shape)

df_test.head()
import random

foo = [0, 1, -1]

prediction = []

for i in range(3000):

    prediction.append(random.choice(foo))
df_test['label'] = prediction
df_test.drop('text', axis=1, inplace=True)
df_test.head()
df_test.to_csv('dummy_submission.csv', index=False)