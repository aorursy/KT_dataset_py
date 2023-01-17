# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')

df_train.head(10)

data = [df_train['target'].sum(), len(df_train[df_train['target'] == 0])]

series = pd.Series([df_train['target'].sum(), len(df_train[df_train['target'] == 0])], index=['1', '0'], name='train')

print(data)

series.plot.pie(figsize=(7, 7), autopct='%.2f', fontsize=16)
features = df_train.drop(['id'], axis=1)

ones = features[features['target'] == 1]

zeros = features[features['target'] == 0]

differ_mean = ones.mean() - zeros.mean()



differ_mean.plot(kind='bar', figsize=(14,10))