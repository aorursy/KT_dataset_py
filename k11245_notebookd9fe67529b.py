# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import matplotlib.pyplot as plt

%matplotlib inline



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/act_train.csv', parse_dates=['date'])

test = pd.read_csv('../input/act_test.csv', parse_dates=['date'])

ppl = pd.read_csv('../input/people.csv', parse_dates=['date'])



df_train = pd.merge(train, ppl, on='people_id')

df_test = pd.merge(test, ppl, on='people_id')

del train, test, ppl

print("loaded")
print(df_train.columns)
date_x = pd.DataFrame()

date_x['Class probability'] = df_train.groupby('date_x')['outcome'].mean()

date_x['Frequency'] = df_train.groupby('date_x')['outcome'].size()

date_x.plot(secondary_y='Frequency', figsize=(15, 10))
date_x = pd.DataFrame()

df_test["outcome"] = 1

date_x['Class probability'] = df_test.groupby('date_x')['outcome'].mean()

date_x['Frequency'] = df_test.groupby('date_x')['outcome'].size()

date_x.plot(secondary_y='Frequency', figsize=(20, 10))
date_test_train = pd.DataFrame()

date_test_train['Frequency1'] = df_train.groupby('date_x')['outcome'].size()

df_test["outcome"] = 1

date_test_train['Frequency2'] = df_test.groupby('date_x')['outcome'].size()



date_test_train[:250].plot(secondary_y='Frequency2', figsize=(20, 10))

print(len(df_train.values))

print(len(df_test.values))