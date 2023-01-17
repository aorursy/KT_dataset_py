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





df_train = pd.read_csv ('../input/train.csv');

df_test = pd.read_csv ('../input/test.csv');



print (df_train.columns);

print (df_test.columns);
import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings



sns.displot(df_train['bone_length']);
print (df_test.describe());



print (df_train['type']);
var = 'bone_length'

data = pd.concat([df_train['type'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x="type", y=var, data=data)

fig.axis(ymin=0, ymax=1);
var = 'rotting_flesh'

data = pd.concat([df_train['type'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x="type", y=var, data=data)

fig.axis(ymin=0, ymax=1);
var = 'hair_length'

data = pd.concat([df_train['type'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x="type", y=var, data=data)

fig.axis(ymin=0, ymax=1);
var = 'has_soul'

data = pd.concat([df_train['type'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x="type", y=var, data=data)

fig.axis(ymin=0, ymax=1);