# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
data.head(10)

data.hist(figsize = (13, 9))
data.columns
y = data.diagnosis
columns_delete = ['id', 'Unnamed: 32', 'diagnosis']
x = data.drop(columns_delete ,axis = 1)
x
y.hist()
ax = sns.countplot(y, label = 'count')
B, M = y.value_counts()
print('Number of malignant patients: ', M)
print('Number of Benign patients: ', B)
x.describe()
data = x
data_std = (data - data.mean()) / data.std()
data = pd.concat([y, data_std.iloc[:, 0:10]], axis = 1)
data = pd.melt(data, id_vars = 'diagnosis',
              var_name = 'features',
              value_name = 'value')
plt.figure(figsize=(20, 20))
sns.violinplot(x = 'features', y = 'value', hue = 'diagnosis', data = data,
              split = True, inner = 'quart')
plt.xticks(rotation = 45);
data = pd.concat([y, data_std.iloc[:, 10:20]], axis = 1)
data = pd.melt(data, id_vars = 'diagnosis',
              var_name = 'features',
              value_name = 'value')
plt.figure(figsize=(10, 20))
sns.violinplot(x = 'features', y = 'value', hue = 'diagnosis', data = data,
              split = True, inner = 'quart')
plt.xticks(rotation = 45);
data = pd.concat([y, data_std.iloc[:, 20:30]], axis = 1)
data = pd.melt(data, id_vars = 'diagnosis',
              var_name = 'features',
              value_name = 'value')
plt.figure(figsize=(20, 20))
sns.violinplot(x = 'features', y = 'value', hue = 'diagnosis', data = data,
              split = True, inner = 'quart')
plt.xticks(rotation = 45);
sns.boxplot(x = 'features', y = 'value', data = data, hue = 'diagnosis')
plt.xticks(rotation = 45)
sns.jointplot(x.loc[:, 'concavity_worst'],
             x.loc[:, 'concave points_worst'],
             kind = 'regg',
             color = '#801515')
sns.set(style = 'whitegrid', palette = 'muted')
data = x
data_std = (data - data.mean()) / data.std()
data = pd.concat([y, data_std.iloc[:, 0:10]], axis = 1)
data = pd.melt(data, id_vars = 'diagnosis',
              var_name = 'features',
              value_name = 'value')
plt.figure(figsize=(10, 10))
sns.swarmplot(x = 'features', y = 'value', hue = 'diagnosis', data = data)
plt.xticks(rotation = 45);
sns.set(style = 'whitegrid', palette = 'muted')
data = x
data_std = (data - data.mean()) / data.std()
data = pd.concat([y, data_std.iloc[:, 10:20]], axis = 1)
data = pd.melt(data, id_vars = 'diagnosis',
              var_name = 'features',
              value_name = 'value')
plt.figure(figsize=(10, 10))
sns.swarmplot(x = 'features', y = 'value', hue = 'diagnosis', data = data)
plt.xticks(rotation = 45);
f, ax = plt.subplots(figsize = (20, 20))
sns.heatmap(x.corr(), annot = True, linewidth = 0.9, fmt = '.1f', ax= ax)
data.index = data['diagnosis']
data.drop(['diagnosis'], axis = 1)
