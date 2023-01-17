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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

dataset = pd.read_csv("../input/aapl-stock/aapl.csv", parse_dates=["Date"])
dataset.head()
#plt.figure(figsize=(20,10))
for col in dataset.columns[1:]:
    gr = sns.relplot(x='Date', y=col, kind='line', data=dataset[:30], height=5, aspect=1.75)
    gr.fig.autofmt_xdate()
dataset.corr()
plt.figure(figsize=(16,8))
sns.heatmap(dataset.corr(), cmap='PuBu', annot=True)
plt.title('Correlation among parameters')
sns.clustermap(dataset[dataset.columns[1:-2][:20]])
plt.figure(figsize=(16,8))
sns.scatterplot(x='High', y='Close', data=dataset)
plt.figure(figsize=(16,8))
sns.scatterplot(x='Open', y='Low', data=dataset)
dataset['hl-ratio'] = ((dataset['High']/dataset['Low']) - 1.0) * 1000
dataset['hl-ratio']
dataset['Volume'].describe()
def get_sale_variability(row):
    if(row['hl-ratio'] < 8.0 and row['Volume'] < 30000000):
        return 'low'
    elif(8.0 < row['hl-ratio'] < 16.0 or 30000000 < row['Volume'] < 60000000):
        return 'med'
    else:
        return 'high'
    
dataset['Sale Variability'] = dataset.apply(get_sale_variability, axis='columns')
dataset['Sale Variability']
dataset['Sale Variability'].value_counts()
plt.figure(figsize=(16,8))
sns.scatterplot(x='Open', y='Low', data=dataset, hue='Sale Variability')
mean_low_month = dataset.groupby(dataset["Date"].dt.month)["Low"].mean()
mean_low_month
plt.figure(figsize=(16,8))
mean_low_month.plot(kind='bar')
plt.figure(figsize=(16,8))
dataset[:100]['Volume'].plot(kind='hist')