# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/heart-disease-uci/heart.csv')
data.info()
#Corelation Matrix

ax = plt.subplots(figsize=(18,18))

ax = sns.heatmap(data.corr(), vmin=-1, vmax=1, center=0, annot=True, linewidths=1, cmap=plt.get_cmap("PiYG", 10))
data.head(10)
data.columns
#Matplotlib

data.age.plot(kind='line', color='g', label='Age', linewidth=1, grid=True, linestyle=':')

data.thalach.plot(color='r', label='Thalach', linewidth=1, grid=True, linestyle='-.')

plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()
#Scatter Plot

#thalach --> The person's maximum heart rate achieved

#chol --> The person's cholesterol measurement in mg/dl

data.plot(kind='scatter', x='thalach', y='chol', alpha=0.5, color='red')

plt.xlabel('trestbps')

plt.ylabel('fbs')

plt.title('The persons maximum heart rate & cholesterol measurement')
#Histogram

data.chol.plot(kind='hist', bins=50, figsize=(18,9))

plt.title('The person\'s cholesterol measurement in mg/dl')

plt.show()

data.shape
data.tail(10)
data.info()
data.describe()
data.boxplot(column='chol', by='target')

plt.show()
data_new = data.head()

melted = pd.melt(frame=data_new, id_vars='age', value_vars=['trestbps', 'oldpeak'])

melted
melted.pivot(index='age', columns='variable', values='value')
data1 = data.head()

data2 = data.tail()

concat_data = pd.concat([data1, data2], axis=0, ignore_index=True)

concat_data
data.info()
data1 = data.loc[:,['chol', 'trestbps']]

data1.plot()