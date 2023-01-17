# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/comptab_2018-01-29 16_00_comma_separated.csv')
import matplotlib.pyplot as plt
import seaborn as sns
dataset.head(5)
dataset.nunique()
# some conclusion of the data
print('The number of sample included:', dataset.shape[0])
print('The number of the taxons is recorded:', dataset['Taxon'].nunique())
print('The most number of the classes is:', dataset['Class'].value_counts().index[0])
print('The most number of the impoter is the Country:', dataset['Importer'].value_counts().index[0])
print('The most number of the expoter is the Country:', dataset['Exporter'].value_counts().index[0])
#The number of different classes is traded
plt.subplots(figsize=(22,30))
plt.title('Classes of the trading wild animmals')
sns.countplot(y='Class', order=dataset['Class'].value_counts().index,data=dataset)
# The top five number of the taxon is trading now
taxon = dataset['Taxon'].value_counts()[:5].to_frame()
sns.barplot(taxon['Taxon'], taxon.index, palette='inferno')
plt.title('Top 5 taxon by num of trading')
plt.xlabel('')
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.show()
#Top Five Importer Countries
plt.subplots(figsize=(10,8))
plt.title('Top 5 countries as the importer')
sns.countplot(y = 'Importer', order = dataset['Importer'].value_counts().index[0:5], data = dataset)
#Top Five Importer Countries
plt.subplots(figsize=(10,8))
plt.title('Top 5 countries as the exporter')
sns.countplot(y = 'Exporter', order = dataset['Exporter'].value_counts().index[0:5], data = dataset)
