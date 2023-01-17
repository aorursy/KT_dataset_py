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
#Reading the dataset
dataset = pd.read_csv('../input/videogamesales/vgsales.csv')
#Looking at the first 5 elements of the dataset to get an idea of how it looks 
dataset.head()
#To see how many unique values are in each column
dataset.nunique()
#Checking for null values
dataset.isnull().sum()
dataset.corr()
#Visulaizing the correlation matrix
import seaborn as sns
sns.heatmap(dataset.corr(),annot=True)
#importing statistics to find the mode 
import statistics
def missing_values(x):
    #In order for us to fill the missing values in the year column with respect to the Genre column we have to find the number of unique values in the Genre column
    #We achieve this by invoking the .unique() function which returns all the 12 unique values in a list
    #We go through all the values and then compare them with the year and find the most occuring year for each Genre and then fill the missing values with the mode
    for i in dataset['Genre'].unique():
        mode = statistics.mode(dataset[dataset['Genre']==i]['Year'])
        dataset.loc[dataset['Genre']==i,'Year'] = dataset[dataset['Genre']==i]['Year'].fillna(mode)
#Now let us check the number of missing values in the dataset
missing_values(dataset)
dataset.isnull().sum()
#Let us describe the dataset
dataset.describe()
y = dataset.groupby(['Year'])['Global_Sales'].sum()
#Let's see what y looks like
print(y,type(y))
#As we can see the year columns are in float so let;s change that to int and then plot the graph
x = y.index.astype(int)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
sns.barplot(x,y)
#The years are not visible to the eyes. So a better way to visualize this is to rotate the values in the x-axis by 60 degrees.
plt.figure(figsize=(12,8))
c = sns.barplot(x,y)
c.set_xticklabels(labels=x,rotation=60)
x = dataset['Year'].sort_values().astype(int)
y = list(set(x))
plt.figure(figsize=(12,8))
c = sns.countplot(x)
c.set_xticklabels(labels=y,rotation=60)