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
#Importing the dataset
data = pd.read_csv("/kaggle/input/gloabl-startup-investment/Startup_investments.csv", encoding = "ISO-8859-1")
#Reviewing the dataset
data
#Reviewing the statistical parameters on columns
data.describe()
# Column names in the dataset.
data.columns
#Total Number of NULL or NAN values in a column
data.isnull().sum()
data
# Checking the last 5 records
data.tail()
# Eliminating rows which has NULL or NAN value in 'name' column
data = data[pd.notnull(data['name'])]
len(data)
data.country_code.value_counts()
# bar plot based on status (for example: How many records or start up in the dataset with the status = operating)
data.status.value_counts().plot.bar()
plt.figure(figsize=(7,7))
data.status.value_counts().plot.pie()
# Pie plot with % of the status
plt.figure(figsize = (8,8))
data.status.value_counts().plot(kind='pie',shadow=False, explode=(0, 0, 0), startangle=45,autopct='%1.1f%%')
plt.title('Status')
plt.show()
plt.figure(figsize=(16,7))
g = sns.countplot(x ='country_code', data = data, order=data['country_code'].value_counts().iloc[:10].index)
plt.xticks(rotation=30)
plt.show()
df_USA = data[(data['country_code'] =='USA')]
plt.figure(figsize=(16,7))
g = sns.countplot(x ='state_code', data = df_USA, order=data['state_code'].value_counts().iloc[:20].index)
plt.figure(figsize=(16,7))
sns.countplot(x =' market ', data = data, order=data[' market '].value_counts().iloc[:10].index)
plt.xticks(rotation=30)
plt.show()
