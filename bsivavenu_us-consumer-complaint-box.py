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
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv("../input/consumer_complaints.csv",low_memory=False)
df.head()
df.shape
df.dtypes
df.isnull().sum().sort_values(ascending=False)
p_product_discussions = round(df["product"].value_counts() / len(df["product"]) * 100,2)
p_product_discussions
plt.figure(figsize=(15,5))
sns.countplot(df['product'])
plt.show()
temp = df.company.value_counts()[:10]
temp
plt.figure(figsize=(20,5))
sns.barplot(temp.index,temp.values)
plt.xticks(rotation=60)
plt.show()
temp = df.state.value_counts()[:10]
temp
plt.figure(figsize=(15,5))
sns.barplot(temp.index,temp.values)
temp = df.company_response_to_consumer.value_counts()
temp
plt.figure(figsize=(15,5))
sns.barplot(y = temp.index, x= temp.values)
df.timely_response.value_counts()
sns.countplot(df.timely_response)
df['consumer_disputed?'].value_counts()
sns.countplot(df['consumer_disputed?'])
top5_disputed = df['company'].loc[df['consumer_disputed?'] == 'Yes'].value_counts()[:5]
top5_disputed
plt.figure(figsize=(15,5))
sns.barplot(x = top5_disputed,y = top5_disputed.index)
plt.show()
top5_nodispute = df['company'].loc[df['consumer_disputed?'] == 'No'].value_counts()[:5]
top5_nodispute
plt.figure(figsize=(15,5))
sns.barplot(x = top5_nodispute.values,y = top5_nodispute.index)
plt.show()
df['date_received'] = pd.to_datetime(df['date_received'])
df['year_received'], df['month_received'] = df['date_received'].dt.year, df['date_received'].dt.month
df.head()
df.year_received.value_counts()
sns.countplot(df.year_received)
df.month_received.value_counts()
sns.countplot(df.month_received)