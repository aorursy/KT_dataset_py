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
schema = pd.read_csv('../input/survey_results_schema.csv')
print(schema)

df = pd.read_csv('../input/survey_results_public.csv',low_memory=False)
df.head()
df.shape
df.isnull().sum().sort_values(ascending=False)[:10]
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
temp = df.Country.value_counts()[:10]
plt.figure(figsize=(15,5))
sns.barplot(temp.index,temp.values)
plt.show()
df.UndergradMajor.value_counts()
temp = df.DevType.value_counts()[:10]
plt.figure(figsize=(15,5))
sns.barplot(temp.values,temp.index)
# plt.xticks(rotation=45)
plt.show()
df.Gender.value_counts()
temp = df.Age.value_counts()[:10]
temp
plt.figure(figsize=(15,5))
sns.barplot(temp.index,temp.values)

plt.figure(figsize=(15,5))
sns.boxplot(df.Age.index,df.Age.values)

plt.figure(figsize=(15,5))
plt.hist(df.Age)
temp = df.OpenSource.value_counts()
sns.barplot(x = temp.index, y = temp.values)
plt.show()
temp = df.OperatingSystem.value_counts()
plt.figure(figsize=(20,5))
sns.barplot(x = temp.index, y = temp.values)
plt.show()
temp = df.LanguageWorkedWith.value_counts()[:10]
plt.figure(figsize=(20,5))
sns.barplot(y = temp.index, x = temp.values)
plt.show()
temp = df.LanguageDesireNextYear.value_counts()[:10]
plt.figure(figsize=(20,5))
sns.barplot(x = temp.index, y = temp.values)
plt.show()
temp = df.DatabaseWorkedWith.value_counts()[:10]
plt.figure(figsize=(20,5))
sns.barplot(x = temp.index, y = temp.values)
plt.show()

temp = df.DatabaseDesireNextYear.value_counts()[:10]
plt.figure(figsize=(20,5))
sns.barplot(y = temp.index, x = temp.values)
plt.show()

temp = df.PlatformWorkedWith.value_counts()[:10]
plt.figure(figsize=(20,5))
sns.barplot(x = temp.index, y = temp.values)
plt.show()
temp = df.PlatformDesireNextYear.value_counts()[:10]
plt.figure(figsize=(20,5))
sns.barplot(x = temp.index, y = temp.values)
plt.show()
df.CommunicationTools.value_counts()[:10]