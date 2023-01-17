# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import seaborn as sns
from pandas import Series

import matplotlib.pyplot as plt
%matplotlib inline
stov_df = pd.read_csv('/kaggle/input/stack-overflow-2018-developer-survey/survey_results_public.csv')
schema = pd.read_csv('/kaggle/input/stack-overflow-2018-developer-survey/survey_results_schema.csv')
stov_df.info()
## Check for Shape
stov_df.shape
## Check for null columns
## There isn't any difference between isna() vs isnull() call. There were some differences in R, but 
## for python this is same
total_null_enteries = pd.DataFrame(stov_df.isna().sum(), columns=[ 'Total NaEntries'])
total_null_enteries['Percentage'] = (stov_df.isna().sum()/stov_df.isna().count())*100
total_null_enteries = total_null_enteries.sort_values(by='Total NaEntries', ascending=False)
total_null_enteries
stov_df.head()
schema
## Percentage of people who don't disclose there salary
100*stov_df.Salary.isna().sum()/stov_df.Salary.isna().count()
## Total Coders who code for hobby
sns.countplot(stov_df.Hobby)
## Percentage of people is close to 80%
(100*stov_df.Hobby.value_counts()/stov_df.Hobby.size).plot(kind='bar', title='Users doing coding as Hobby')
## How many developers contribute to open source
((stov_df.OpenSource.value_counts()/stov_df.OpenSource.count())*100).plot(kind='pie', figsize=(6,6), 
                                                                          autopct='%.2f',
                                                                          title='Percentage of users contributing to open source')
stov_df_not_null = stov_df[stov_df['Country'].isna() == False]
print(stov_df_not_null['Country'].value_counts().sort_values(ascending=False).head(7))
stov_df_not_null['Country'].value_counts().sort_values(ascending=False).head(15).plot(kind='bar', 
                                                                                      title='Countries from which highest number of people participated',
                                                                                      figsize=(10,6))
stov_df.Student.dropna().value_counts().plot(kind='pie', autopct="%0.2f", figsize=(10,10))
temp = stov_df.Employment.dropna().value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values})
df = df.set_index('labels')
df.plot(kind='pie',autopct="%0.2f", figsize=(10,10), subplots=True)
plt.figure(figsize=(10,10))
chart = sns.countplot(stov_df.FormalEducation.dropna())
chart.set_xticklabels(chart.get_xticklabels(), rotation=45,horizontalalignment='right',
    fontweight='light',
    fontsize='x-large')
stov_df.UndergradMajor.dropna().value_counts().plot(kind='pie', autopct='%0.2f', figsize=(10,10))
stov_df.CompanySize.dropna().value_counts().plot(kind='bar')
