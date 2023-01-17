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
pd.options.mode.chained_assignment = None
# Multiple Choice Questions
multi = pd.read_csv('../input/multipleChoiceResponses.csv')
# print(multi.shape)
# print(multi.head())

# Columns of interest
index_list = [1, 3, 4, 5, 6, 7, 9, 11, 12]
index_names = ['gender', 'age_bin', 'country', 'degree', 'major', 'title', 'industry', 'years_experience', 'compensation']

# Function to categorize & drop columns
def cat_and_drop(df, index_list, index_names):
    
    # Drop if not male or female
    df = df.loc[(df.iloc[:,1] == 'Male') | (df.iloc[:,1] == 'Female')]
    
    # Convert above indexs & names to categorical
    df[index_names] = df.iloc[:,index_list].astype('category')
    
    # Drop First Row (because prior column names were there)
    df.drop(df.index[:1], inplace=True)
    
    # Return columns of interest only
    df = df.loc[:, df.columns.isin(index_names)]
    
    return df

df = cat_and_drop(multi, index_list, index_names)
df.head()
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x="age_bin", hue='gender', data=df)
plt.show()
sns.countplot(x='age_bin', hue='title', data=df)
plt.show()
from collections import Counter
df['compensation'].describe()
Counter(df['compensation'])
# Pulling out the top salaries from 400k+
top_comp = df[(df['compensation'] == '500,000+') | (df['compensation'] == '400-500,000') | (df['compensation'] == '300-400,000')]
top_comp.describe()
math_stats = top_comp[top_comp['major'] == 'Mathematics or statistics']
math_stats.describe()
cs = top_comp[top_comp['major'] == 'Computer science (software engineering, etc.)']
cs.describe()
