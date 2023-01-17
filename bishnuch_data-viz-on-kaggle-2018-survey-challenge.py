# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style ='whitegrid')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/multipleChoiceResponses.csv')
udf = df
df.head()
df.info()
df.shape
# Columns of interest
index_list = [0, 1, 3, 4, 5, 6, 7, 9, 11, 12]
index_names = ['time-taken', 'sex', 'age-bar', 'country', 'degree', 'major', 'title', 'industry', 'work-exp', 'compensation']
# Function to categorize & drop columns
def cat_and_drop(df, index_list, index_names):
    
    # Drop if not male or female
    df = df.loc[(df.iloc[:,1] == 'Male') | (df.iloc[:,1] == 'Female')]
    
    # Convert
    df[index_names] = df.iloc[:,index_list].astype('category')
    
    # Drop First Row 
    df.drop(df.index[:1], inplace=True)
    
    # Return columns of interest only
    df = df.loc[:, df.columns.isin(index_names)]
    
    return df

df = cat_and_drop(df, index_list, index_names)

df.describe()
sns.countplot(x="age-bar", hue='sex', data=df)
plt.show()
sns.countplot(x='age-bar', hue='title', data=df)
plt.show()
# which country has more aged data science enthutiasts
sns.countplot(x="age-bar", hue='country', data=df)
plt.show()
#number of participants from each country:
df['country'].value_counts()

# we can clearly see USA has the more number of users then India then china who have filled up the survey in first second and 3rd place.

#last place is secured by austria
df.iloc[np.where(df.country.values=='United States of America')]
df['sex'].value_counts()
#shows the ratio of male and female users who took part in the survey from us.
sns.countplot(x="sex", hue='degree', data=df)
plt.show()
#showing the number of degree holders in male and female in us who filled this survey

#we can see maximum users were masters degree in both male and female.
sns.countplot(x="sex", hue='title', data=df)
plt.show()
#we can see from the data that maximum of teh survey users were students in male and female who filled the survey.
sns.countplot(x="age-bar", hue='title', data=df)
plt.show()

#interesting to see maximum were students in us who filled this survey are in age group 18-21