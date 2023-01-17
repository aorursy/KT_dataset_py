# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (10,6) # define figure size of pyplot
pd.set_option("display.max_columns", 100) # set max columns when displaying pandas DataFrame

pd.set_option("display.max_rows", 200) # set max rows when displaying pandas DataFrame
df = pd.read_csv('/kaggle/input/uisummerschool/Online_sales.csv')
df
Question: "How many ..."
df.describe() # statistical description of DataFrame columns, numerical only
df.info() # get DataFrame general info
df.head() # get first 5 records of DataFrame
df = df.iloc[:, 2:] # get columns from Pclass to Embarked
df.head()
cols = [x.lower() for x in df.columns]

df.columns = cols
df.head()
df[df['age'].isnull()].tail() # get last 5 records of DataFrame
df[df['cabin'].isnull()].head()
null_age = df[df['age'].isnull()].copy()
null_age.groupby('sex')['name'].count().reset_index(name='total_passengers')
grouped_null_age = null_age.groupby('sex')['name'].count().reset_index(name='total_passengers')
grouped_null_age.plot(kind='bar', x='sex');
df.plot(kind='box');
df['fare'].plot(kind='box');
df.hist();
df['age'].hist();
male_filler_age = df[df['sex']=='male']['age'].median()

female_filler_age = df[df['sex']=='female']['age'].median()
# using median of each gender to fill null values

df['age'] = np.where(df['age'].isnull(), np.where(df['sex'] == 'male', male_filler_age, female_filler_age), df['age'])
df[df['age'].isnull()] # there's no null values anymore
group_embarked = df.groupby('embarked')['name'].count().reset_index(name='total_passengers')
# see the distribution of passengers by place they embarked

group_embarked.plot(kind='bar', x='embarked');
grouped_age = df.groupby('age')['name'].count().reset_index(name='total_passengers')
# see the distribution of passengers by age (similar to histogram)

grouped_age.plot(kind='line', x='age');