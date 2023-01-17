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
# Reading the dataset using pandas

# taking the year as index col 

AMZ_df = pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv', encoding='latin1', index_col='year')
# Lets have a look 

AMZ_df.head()
# States information 

print("States: ",AMZ_df.state.unique())

print("-"*50)

print("Count of States:", AMZ_df.state.nunique())


AMZ_df.describe()
#translating the month in english

AMZ_df['month'].replace(to_replace = 'Janeiro', value = 'Jan', inplace = True)

AMZ_df['month'].replace(to_replace = 'Fevereiro', value = 'Feb', inplace = True)

AMZ_df['month'].replace(to_replace = 'Março', value = 'Mar', inplace = True)

AMZ_df['month'].replace(to_replace = 'Abril', value = 'Apr', inplace = True)

AMZ_df['month'].replace(to_replace = 'Maio', value = 'May', inplace = True)

AMZ_df['month'].replace(to_replace = 'Junho', value = 'Jun', inplace = True)

AMZ_df['month'].replace(to_replace = 'Julho', value = 'Jul', inplace = True)

AMZ_df['month'].replace(to_replace = 'Agosto', value = 'Aug', inplace = True)

AMZ_df['month'].replace(to_replace = 'Setembro', value = 'Sep', inplace = True)

AMZ_df['month'].replace(to_replace = 'Outubro', value = 'Oct', inplace = True)

AMZ_df['month'].replace(to_replace = 'Novembro', value = 'Nov', inplace = True)

AMZ_df['month'].replace(to_replace = 'Dezembro', value = 'Dec', inplace = True)
AMZ_df.plot.line(y='number', x='date',figsize=(18,7));
AMZ_df.groupby('year')['number'].sum().plot.line();
AMZ_df['number'].plot.hist();
AMZ_df.groupby('year')['number'].sum().plot.bar(figsize=(20,10),title='Number of Fire per year').set(ylabel='Number', xlabel='Year');
df_2k3 = AMZ_df.filter(like='2003', axis=0)
df_2k3.groupby('month')['number'].sum().plot.bar(figsize=(20,10),title='Number of Fire in 2003').set(ylabel='Number', xlabel='Year');
AMZ_df.groupby('month')['number'].sum().plot.bar(figsize=(20,10),title='Number of Fire per month',grid=True).set(ylabel='Number', xlabel='month');