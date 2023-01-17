# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train_users_2.csv')

#print(df)



plt.subplot2grid((1,1), (0,0))

df.country_destination[df.country_destination != 'NDF'].value_counts().plot(kind='bar')

plt.title('Destination graph')



df2 = df.groupby(['gender', 'country_destination'])['gender'].count().unstack('gender')

df2 = df2.drop('NDF')

df2[['MALE', 'FEMALE']].plot(kind='bar', stacked=False, figsize=(10,5))

plt.title('MALE vs FEMALE')

#plt.title('Destination graph')
df['id'] = df.index

df3 = df.dropna(how='any')

df3 = df3[df3.age <= 200]

df3.plot.scatter('id', 'age')
df4 = df.groupby(['signup_method', 'country_destination'])['signup_method'].count().unstack('signup_method')

df4 = df4.drop('NDF')

print(df4)

df4[['basic', 'facebook', 'google']].plot(kind='bar', figsize=(12,5))
df_sessions = pd.read_csv('../input/sessions.csv')

df_users = pd.read_csv('../input/train_users_2.csv')

#df_join = pd.merge(left = df_sessions, right = df_users, how = 'outer', left_on = 'user_id', right_on='id')



df_sessions.device_type.value_counts().plot(kind='bar')

plt.title('Destination graph')