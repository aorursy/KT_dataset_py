# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt 

pk = pd.read_csv('../input/PakistanSuicideAttacks Ver 11 (30-November-2017).csv', encoding='latin1')

pk.head()

pk.groupby(["Longitude", "Latitude"]).size()
plt.figure(figsize=(5, 20))

pk.isnull().mean(axis=0).plot.barh()

plt.title("Proportion of NaNs in each column")
pk.groupby(["Date","Time"]).size()
pk.drop(["S#", "Temperature(F)"], 1, inplace = True)

pk.columns = map(lambda x: x.replace(".", "").replace("_", ""), pk.columns)

pk.fillna(value = -1, inplace = True)
print(pk.shape)

print(pk.dtypes)

print(pk.head(10))
df_loc_count = pk.groupby('City')[['Location']].count()

df_loc_count.columns = ['loc_count']

df_bc = df_loc_count.ix[df_loc_count.loc_count>1,:]

df_bc.plot.barh(title='Location of Attack', legend=True, figsize=(6,8))

plt.show()

df_dead = pk.groupby('Blast Day Type')[['Location']].count()

df_dead.columns = ['dead_count']

df_maxdead = df_dead.ix[df_dead.dead_count>2,:]

df_maxdead.plot.pie(y='dead_count', autopct='%2f', title='Total people died',legend=False ,figsize=(6,6)) 

plt.show()
import seaborn as sns 

state_color = ["#FD2E2E", "#E6E6E6", "#17B978", "#CFCBF1", "#4D4545", "#588D9C"]

sns.factorplot('Targeted Sect if any',data=pk,kind='count', size=10,palette=state_color)
plt.style.use('fivethirtyeight')

pk[:6].plot(kind='bar', x='Province', y='Killed Max')



ax = plt.gca()

ax.set_ylabel('No. of People killed')

ax.set_xlabel('Province')

pass