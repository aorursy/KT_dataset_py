# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

data_set= pd.read_csv('/kaggle/input/table.csv')

pd.set_option("display.max.rows", None)

data_set.head(n=550)
d=data_set['Diagnosed date'].tolist()

print(type(d))

#data_set.replace(to_replace=(data_set['Diagnosed date'].to_string()),value=(str(data_set['Diagnosed date']))[0:2])

#data_set['Diagnosed date'] = data_set['Diagnosed date'].apply(lambda x: int(str(x)[3:6]))

#print(type(data_set['Diagnosed date']))

#data_set['no.of_person']=1

#data_set_new=data_set[['Diagnosed date','no.of_person']]

data = {'Diagnosed date': d , 'people':1} 

df = pd.DataFrame(data)

df.head()

df_t=df.T

pd.set_option("display.max.columns", None)

#new_header = df_t.iloc[0] #grab the first row for the header

#df_t = df_t[1:] #take the data less the header row

#df_t.columns = new_header #set the header row as the df header

df_t.columns = df_t.iloc[0]

df_t = df_t[1:]

new_df=df_t.groupby(level=0,axis=1,sort=False).sum()

new_df.head()



#data_set_new.groupby('Diagnosed date').sum()







final_df=new_df.T

final_df.reset_index(level=0, inplace=True)

final_df['Inc cases']=final_df['people'].cumsum()

final_df.head(n=100)
col= final_df.keys()

print(col)

final_df['people'].sum()
plt.plot(list(final_df['Diagnosed date']), list(final_df['people']),marker='o')

plt.xlabel('Diagnosed Date')

plt.ylabel('No. of Confirmed Cases in a single day')

plt.xticks(rotation=90)

plt.show()
plt.plot(list(final_df['Diagnosed date']), list(final_df['Inc cases']),marker='o',color='Red')

plt.xlabel('Diagnosed Date')

plt.ylabel('Growth Rate of cases')

plt.tick_params(axis='x',width=1,rotation=90)

plt.show()