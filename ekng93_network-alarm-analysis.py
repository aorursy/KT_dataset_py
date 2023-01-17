# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Libraries

from matplotlib import pyplot as plt

import pandas as pd

import datetime 

import calendar

from datetime import datetime



date = datetime.date(datetime.now())

print (date)
data_url = '../input/alarms/ALARM.csv'

df = pd.read_csv(data_url, index_col=0)   #index_col=0 means first column will become index, otherwise specify with column name 'example name'



print (df)

print (df.dtypes)

print (df.shape)

print (df.columns)

b=len(df)

print (b)

#Select out desired features to analyse and put into new dataframe

df1= df.loc[:,['Severity','Alarm Code Name']]

df1['Alarm Code Name 2'] = df1 ['Alarm Code Name']

df1['Severity 2'] = df1 ['Severity']

    

#severity analysis and sorting

df1.set_index('Severity 2',inplace=True)

df1.sort_values("Severity 2", axis = 0, ascending = True, 

                 inplace = True, na_position ='last')

print (df1)

temp=[]

freq=[]

temp.clear()

freq.clear()

count=0

for i in range (0,b-1):

    if df1.iloc[i,0] != df1.iloc[i+1,0]:

        temp.append(df1.iloc[i,0])

        count=count+1

        freq.append(count)

        count=0



    elif df1.iloc[i,0] == df1.iloc[i+1,0] and i<(b-2):

        count=count+1

    

    elif df1.iloc[i,0] == df1.iloc[i+1,0] and i==(b-2):

        temp.append(df1.iloc[i,0])

        count=count+2

        freq.append(count)

        count=0



df2 = pd.DataFrame({'Severity':temp,'Count':freq})

df2.sort_values("Count", axis = 0, ascending = False,inplace = True, na_position ='last')

print (df2)

# plot pie chart

labels=df2['Severity']

sizes=df2['Count']



plt.clf()

plt.title(date)

plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=False, startangle=140)
df1.set_index('Alarm Code Name 2',inplace=True)

df1.sort_values("Alarm Code Name 2", axis = 0, ascending = True, inplace = True, na_position ='last')
temp.clear()

freq.clear()

count=0

for i in range (0,b-1):

    if df1.iloc[i,1] != df1.iloc[i+1,1]:

        temp.append(df1.iloc[i,1])

        count=count+1

        freq.append(count)

        count=0



    elif df1.iloc[i,1] == df1.iloc[i+1,1] and i<(b-2):

        count=count+1



    elif df1.iloc[i,1] == df1.iloc[i+1,1] and i==(b-2):

        temp.append(df1.iloc[i,0])

        count=count+2

        freq.append(count)

        count=0



df3 = pd.DataFrame({'Alarm name':temp,'Count':freq})

df3.sort_values("Count", axis = 0, ascending = False,inplace = True, na_position ='last')

print (df3)



        
print (df3.iloc[0:20,:])
