# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt

%matplotlib inline

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/covid19-corona-virus-india-dataset/complete.csv')

df =df[['Date','Name of State / UT','Cured/Discharged/Migrated','Death','Total Confirmed cases']]

df = df.rename(columns = {'Name of State / UT':"State",'Cured/Discharged/Migrated':"Cured",'Total Confirmed cases':"Total"})

df.head()

df['Active'] = df['Total']- (df['Cured']+df["Death"])

df.head()
df_1 =df.pivot_table("Active",["State"],"Date")

df_1 = df_1.fillna(0)

df_1.head()


df_2 =df_1.iloc[:,30:] #selecting the dates from significant numbers starts reported

df_2 = df_2.transpose()

df_2.head()
df_2.plot(kind="line",figsize=(20,10))

plt.legend(loc='upper left')

plt.show
df_3 = df_2[["Kerala","Maharashtra","Tamil Nadu","Gujarat","Karnataka","Uttar Pradesh"]]

df_3.head()
df_3.plot(kind="line",figsize=(10,10))



plt.title("Kerala vs The Big 5")

plt.xlabel("Date")

plt.ylabel("Active cases")
df_3.iloc[20:32,:].plot(kind='line',figsize=(20,10))
df_4 =df.pivot_table("Cured",["State"],"Date")

df_4 = df_4.fillna(0)

df_4.head()

df_5 = df_4.transpose()

df_5 = df_5[["Kerala","Maharashtra","Tamil Nadu","Gujarat","Karnataka","Uttar Pradesh"]].iloc[30:,:]

df_5.plot(kind='line',figsize=(20,10))