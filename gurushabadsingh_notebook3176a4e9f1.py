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
df=pd.read_csv("/kaggle/input/co2-ghg-emissionsdata/co2_emission.csv")

df
#1. Entity and Co2 emission



df1=df[['Entity','Annual CO₂ emissions (tonnes )']]

df1
df2 = df1.groupby('Entity').mean()

df2
#2. Which is the  country with minimum average carbon Emission ?



df3=df2.sort_values(by=['Annual CO₂ emissions (tonnes )'],ascending=False, na_position='first')

df4=df3.tail(2)

df4
#3. Name top 10 countries with maximum average Co2 emission? plot it .





df5=df3.tail(11)

df5.T.plot(kind='bar')

df5
#4. Name top 10 countries with minimum average Co2 emission? plot it .



df6=df3.head(11)

df6

df6.T.plot(kind='bar')

#8. compare co2 production of top 5 countries(by max co2 emission) over the years by line plot.



top5=df3.head(6)

top5

import seaborn as sns

sns.lineplot(data=top5)
df1=df[['Entity','Annual CO₂ emissions (tonnes )']]

df1
latestdf=df[df.Year > 1999]

latestdf
updateddf=latestdf[['Entity','Year','Annual CO₂ emissions (tonnes )']]

updateddf
updateddf2=updateddf.groupby('Entity').mean()

updateddf2
#Name the 10 countries which produced minimum average CO2 after year 2000. plot it



updateddf3=updateddf2.sort_values(by='Annual CO₂ emissions (tonnes )')

after2000=updateddf3.head(11)

after2000
import seaborn as sns

sns.lineplot(data=after2000)
# Name the 10 countries which produced maximum average CO2 after year 2000. plot it



updateddf4=updateddf2.sort_values(by=['Annual CO₂ emissions (tonnes )'],ascending=False, na_position='first')

max2000=updateddf4.head(11)

max2000
import seaborn as sns

sns.lineplot(data=max2000)

years7data = df[df["Entity"]=='World']

years7data
#Plot yearwise Co2 production of the world between 2012-2019.



years7data1=years7data[years7data.Year > 2011]

years7data1
years7data1.rename(columns ={"Annual CO₂ emissions (tonnes )":"emission"},inplace =True)

years7data1

import matplotlib.pyplot as plt

plt.plot(years7data1.Year,years7data1.emission)

plt.xlabel('Year')

plt.ylabel('Emission')

plt.show()