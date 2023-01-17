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
#Open the Kaggle and add above dataset



df=pd.read_csv("/kaggle/input/co2-ghg-emissionsdata/co2_emission.csv")

df
#Q)create a subset dataframe with two columns



df1=df[['Entity','Annual CO₂ emissions (tonnes )']]

df1
df2=df1.groupby('Entity').mean()

df2
df3 = df2.sort_values(by='Annual CO₂ emissions (tonnes )')

df3
#Which is the  country with minimum average carbon Emission



minEmission=df3.head(2)

minEmission
df3=df2.sort_values(by=['Annual CO₂ emissions (tonnes )'],ascending=False, na_position='first')

df4=df3.head(10)

df4

#Which is the country  with maximum average carbon Emission 



maxEmission=df3.head(2)

maxEmission
#Name top 10 countries with maximum average Co2 emission? plot it .



top10max=df4

top10max
top10max.T.plot(kind='bar')
df3 = df2.sort_values(by='Annual CO₂ emissions (tonnes )')

df3
#Name top 10 countries with minimum average Co2 emission? plot it .



top10MIN=df3.head(10)

top10MIN
import seaborn as sns

sns.lineplot(data=top10MIN)
#compare co2 production of top 5 countries(by max co2 emission) over the years by line plot.



df3=df2.sort_values(by=['Annual CO₂ emissions (tonnes )'],ascending=False, na_position='first')

Top5Max=df3.head(6)

Top5Max

import seaborn as sns

sns.lineplot(data=Top5Max)
df2=df1.groupby('Entity').mean()

df2
# DATA AFTER YEAR 200



newdf=df[df.Year > 1999]

newdf

newdf1=newdf[['Entity','Year','Annual CO₂ emissions (tonnes )']]

newdf1
newdf2=newdf1.groupby('Entity').mean()

newdf2
#Name the 10 countries which produced minimum average CO2 after year 2000. plot it



newdf3=newdf2.sort_values(by='Annual CO₂ emissions (tonnes )')

MIN2000=newdf3.head(11)

MIN2000



                    
import seaborn as sns

sns.lineplot(data=MIN2000)
# Name the 10 countries which produced maximum average CO2 after year 2000. plot it



newdf4=newdf2.sort_values(by=['Annual CO₂ emissions (tonnes )'],ascending=False, na_position='first')

MAX2000=newdf4.head(11)

MAX2000
import seaborn as sns

sns.lineplot(data=MAX2000)
fdata = df[df["Entity"]=='World']

fdata
#Plot yearwise Co2 production of the world between 2012-2019.



fdata1=fdata[fdata.Year > 2011]

fdata1

fdata1.rename(columns ={"Annual CO₂ emissions (tonnes )":"emission"},inplace =True)

fdata1
import matplotlib.pyplot as plt

plt.bar(fdata1.Year , fdata1.emission)

plt.show()