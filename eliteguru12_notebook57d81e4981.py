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
import pandas as pd

import numpy as np

df=pd.read_csv("/kaggle/input/co2-emission/co2_emission - co2_emission.csv")

df
#Entity and Co2 emission

df.rename(columns ={"Annual CO₂ emissions (tonnes )":"emission"},inplace =True)

df
#Entity and Co2 emission

df1=df[['Entity','emission']]

df1
gk=df.groupby('Entity').sum()

gk
#mean of country co2 emission

df2=df.groupby('Entity').mean()

df2
#sorted value from lowest to highest

df3 = df2.sort_values(by='emission')

df3
#Name top 10 countries with maximum average Co2 emission? plot it .

df4=df3.tail(10)

df4
df4=df4.tail(10)

df4

df4.T.plot(kind='bar')
#Name top 10 countries with minimum average Co2 emission? plot it .

df5=df3.head(10)

df5
df5=df3.head(10)

df5

df5.T.plot(kind='bar')
# Name the 10 countries which produced minimum average CO2 after year 2000. plot it

#here head is the miniumum

filtered_data = df2[df2["Year"]>1999]

filtered_data.head(10)

d=filtered_data.head(10)

d

d.T.plot(kind='bar')
#Name the 10 countries which produced maximum average CO2 after year 2000. plot it

#here tail is the maximum

filtered_data = df2[df2["Year"]>1999]

filtered_data.tail(10)

dt=filtered_data.tail(10)

dt

dt.T.plot(kind='bar')
#Plot yearwise Co2 production of the world between 2012-2019.

fdata = df[df["Entity"]=='World']

fdata
fdata1=fdata[fdata.Year > 2011]

fdata1
import matplotlib.pyplot as plt

plt.bar(fdata1.Year , fdata1.emission)

plt.show()
#compare co2 production of top 5 countries(by max co2 emission) over the years by line plot.

df3=df2.sort_values(by=['emission'],ascending=False, na_position='first')

df4=df3.head(10)

df4
import seaborn as sns

sns.lineplot(data=df4)