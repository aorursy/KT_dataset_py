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



import pandas as pd

df=pd.read_csv("/kaggle/input/co2-ghg-emissionsdata/co2_emission.csv")

df
#1.	Which is the country  with maximum average carbon Emission ?



df.loc[df['Annual CO₂ emissions (tonnes )'].idxmax()]
#Step1- create a subset dataframe with two columns

#1. Entity and Co2 emission



df1=df[['Entity','Annual CO₂ emissions (tonnes )']]

df1
df2=df1.groupby('Entity').mean()

df2
#2.	Which is the  country with minimum average carbon Emission ?



df3=df2.loc[df2['Annual CO₂ emissions (tonnes )'].idxmin()]

df3
#3.Name top 10 countries with maximum average Co2 emission? plot it 



df4=df2.sort_values(by=['Annual CO₂ emissions (tonnes )'],ascending=False, na_position='first')

b=df4.head(10)

b
from matplotlib import pyplot as plt

b.plot.bar()
#4.	Name top 10 countries with minimum average Co2 emission? plot it .



df5=df2.sort_values(by=['Annual CO₂ emissions (tonnes )'],ascending=True, na_position='first')

a=df5.head(10)

a
from matplotlib import pyplot as plt

a.plot.bar()
df6=df.drop(['Code'],axis=1)

df6
#8. compare co2 production of top 5 countries(by max co2 emission) over the years by line plot.



df3=df2.sort_values(by=['Annual CO₂ emissions (tonnes )'],ascending=False, na_position='first')

df5=df3.head(5)

df5.T.plot(kind='bar')