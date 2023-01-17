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
df=pd.read_csv('/kaggle/input/co2-ghg-emissionsdata/co2_emission.csv')
df
df1=df[['Entity','Annual CO₂ emissions (tonnes )']]
df1
df2=df1.groupby('Entity').mean()
df2


df3=df2.sort_values(by=['Annual CO₂ emissions (tonnes )'],ascending=False, na_position='first')

df4=df3.head(10)
df4


from matplotlib import pyplot as plt
plt.plot(df4,'g')
df4.T.plot(kind='bar')
df5=df2.sort_values(by=['Annual CO₂ emissions (tonnes )'])
df6=df5.head(10)
df6
df6.T.plot(kind='bar')
#ques5
#Name the 10 countries which produced minimum average CO2 after year 2000. plot it
df7=df[df.Year>2000]
df8=df7[['Entity','Annual CO₂ emissions (tonnes )']]
df9=df8.groupby('Entity').mean()
df10=df9.sort_values(by=['Annual CO₂ emissions (tonnes )'])
df11=df10.head(10)
print(df11)
from matplotlib import pyplot as plt
plt.plot(df11)


#ques6
#Name the 10 countries which produced maximum average CO2 after year 2000. plot it
df12=df[df.Year>2000]
df13=df12[['Entity','Annual CO₂ emissions (tonnes )']]
df14=df13.groupby('Entity').mean()
df15=df4.sort_values(by=['Annual CO₂ emissions (tonnes )'],ascending=False, na_position='first')
df16=df15.head(10)
print(df16)
from matplotlib import pyplot as plt
plt.plot(df16)

#ques7
#Plot yearwise Co2 production of the world between 2012-2019.
df7=df[df['Year']>=2012]
df71=df7[['Entity','Year','Annual CO₂ emissions (tonnes )']]
df72=df71[df71['Entity']=='World']
df73=df72.groupby('Year').sum()
plt.plot(df73,'c')

#ques8
#compare co2 production of top 5 countries(by max co2 emission) over the years by line plot.
df8=df[['Entity','Year','Annual CO₂ emissions (tonnes )']]
df81=df8[df8['Entity']=='World']
df811=df81.groupby('Year').sum()
plt.plot(df811,'c',label='World')
df82=df8[df8['Entity']=='Russia']
df821=df82.groupby('Year').sum()
plt.plot(df821,'b',label='Russia')
df83=df8[df8['Entity']=='United States']
df831=df83.groupby('Year').sum()
plt.plot(df831,'g',label='United States')
df84=df8[df8['Entity']=='EU-28']
df841=df84.groupby('Year').sum()
plt.plot(df841,'r',label='EU-28')
df85=df8[df8['Entity']=='China']
df851=df85.groupby('Year').sum()
plt.plot(df851,'y',label='China')
plt.legend()