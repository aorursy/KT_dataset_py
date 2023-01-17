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
df1.rename(columns={'Annual CO₂ emissions (tonnes )':'emissions'},inplace=True)
df1.max()
df1
df2=df1.groupby('Entity').mean()
df3 = df2.sort_values(by='emissions')
df3.min()
#3. Name top 10 countries with maximum average Co2 emission? plot it .
df2=df1.groupby('Entity').mean()
df3 = df2.sort_values(by='emissions')
df3=df2.sort_values(by=['emissions'],ascending=False, na_position='first')
df4=df3.head(10)
df4
df4.T.plot(kind='bar')
#4. Name top 10 countries with minimum average Co2 emission? plot it 
mi_n=df2.sort_values(by=['emissions'], na_position='first')
da=mi_n.head(10)
da
da.T.plot(kind='bar')
#5. Name the 10 countries which produced minimum average CO2 after year 2000. plot it
df5=df[df['Year']>2000]
df61=df5[['Entity','Annual CO₂ emissions (tonnes )']]
df62=df61.groupby('Entity').mean()
df63 = df62.sort_values(by='Annual CO₂ emissions (tonnes )',ascending=True)
df64=df63.head(10)
print(df64)
#Plotting
from matplotlib import pyplot as plt
plt.plot(df64,'c')
#6.Name the 10 countries which produced maximum average CO2 after year 2000. plot it 
df63 = df62.sort_values(by='Annual CO₂ emissions (tonnes )',ascending=False)
df64=df63.head(10)
print(df64)
#Plotting
plt.plot(df64,'y')
#7. Plot yearwise Co2 production of the world between 2012-2019.
df7=df[df['Year']>=2012]
df71=df7[['Entity','Year','Annual CO₂ emissions (tonnes )']]
df72=df71[df71['Entity']=='World']
df73=df72.groupby('Year').sum()
plt.plot(df73,'c')
#8. compare co2 production of top 5 countries(by max co2 emission) over the years by line plot.
df3=df2.sort_values(by=['emissions'],ascending=False, na_position='first')
df5=df3.head(5)
df5.T.plot(kind='bar')
