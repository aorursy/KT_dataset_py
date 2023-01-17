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
df1=df[['Entity','Annual CO₂ emissions (tonnes )']]
df1
# applying groupby() function to 
# group the data on Name value. 
df2 = df1.groupby('Entity').mean()
	
# Let's print the first entries 
# in all the groups formed. 

#df2.rename(columns ={"Annual CO₂ emissions (tonnes )":"emission"},inplace =True)
#df2
#df3 = df2.sort_values(by='Annual CO₂ emissions (tonnes )')

df3=df2.sort_values(by=['Annual CO₂ emissions (tonnes )'],ascending=False, na_position='first')
df4=df3.head(10)
df4
df4.T.plot(kind='bar')
df2.loc[df2['emission'].idxmax()]

