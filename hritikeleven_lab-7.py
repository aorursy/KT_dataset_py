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
df=pd.read_csv("/kaggle/input/co2-emission/co2_emission.csv")
df
import pandas as pd
df=pd.read_csv("/kaggle/input/co2-emission/co2_emission.csv")
df.columns = ['Entity', 'Code', 'Year','Emissions']
df1=df[['Entity','Emissions']]
df1
df2 = df1.groupby('Entity').mean()
df2
print(df2.describe(include='all'))
df2 = df1.groupby('Entity').mean()
df2['Emissions'].max()
df2 = df1.groupby('Entity').mean()
df2['Emissions'].min()
df3=df2.sort_values(by=['Emissions'],ascending=False, na_position='first')
df4=df3.head(10)
df4
df4.T.plot(kind='bar')
df3=df2.sort_values(by=['Emissions'],ascending=True, na_position='first')
df4=df3.head(10)
df5=df4.drop(['Statistical differences'])
df5
df5.T.plot(kind='bar')


df7=df[df['Year']>2000]
df8 = df7.groupby('Entity').mean()
df9=df8.sort_values(by=['Emissions'],ascending=True, na_position='first')
df10=df9.drop(['Statistical differences'])
df11=df10.drop(['Year'], axis = 1) 
df12=df11.head(10)
df12
df12.T.plot(kind='bar')
df7=df[df['Year']>2000]
df8 = df7.groupby('Entity').mean()
df9=df8.sort_values(by=['Emissions'],ascending=False, na_position='first')
df10=df9.drop(['Statistical differences'])
df11=df10.drop(['Year'], axis = 1) 
df12=df11.head(10)
df12
df12.T.plot(kind='bar')
df7= df[df['Year'].between(2012, 2019)]
df7

df8 = df7.groupby('Year').mean()
df8
df8.T.plot(kind='bar')
import matplotlib.pyplot as plt

df

df.plot(x ='Year', y='Emissions', kind = 'line')
plt.show()
df3=df2.sort_values(by=['Emissions'],ascending=False, na_position='first')
df4=df3.drop(['World'])
df5=df4.head(5)
df5.T.plot(kind='bar')
import matplotlib.pyplot as plt
df3=df2.sort_values(by=['Emissions'],ascending=False, na_position='first')
df4=df3.drop(['World'])
df5=df4.head(5)
df5