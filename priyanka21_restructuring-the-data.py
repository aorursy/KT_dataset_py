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
import os
os.chdir('/kaggle/input')
df = pd.read_csv('obesity-among-adults-by-country-19752016/data.csv')
df.head()
df.info()
df.shape
df_clean = pd.read_csv('obesity-among-adults-by-country-19752016/obesity-cleaned.csv')
df_clean.head()
df.iloc[0,1]
df.columns
df.head()
df.drop(0,inplace=True)
df.head()
df.set_index('Unnamed: 0',inplace=True)
df.head()
import numpy as np
df = df.rename(index={np.nan:'Age','Country':'Sex'})
df.head()
df_clean.head()
df.reset_index(inplace=True)
df.head()
df.columns[1:]
df = pd.melt(df, id_vars=['Unnamed: 0'], value_vars=df.columns[1:], var_name='Year', value_name='Obesity %')
df.head()
temp = df.iloc[[0,1]]
temp
df.drop([0,1],inplace=True)
df.head()
df.rename(columns={'Unnamed: 0' : 'Country'}, inplace=True)
df.head()
df.sort_values(by=['Year','Country'],inplace=True)
df.head()
df.reset_index(drop=True,inplace=True)
df.head()
df.index[df['Country'] == 'Age']
df.drop(df.index[df['Country'] == 'Age'],inplace=True)
df.head()
df.reset_index(drop=True,inplace=True)
df.head()
