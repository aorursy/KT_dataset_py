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
df=pd.read_csv('/kaggle/input/obesity-among-adults-by-country-19752016/data.csv')
df.head()
df.shape
df.iloc[0,1]
df.columns
df.drop([0,1,2],inplace=True)

df.head()
df.reset_index(drop=True,inplace=True)
df.head()
df.rename(columns={'Unnamed: 0': 'Country'}, inplace=True)
df.head()
ndf = df.melt('Country', var_name='Year', value_name='Obesity (%)')

ndf[['Year', 'Sex']] = ndf['Year'].str.split('.', expand=True)
ndf.head(10)
ndf=ndf.sort_values(by=['Country','Year'])
ndf=ndf.reset_index(drop=True)
ndf.head()
ndf['Sex']=ndf['Sex'].map({None: 'Both sexes', '1': 'Male', '2':'Female'})
ndf['Age standardized estimate']=ndf['Obesity (%)'].apply(lambda x:x.split()[1])
ndf['Obesity (%)']=ndf['Obesity (%)'].apply(lambda x:x.split()[0])
ndf.head()
ndf.to_csv('/kaggle/working/obesity-clean-split.csv')