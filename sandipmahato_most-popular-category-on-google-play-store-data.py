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
df=pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')
df.head()
df['Installs'].isnull().values.any()
df.Installs = df.Installs.apply(lambda x: str(x).replace('+', ''))
df.Installs = df.Installs.apply(lambda x: str(x).replace(',', ''))
df['Installs']
df.drop(index=10472,inplace=True)
df['Installs'] = pd.to_numeric(df['Installs'])
df['Installs']
categ=df['Category'].unique()
ans=df.groupby('Category')['Installs'].sum()
ans
categ
showing=pd.DataFrame({'Install':ans})
showing

showing['Category']=showing.index
showing.reset_index(inplace=True,drop=True)
showing
import seaborn as sns
sns.catplot(data=showing,x='Category',y="Install",size=8)
final_df=showing.sort_values(by='Install',ascending=False)
final_df
sns.catplot(data=final_df[:3],y='Install',x="Category")
final_df[0:1]
