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
df=pd.read_csv('/kaggle/input/cholera-dataset/data.csv')
df
df.info()
df.isnull().sum()
import seaborn as sns

sns.heatmap(df.isnull())
df=df.replace(np.nan,'0',regex=True)

df=df.replace('Unknown','0',regex=True)
sns.heatmap(df.isnull())
df['Country']=df['Country'].astype('str')
df['WHO Region']=df['WHO Region'].astype('str')
df['Number of reported deaths from cholera']=df['Number of reported deaths from cholera'].str.replace('0 0','0')
df['Number of reported deaths from cholera']=df['Number of reported deaths from cholera'].astype('int64')
df['Number of reported cases of cholera']=df['Number of reported cases of cholera'].str.replace('3 5','0')
df['Number of reported cases of cholera']=df['Number of reported cases of cholera'].astype('int64')
df.dtypes
df['Cholera case fatality rate']=df['Cholera case fatality rate'].str.replace('0.0','0')
df['Cholera case fatality rate']=df['Cholera case fatality rate'].str.replace('0 0','0')
df['Cholera case fatality rate']=df['Cholera case fatality rate'].astype('float')
df.dtypes
df_new = df[(df['Year'] <= 2016) & (df['Year'] >= 2007)]
resultant_df=df_new.groupby('Country')['Number of reported cases of cholera'].sum()
import matplotlib.pyplot as plt

resultant_df.sort_values()[:10].plot(kind='bar')

plt.title('countries with least no of cases in calender year 2007-2016')
resultant_df.sort_values()[:10]