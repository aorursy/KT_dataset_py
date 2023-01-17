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
df = pd.read_csv('/kaggle/input/szeged-weather/weatherHistory.csv')
df.head()
df.describe()
df.info()
for col in df.columns:
    if len(df[col].unique() )< 30:
        print("")
        print(len(df[col].unique()),"is the unique number of values in the column->",col,"and this is a CATEGORICAL Variable")
        print(df[col].unique())
        print("")
        print("")
for col in df.columns:
    if len(df[col].unique() )> 30:
        print("")
        print(len(df[col].unique()),"is the unique number of values in the column->",col,"and this is a CONTINOUS Variable")
        if(len(df[col].unique()) < 100): #just to make sure , there is no variable with a large number of categories
            print(df[col].unique())
        print("")
        print("")
print(100-(df['Precip Type'].count()/len(df))*100)
print("percent of data is missing")

df_precip_drop = df.dropna()
df_precip_drop.info()
import seaborn as sns
ax = sns.countplot(x="Precip Type",  data=df)
df_mode_handled = df
for i in range(len(df)):
    if(df['Precip Type'].iloc[i]!='snow' and df['Precip Type'].iloc[i]!='rain'):
        df_mode_handled['Precip Type'].iloc[i] = 'rain'
ax = sns.countplot(x="Precip Type",  data=df_mode_handled)
df_unique_missing_handler = df
for i in range(0,len(df)):
    if(df['Precip Type'].iloc[i]!='rain' and df['Precip Type'].iloc[i]!='snow'):
        df_unique_missing_handler['Precip Type'].iloc[i] = 'Unknown'
ax = sns.countplot(x="Precip Type",  data=df_unique_missing_handler)
sns.catplot(x="Precip Type", y="Temperature (C)", data=df)
for i in range(0,len(df)):
    if(df['Precip Type'].iloc[i] not in ['rain','snow']):
        if(df['Temperature (C)'].iloc[i]>0):
            df['Precip Type'].iloc[i] = 'rain'
        else:
            df['Precip Type'].iloc[i] = 'snow'
df.info()
sns.catplot(x="Precip Type", y="Temperature (C)", data=df)
from category_encoders import TargetEncoder

encoder = TargetEncoder()
df['Summary Encoded'] = encoder.fit_transform(df['Summary'], df['Temperature (C)'])
df['Summary Encoded']
df=df.drop(columns= 'Loud Cover')
df.head()
sns.scatterplot(data=df, x="Temperature (C)", y="Apparent Temperature (C)")
