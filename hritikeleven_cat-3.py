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

df= pd.read_csv("/kaggle/input/world-happiness/2019.csv")
df
print(df.describe(include='all'))
df1=df[['Country or region','Score','Social support','Healthy life expectancy','Generosity','Freedom to make life choices']]
df1
df2 = df.groupby('Country or region').mean()
df2
df4 = df.groupby('Country or region').median()
df4
df3 = df.mode()
df3
numeric_data = list(df._get_numeric_data().columns)
numeric_data
categorical_data = list(set(df.columns) - set(df._get_numeric_data().columns))
categorical_data
import matplotlib.pyplot as plot
df1=df[['Country or region','Score']]

df2 = df1.groupby('Country or region').mean()

df3=df2.sort_values(by=['Score'],ascending=False, na_position='first')
df4=df3.head(20)
df4
df4.T.plot(kind='bar')
import matplotlib.pyplot as plot
df1=df[['Country or region','Score']]

df2 = df1.groupby('Country or region').mean()

df3=df2.sort_values(by=['Score'],ascending=True, na_position='first')
df4=df3.head(20)
df4
df4.T.plot(kind='bar')
import matplotlib.pyplot as plot
df1=df[['Country or region','Social support']]
df2 = df1.groupby('Country or region').mean()
df3=df2.sort_values(by=['Social support'],ascending=False, na_position='first')
df4=df3.head(20)
df4.T.plot(kind='bar')
import matplotlib.pyplot as plot
df1=df[['Country or region','Social support']]
df2 = df1.groupby('Country or region').mean()
df3=df2.sort_values(by=['Social support'],ascending=True, na_position='first')
df4=df3.head(20)
df4.T.plot(kind='bar')
import matplotlib.pyplot as plot
df1=df[['Country or region','Healthy life expectancy']]
df2 = df1.groupby('Country or region').mean()
df3=df2.sort_values(by=['Healthy life expectancy'],ascending=False, na_position='first')
df4=df3.head(20)
df4.T.plot(kind='bar')
import matplotlib.pyplot as plot
df1=df[['Country or region','Healthy life expectancy']]
df2 = df1.groupby('Country or region').mean()
df3=df2.sort_values(by=['Healthy life expectancy'],ascending=True, na_position='first')
df4=df3.head(20)
df4.T.plot(kind='bar')

column_1 = df["Score"]
column_2 = df["Healthy life expectancy"]
correlation = column_1.corr(column_2)
correlation
column_1 = df["Score"]
column_2 = df["Social support"]
correlation = column_1.corr(column_2)
correlation
column_1 = df["Score"]
column_2 = df["Freedom to make life choices"]
correlation = column_1.corr(column_2)
correlation
column_1 = df["Score"]
column_2 = df["Generosity"]
correlation = column_1.corr(column_2)
correlation

column_1 = df["Score"]
column_2 = df["Perceptions of corruption"]
correlation = column_1.corr(column_2)
correlation
df1=df.head(20)

correlation_df1 = df1.corr()
correlation_df1
import seaborn as sns
plot.figure(figsize=(20,20))
sns.heatmap(correlation_df1, annot=True, cmap="Reds")
plot.title('Correlation Heatmap', fontsize=20)


