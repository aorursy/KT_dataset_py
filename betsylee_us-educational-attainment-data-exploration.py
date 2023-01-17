import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# import data visualization library

import seaborn as sns
df = pd.read_csv('/kaggle/input/us-educational-attainment-19952015/1995_2015.csv')
df.info()
df.head()
sns.barplot(x = 'Sex', y = 'Bachelor', data = df, palette = 'coolwarm')
sns.barplot(x = 'Sex', y = 'Advanced', data = df, palette = 'coolwarm')
sns.barplot(x = 'Sex', y = 'Associate', data = df, palette = 'coolwarm')
sns.barplot(x = 'Sex', y = 'Some_College_No_Degree', data = df, palette = 'coolwarm')
sns.barplot(x = 'Sex', y = 'HS_Diploma', data = df, palette = 'coolwarm')
sns.barplot(x = 'Sex', y = 'No_HS_Diploma', data = df, palette = 'coolwarm')
sns.barplot(x = 'Sex', y = 'No_HS_Diploma', data = df, hue = 'Year', palette = 'coolwarm')
sns.barplot(x = 'Sex', y = 'HS_Diploma', data = df, hue = 'Year', palette = 'coolwarm')
sns.barplot(x = 'Sex', y = 'Some_College_No_Degree', data = df, hue = 'Year', palette = 'coolwarm')
sns.barplot(x = 'Sex', y = 'Associate', data = df, hue = 'Year', palette = 'coolwarm')
sns.barplot(x = 'Sex', y = 'Bachelor', data = df, hue = 'Year', palette = 'coolwarm')
sns.barplot(x = 'Sex', y = 'Advanced', data = df, hue = 'Year', palette = 'coolwarm')
df.describe()
df[df['Year']==1995].describe()
df[df['Year']==2005].describe()
df[df['Year']==2015].describe()
