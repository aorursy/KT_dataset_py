# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

path = '/kaggle/input/sf-salaries/Salaries.csv'
df = pd.read_csv(path)
print(df.head())
print('~~~~~~~~~~~~~')
print(df.shape)
print('~~~~~~~~~~~~~')
print(df.columns)
print('~~~~~~~~~~~~~')
print(df.isna().sum())
print ('~~~~~~~~~~~~')
df = df.drop(['Id','Notes','Agency','Status'],axis=1)
df.head()
df['JobTitle'] = df['JobTitle'].str.upper()
df.JobTitle
df.dtypes
df['BasePay'] = df['BasePay'].replace('Not Provided', np.nan)
df['BasePay'] = df['BasePay'].replace(0, np.nan)
df.BasePay = df.BasePay.astype('float64')

df['OvertimePay'] = df['OvertimePay'].replace('Not Provided', np.nan)
df['OvertimePay'] = df['OvertimePay'].replace(0, np.nan)
df.BasePay = df.OvertimePay.astype('float64')

df['OtherPay'] = df['OtherPay'].replace('Not Provided', np.nan)
df['OtherPay'] = df['OtherPay'].replace(0, np.nan)
df.BasePay = df.OtherPay.astype('float64')

df['Benefits'] = df['Benefits'].replace('Not Provided', np.nan)
df['Benefits'] = df['Benefits'].replace(0, np.nan)
df.BasePay = df.Benefits.astype('float64')
df.describe()
df_duplicated = df[df['JobTitle'].duplicated()]

df_duplicated.head()
df_unique = df['JobTitle'].unique()

df_unique
selected_professions = ['CHEMIST',
        'BIOLOGIST',
        'EPIDEMIOLOGIST',
        'NUTRITIONIST']

df1 = df[df.JobTitle.str.contains('|'.join(selected_professions))]
df2 = df1.groupby(['TotalPay','JobTitle']).count()['EmployeeName']
fig1 = plt.figure(figsize=(20,8))
chart1 = sns.boxplot(data=df1, 
        x='JobTitle',
        y='TotalPay',
        hue='Year')
chart1.set_xticklabels(chart1.get_xticklabels(), rotation=25)
df3 = df[df['JobTitle'].str.contains('EPIDEMIOLOGIST', regex=False)]
fig2 = plt.figure(figsize=(8,5))
chart2 = sns.boxplot(data=df3, 
        x='JobTitle',
        y='TotalPay',
        hue='Year')
chart2.set_xticklabels(chart2.get_xticklabels(), rotation=25)
fig3 = plt.figure(figsize=(15,5))
chart3 = sns.countplot(data=df3[df3['Year']==2011], 
#        x='JobTitle',
        x='TotalPay',
        hue='JobTitle')
chart3.set_xticklabels(chart3.get_xticklabels(), rotation=25)