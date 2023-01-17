!pip install --upgrade bamboolib>=1.4.1
# RUN ME AFTER YOU HAVE RELOADED THE BROWSER PAGE

# Uncomment and run lines below
import bamboolib as bam
# Uncomment and run the line below.
# df
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('/kaggle/input/russian-passenger-air-service-20072020/russian_passenger_air_service_2.csv')
df

# bamboolib live code export
df1 = df.groupby(['Airport name', 'Year']).agg(**{'Whole year_sum': ('Whole year', 'sum')}).reset_index()
df1 = df1.sort_values(by=['Whole year_sum'], ascending=[False])
df2007 = df1.loc[df1['Year'] == 2007]
df
# bamboolib live code export
split_df = df["Airport coordinates"].str.split('\'', expand=True)
split_df.columns = [f"Airport coordinates_{id_}" for id_ in range(len(split_df.columns))]
df = pd.merge(df, split_df, how="left", left_index=True, right_index=True)
df = df.drop(columns=['Airport coordinates_0', 'Airport coordinates_2', 'Airport coordinates_4'])
df = df.rename(columns={'Airport coordinates_3': 'Lat Airport coordinates'})
df = df.rename(columns={'Long Airport coordinates_1': 'Long Airport coordinates'})
df



df.head()

df.info()
df.describe().transpose()
df_y = df.groupby('Year')
df_ys = df_y.sum().drop('Whole year', axis=1)
df_ys
plt.figure(figsize=(12,10))
sns.heatmap(df_ys/1000000,annot=True)
plt.title('Tot Pax Number per month')
df['Airport name'].nunique()
df_a = df.groupby('Airport name').sum()
df_a

df[df['Whole year'] == df['Whole year'].max()]['Airport name']
df_a1 = df_a['Whole year'].sort_values(ascending=False)
df_a1.head(10)
#Top Ten
df_asort = df['Whole year'].sort_values(ascending=False)
df_asort.head()
df.iloc[353]
df.iloc[601]
df_a = df.groupby('Airport name')
df_a.head()







