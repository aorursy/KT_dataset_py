# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/database.csv')

print(df.head())
print(df.info())
df1 = df[['Record ID','Model Year']].groupby(by=['Model Year']).count().rename(columns={'Record ID':'Recalls'})


df2 = df1.drop([9999])

print(df2)
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

plt.plot(df2)
plt.xlabel('Model Year')
plt.ylabel('Recalls')

plt.show()
df2 = df[['Record ID','Vehicle Make']].groupby(by=['Vehicle Make']).count().sort_values(by='Record ID',ascending=False).rename(columns={'Record ID':'Recalls'})
print(df2)
df_make= df[['Vehicle Make','Model Year','Record ID']]


df_make_2016 = df_make[df_make['Model Year']==2016]


df_make_2016_num = df_make_2016.groupby('Vehicle Make').count().drop('Model Year',axis=1).sort_values(by='Record ID',ascending=False).rename(columns={'Record ID':'Recalls'})
print(df_make_2016_num)


print(df_make_2016_num[df_make_2016_num == df_make_2016_num.max()].dropna())

df_type = df.groupby(by='Recall Type').count()
#print(df_type.head())

df_tires = df[df['Recall Type']=='TIRE'][['Record ID','Vehicle Make']].groupby(by='Vehicle Make').count().sort_values(by='Record ID',ascending=False).reset_index().rename(columns={'Record ID':'Recalls','Vehicle Make':'Tire make'})
print(df_tires)

df_tires_most = df_tires.loc[:15]
sns.barplot(y=df_tires_most['Tire make'],x=df_tires_most['Recalls'])
plt.title('The 15 Tire Makers with the most Recalls')

plt.show()