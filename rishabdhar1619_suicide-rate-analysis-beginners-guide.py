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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')
df=pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')

df.shape
df.head()
df.info()
df.describe().T
plt.figure(figsize=(12,6))

sns.heatmap(df.corr(),cmap='viridis')
sns.countplot(df['sex'],palette='Set2');

df['sex'].value_counts()
plt.figure(figsize=(15,30))

sns.countplot(y=df['country'])
explode=[0.1,0.1,0.1,0.1,0.3,0.2,0.2,0.1,0.2,0.4]

fig,ax=plt.subplots(1,2,figsize=(12, 6))

df.groupby('country')['suicides_no'].agg(sum).sort_values(ascending=False).head(10).plot(kind='bar',

                                                                                         cmap='Set2',ax=ax[0]);

_=df.groupby('country')['suicides_no'].agg(sum).sort_values(ascending=False).head(10).plot(kind='pie'

                                                                                         ,autopct='%.1f%%'

                                                                                         ,explode=explode

                                                                                         ,startangle=15

                                                                                        ,cmap='Wistia');
plt.figure(figsize=(12,9))

explode=[0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03]

fig,ax=plt.subplots(1,2,figsize=(16, 6))

_=df.groupby('country')['suicides/100k pop'].agg(sum).sort_values(ascending=False).head(10).plot(kind='bar',

                                                                                         cmap='Set2',ax=ax[0]);

_=df.groupby('country')['suicides/100k pop'].agg(sum).sort_values(ascending=False).head(10).plot(kind='pie'

                                                                                         ,autopct='%.1f%%'

                                                                                         ,explode=explode

                                                                                         ,startangle=15

                                                                                        ,cmap='Wistia');

plt.show()
suicide_age=df.pivot_table('suicides_no',index='age',aggfunc='sum')

x=suicide_age.index.values

y=suicide_age.values

y=y.reshape(6,)



fig,ax=plt.subplots(1,2,figsize=(15, 6))

explode=[0.05,0.05,0.05,0.1,0.05,0.05]

_=df.groupby('age')['suicides_no'].agg(sum).sort_values(ascending=False).head(10).plot(kind='bar',cmap='Set2',ax=ax[0]);

_=plt.pie(y,explode=explode,labels=x,autopct='%1.1f%%',shadow=True,startangle=7)

plt.show()
suicide_age=df.pivot_table('suicides/100k pop',index='age',aggfunc='sum')

x=suicide_age.index.values

y=suicide_age.values

y=y.reshape(6,)



fig,ax=plt.subplots(1,2,figsize=(15, 6))

explode=[0.05,0.05,0.05,0.1,0.05,0.05]

_=df.groupby('age')['suicides/100k pop'].agg(sum).sort_values(ascending=False).head(10).plot(kind='bar',cmap='Set2',ax=ax[0]);

_=plt.pie(y,explode=explode,labels=x,autopct='%1.1f%%',shadow=True,startangle=7)

plt.show()
suicide_sex=df.pivot_table('suicides_no',index='sex',aggfunc='sum')

x=suicide_sex.index.values

y=suicide_sex.values

y=y.reshape(2,)



fig,ax=plt.subplots(1,2,figsize=(15, 6))

explode=[0.05,0.05]

_=df.groupby('sex')['suicides_no'].agg(sum).plot(kind='bar',cmap='Set2',ax=ax[0])

_=plt.pie(y,explode=explode,labels=x,autopct='%1.1f%%',shadow=True,startangle=90)

plt.show()
suicide_sex=df.pivot_table('suicides/100k pop',index='sex',aggfunc='sum')

x=suicide_sex.index.values

y=suicide_sex.values

y=y.reshape(2,)



fig,ax=plt.subplots(1,2,figsize=(15, 6))

explode=[0.05,0.05]

_=df.groupby('sex')['suicides/100k pop'].agg(sum).plot(kind='bar',cmap='Set2',ax=ax[0])

_=plt.pie(y,explode=explode,labels=x,autopct='%1.1f%%',shadow=True,startangle=90)

plt.show()
plt.figure(figsize=(12, 6))

df.groupby('age')['generation'].value_counts().sort_values(ascending=False).head(10).plot(kind='bar');
generation_suicide=df.pivot_table('suicides_no',index='generation',aggfunc='sum')

x=generation_suicide.index.values

y=generation_suicide.values

y=y.reshape(6,)



fig,ax=plt.subplots(1,2,figsize=(15, 6))

explode=(0.05,0.05,0.05,0.1,0.05,0.05)

_=df.groupby('generation')['suicides_no'].agg(sum).sort_values(ascending=False).plot(kind='bar',cmap='Set2',ax=ax[0])

_=plt.pie(y,explode=explode,labels=x,autopct='%1.1f%%',shadow=True,startangle=0)

plt.show()
generation_suicide=df.pivot_table('suicides/100k pop',index='generation',aggfunc='sum')

x=generation_suicide.index.values

y=generation_suicide.values

y=y.reshape(6,)



fig,ax=plt.subplots(1,2,figsize=(15, 6))

explode=(0.05,0.05,0.05,0.1,0.05,0.05)

_=df.groupby('generation')['suicides/100k pop'].agg(sum).sort_values(ascending=False).plot(kind='bar',cmap='Set2',ax=ax[0])

_=plt.pie(y,explode=explode,labels=x,autopct='%1.1f%%',shadow=True,startangle=0)

plt.show()
plt.figure(figsize=(12,6))

sns.lineplot(x='year',y='suicides_no',marker='o',data=df,palette='Set2');
plt.figure(figsize=(12,6))

sns.lineplot(x='year',y='suicides/100k pop',marker='o',data=df,palette='Set2');
plt.figure(figsize=(12,6))

sns.lineplot('year','suicides_no',hue='sex',marker='o',data=df);
plt.figure(figsize=(12,6))

sns.lineplot(x='year',y='suicides/100k pop',hue='sex',marker='o',data=df,palette='Set2');
plt.figure(figsize=(12,6))

sns.scatterplot('year','suicides_no',hue='sex',data=df);
plt.figure(figsize=(12,6))

sns.scatterplot('year','suicides/100k pop',hue='sex',data=df);
plt.figure(figsize=(12,6))

sns.lineplot('year','suicides_no',hue='age',marker='o',data=df);
plt.figure(figsize=(12,6))

sns.lineplot(x='year',y='suicides/100k pop',hue='age',marker='o',data=df,palette='Set2')

plt.legend(loc='upper right', bbox_to_anchor=(0.4,0.7,0.7, 0.7));
sns.catplot('age','suicides_no',col='year',data=df,kind='bar',col_wrap=3,palette='Set2');
sns.catplot('age','suicides_no',hue='sex',col='year',data=df,kind='bar',col_wrap=3,palette='Set2');
plt.figure(figsize=(12,6))

sns.lineplot('year','suicides_no',hue='generation',marker='o',data=df)

plt.legend(loc='upper right', bbox_to_anchor=(0.4,0.7,0.7, 0.7));
plt.figure(figsize=(12,6))

sns.lineplot('year','suicides/100k pop',hue='generation',marker='o',data=df)

plt.legend(loc='upper right', bbox_to_anchor=(0.4,0.7,0.7, 0.7));