

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from __future__ import (absolute_import,division,print_function,unicode_literals)
import warnings
warnings.simplefilter('ignore')

%pylab inline
from pylab import rcParams
rcParams['figure.figsize']=8,5
import seaborn as sns
df = pd.read_csv('../input/Video_Games_Sales_as_at_22_Dec_2016.csv')
df.info()
df.head()
df = df.dropna()
df['User_Score'] = df.User_Score.astype('float64')
df['Year_of_Release'] = df.Year_of_Release.astype('int64')
df['User_Count'] = df.User_Count.astype('int64')
df['Critic_Count'] = df.Critic_Count.astype('int64')
df.shape
useful_cols = ['Name','Platform','Year_of_Release','Genre','Global_Sales',
              'Critic_Score','Critic_Count','User_Score','User_Count',
              'Rating']
df[useful_cols].head()
sales_df = df[[x for x in df.columns if 'Sales' in x]+['Year_of_Release']]
sales_df.groupby('Year_of_Release').sum().plot()
sales_df.groupby('Year_of_Release').sum().plot(kind='bar',rot=45)
cols = ['Global_Sales','Critic_Score', 'Critic_Count', 'User_Score','User_Count']
sns_plot = sns.pairplot(df[cols])
sns_plot.savefig('pairplot.png')
sns.distplot(df.Critic_Score)
sns.jointplot(df['Critic_Score'],df['User_Score'])
top_platforms = df.Platform.value_counts().sort_values(ascending=False).head(5).index.values
sns.boxplot(y='Platform',x='Critic_Score',data=df[df.Platform.isin(top_platforms)],orient='h')
platform_genre_sales = df.pivot_table(index='Platform',columns='Genre',
                                     values='Global_Sales',
                                     aggfunc=sum).fillna(0).applymap(float)
sns.heatmap(platform_genre_sales,annot=True,fmt=".1f",linewidth=.5)