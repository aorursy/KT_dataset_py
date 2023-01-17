

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv("../input/avocado.csv")
df.head()
# Any results you write to the current directory are saved as output.
import seaborn as sns
%matplotlib inline
#sns.set_style()
#df.columns
sns.catplot(x='Total Volume',y='region',hue='year',col = 'type',data=df,kind = "point", height = 10,aspect = 0.5, join=False, sharex=False)
df = df[df['region'] != 'TotalUS']
df.sort_values(by=['Total Volume'],inplace=True, ascending=False)
sns.catplot(x='Total Volume',y='region',hue='year',col = 'type',data=df,kind = "point", height = 10,aspect = 0.8, join=False,sharex=False)
df.sort_values(by=['AveragePrice'],inplace=True, ascending=False)
sns.catplot(x='AveragePrice',y='region',hue='year',col = 'type',data=df,kind = "point", height = 10,aspect = 0.8, join=False,sharex=False)
sort_list_average_price = df[(df['type'] == 'conventional') & (df['year'] == 2018)].groupby('region')['AveragePrice'].mean().sort_values().index#sort_list_average_price = mask['region'].tolist()
sns.catplot(x='AveragePrice',y='region',hue='year',col = 'type',data=df,kind = "point", height = 10,aspect = 0.8, join=False,sharex=False,order=sort_list_average_price, col_order = ['conventional','organic'])