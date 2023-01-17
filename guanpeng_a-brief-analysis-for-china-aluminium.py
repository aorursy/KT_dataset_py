import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print('set up complete')
import pandas as pd

df = pd.read_csv("../input/global-commodity-trade-statistics/commodity_trade_statistics_data.csv")
df.head()
df.category.unique()
alum=df.loc[(df.category=='76_aluminium_and_articles_thereof')]

df_byyear=alum.groupby(['country_or_area','year','flow']).weight_kg.agg([sum])

df_byyear.tail()
df_im=df_byyear['sum'].loc['China',:,'Import'].reset_index()

df_ex=df_byyear['sum'].loc['China',:,'Export'].reset_index()

Imports_cn=df_im.loc[:,['year','sum']].set_index('year').rename(columns={'sum':'Imports'})

Exports_cn=df_ex.loc[:,['year','sum']].set_index('year').rename(columns={'sum':'Exports'})

Imports_cn.head()
sns.lineplot(data=Imports_cn)

sns.lineplot(data=Exports_cn,palette='pastel')

plt.ylabel('Weight/kg')

plt.title('China Imports VS Exports')
df_2016=df.loc[(df.year==2016)&(df.country_or_area=='China')&(df.category=='76_aluminium_and_articles_thereof')&(df.flow=='Export')].sort_values(by='weight_kg',ascending=False)

df_2016.head()
plt.figure(figsize=(18,5))

sns.barplot(y=df_2016['commodity'],x=df_2016['weight_kg'])

plt.title('China Aluminium Export commodity 2016')