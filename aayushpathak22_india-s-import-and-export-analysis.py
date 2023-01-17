import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
%matplotlib inline
uncleaned_data = pd.read_csv('../input/commodity_trade_statistics_data.csv')
#cleaned data
global_trade_df = uncleaned_data.dropna(how='all')
global_trade_df.shape
global_trade_df.head(10)
global_trade_df.count()
# the array of unique categories
global_trade_df['category'].unique()
#get the top 10 commodity which contribute maximum in the trade 
df=global_trade_df.groupby('commodity').trade_usd.mean().reset_index(name='trade_usd')
df = df.nlargest(10,'trade_usd').reset_index()
df
#checking data type of the column 'year'
global_trade_df['year'].unique()
trade_by_country = global_trade_df[['country_or_area','year','flow', 'category' ,'trade_usd']]

#using groupby function and building a multiIndex to make analysis easier
trade_by_country = trade_by_country.groupby(['country_or_area','year','flow', 'category'])[['trade_usd']].sum()
trade_by_country.head(30)
India_df = global_trade_df[global_trade_df['country_or_area'] == 'India']
India_years = India_df['year'].unique()
India_years.sort()

exports_br = trade_by_country['trade_usd'].loc['India', : ,'Export', 'all_commodities']
imports_br = trade_by_country['trade_usd'].loc['India', : ,'Import', 'all_commodities']


fig=plt.figure(figsize=(10, 8), dpi= 80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size':15})


p2 = plt.bar(India_years, imports_br)
p1 = plt.bar(India_years, exports_br)

plt.title("Exports vs Imports")
plt.ylabel('Trade worth - in 100 billion US dollars')
plt.xlabel('year')
plt.legend((p1, p2), ('Exports', 'Imports'))
# Most Imp commodity of Imp India
India_df[India_df['flow']=='Import'].groupby('commodity').trade_usd.mean()
del(global_trade_df)
India_Import=India_df[India_df['flow']=='Import']
df=India_Import.groupby('commodity').trade_usd.mean().head(10).reset_index()
df
#df = df.to_frame().reset_index()
#drop 'All commodities' to get the barplot properly 
df=df.drop(0).reset_index(drop=True)  
#df.loc[0]==""
sn.barplot(x='trade_usd',y='commodity',data=df)
#sn.set_xticklabels(rotation=30)
plt.xticks(rotation=20)
plt.title("India's Top 10 Import Commodity")
plt.xlabel("Trade worth - in 100 billion US dollars")
plt.ylabel("Commodity")
# Global_India=Global_India[Global_India['flow']=='Export']
# df=Global_India.groupby('commodity').trade_usd.agg(['count','min','max','mean']).head(10)
India_Export=India_df[India_df['flow']=='Export']
#df=India_Export.groupby('commodity').trade_usd.mean().head(10).reset_index()
df
df=df.drop(0).reset_index(drop=True)

sn.barplot(x='trade_usd',y='commodity',data=df)
#sn.set_xticklabels(rotation=30)
plt.xticks(rotation=20)
plt.title("India's Top 10 Export Commodity")
plt.xlabel("Trade worth - in 100 billion US dollars")
plt.ylabel("Commodity")
