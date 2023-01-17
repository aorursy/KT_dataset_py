import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style="darkgrid")
from mpl_toolkits.basemap import Basemap # For geographical map
df = pd.read_csv("../input/kc_house_data.csv")
df.head(1)
map = Basemap(lat_0=47.0, lon_0=-121)
plt.figure(figsize=[20,10])
map.drawcoastlines(linewidth=.5,color="g")
map.drawcountries(linewidth=.5)
map.drawcounties()
# map.drawstates(color="r")
map.fillcontinents(color="green",alpha=.1)
plt.scatter(df.long,df.lat,alpha=.6,color="r")
plt.show()
df.drop(["id"],axis=1,inplace=True)
corr = df.corr()
plt.figure(figsize=(16,8))
sns.heatmap(corr,annot=True,cmap="RdBu")
plt.figure(figsize=(16,8))
mask = np.zeros_like(corr,dtype=np.bool)

# Create a msk to draw only lower diagonal corr map
mask[np.triu_indices_from(mask)] = True
sns.set_style(style="white")
sns.heatmap(corr,annot=True,cmap="RdBu",mask=mask)
# corr[corr>=.5]
plt.figure(figsize=(16,8))
mask = np.zeros_like(corr[corr>=.5],dtype=np.bool)

# Create a msk to draw only lower diagonal corr map
mask[np.triu_indices_from(mask)] = True
sns.set_style(style="white")
sns.heatmap(corr[corr>=.5],annot=True,mask=mask,cbar=False)
# plt.figure(figsize=(16,8))
# for idx in high_corr.index:
#     corr_idx = high_corr.index[high_corr[idx].notna()]
#     for c_i in corr_idx:
#         if c_i != idx:
#             sns.scatterplot(df[c_i],df[idx],alpha=.2,color="m")
#             plt.show()
# plt.figure(figsize=(16,8))
# for idx in high_corr.index:
#     corr_idx = high_corr.index[high_corr[idx].notna()]
#     for c_i in corr_idx:
#         if c_i != idx:
#             sns.regplot(df[c_i],df[idx],color="g")
#             plt.show()
date_df = df.sort_values(by="date")
plt.figure(figsize=(16,8))
sns.scatterplot(date_df.date,date_df.price,hue=date_df.floors,alpha=.9,size=date_df.grade,palette="winter_r")
plt.xticks([])
plt.show()
plt.figure(figsize=(16,8))
# sns.scatterplot(df.zipcode,df.price,hue=date_df.floors,alpha=.9,size=date_df.grade,palette="winter_r")
# sns.boxplot(df.zipcode,df.price)
sns.swarmplot(df.zipcode,df.price)
plt.xticks([])
plt.show()
plt.figure(figsize=(16,8))
sns.scatterplot(df.condition,df.price)
plt.xticks([])
plt.show()
plt.figure(figsize=(16,8))
yr_price = df.sort_values(by="yr_built")
sns.lineplot(yr_price.yr_built.sort_values(),yr_price.price)
plt.xticks([])
plt.show()
plt.figure(figsize=(16,8))
zip_price = df.sort_values(by="price",ascending=False)
# zip_price
sns.barplot(x=zip_price.zipcode,y=zip_price.price,palette="cool")
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(16,8))
high_corr = corr[corr>=.5]
sns.set_style(style="darkgrid")
sns.scatterplot(df.sqft_living,df.price,hue=df.yr_renovated)
df.head(1)
plt.figure(figsize=(16,4))
date_bedroom = df.sort_values(by="date")
# date_bedroom
sns.scatterplot(date_bedroom.date,date_bedroom.bedrooms,hue=df.bathrooms,size=df.bedrooms,palette="winter")
# plt.xticks(rotation=90)
plt.show()
sns.barplot(df.waterfront,df.price)
sns.barplot(df.view,df.price)
sns.barplot(df.condition,df.price)
plt.figure(figsize=(16,4))
sns.barplot(df.grade,df.price)
plt.figure(figsize=(16,4))
sns.countplot(df.sqft_living[:200],palette="winter_r")
plt.xticks([])
plt.show()