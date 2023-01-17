import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
data = pd.read_csv('../input/tabelog-restaurant-review-dataset/Kyoto_Restaurant_Info.csv')
data.head()
data.dtypes
import folium
from folium import plugins
print( "folium version is {}".format(folium.__version__) )
# 地図作成して全体を把握

LAT = data.Lat.iloc[0]
LNG = data.Long.iloc[0]

m = folium.Map(location=[LAT, LNG], zoom_start=10)
data_nodup = data.drop_duplicates(subset=["Lat", "Long"])
for lat, lon in zip(data_nodup.Lat, data_nodup.Long):
    folium.Marker(
        location=[lat, lon],
        popup="/".join([str(lat), str(lon)]),
        tooltip=str(lat) + "_" + str(lon),
    ).add_to(m)
m
data['TotalRating'].max()
data.describe()
data.isnull().sum()
df = data.fillna(data.mean())
df.isnull().sum()
# ランクが４．０以上のお店

df[df['TotalRating'] > 4.0]
# ディナーの値段別の点数

plt.figure(figsize=(20, 10))
sns.boxplot(x="DinnerPrice", y="TotalRating", data=df)
df['DinnerPrice'].unique()
df.replace('￥4000～￥4999', '4000', inplace=True)
df.replace('￥3000～￥3999', '3000', inplace=True)
df.replace('￥8000～￥9999', '8000', inplace=True)
df.replace('￥2000～￥2999', '2000', inplace=True)
df.replace('￥10000～￥14999', '10000', inplace=True)
df.replace('￥5000～￥5999', '5000', inplace=True)
df.replace('￥6000～￥7999', '6000', inplace=True)
df.replace('￥20000～￥29999', '20000', inplace=True)
df.replace('￥1000～￥1999', '1000', inplace=True)
df.replace('～￥999', '500', inplace=True)
df.replace('￥15000～￥19999', '15000', inplace=True)
df.replace('￥30000～', '30000', inplace=True)
df.head()
df.dtypes
df["DinnerPrice"] = pd.to_numeric(df["DinnerPrice"], errors='coerce')
df["LunchPrice"] = pd.to_numeric(df["LunchPrice"], errors='coerce')
df.dtypes
df['ce_d'] = df['DinnerPrice'] / df['DinnerRating']
df['ce_l'] = df['LunchPrice'] / df['LunchRating']
df.head()
df['ce_d'].min()
df[df['ce_d'] < 300]
x = df.ce_d
y = df.TotalRating

plt.scatter(x, y)
df_９９ = df.query('(DinnerPrice < 8000) & (DinnerRating > 3.5)')
df_99.head()
df_9920 = df_99.sort_values(by="DinnerRating", ascending=False)[0:20]
df_9920.head()
df_100 = df.query('(DinnerPrice > 10000) & (DinnerRating > 3.5)')
df_100.head()
df_10020 = df_100.sort_values(by="DinnerRating", ascending=False)[0:20]
df_10020.head()
# 最高点数の「京懐石 吉泉」の場所を見てみる

LAT = data.Lat.iloc[0]
LNG = data.Long.iloc[0]

m = folium.Map(location=[LAT, LNG])
folium.Marker(
    location=[data.Lat.iloc[767], data.Long.iloc[767]],
    popup="This is Simple Marker",
).add_to(m)
m
df_fc = data.groupby(["FirstCategory"])["TotalRating"].count()
df_fc = df_fc.sort_values(ascending=False)[0:30]
df_fc
df_ikitai = df.query('(DinnerPrice < 10000) & (DinnerRating > 3.5) & (FirstCategory == "Steak")')
df_ikitai
# 1万円以下で、点数３．５以上のステーキ店「プランチャー健」の場所を見てみる

LAT = data.Lat.iloc[0]
LNG = data.Long.iloc[0]

m = folium.Map(location=[LAT, LNG])
folium.Marker(
    location=[data.Lat.iloc[304], data.Long.iloc[304]],
    popup="This is Simple Marker",
).add_to(m)
m

# 点数３．５以上、5000円以下で飲める居酒屋リスト

df_ikitai_izakaya = df.query('(DinnerPrice < 5000) & (DinnerRating > 3.5) & (FirstCategory == "Izakaya (Tavern)")')
df_ikitai_izakaya
# 石塀小路 豆ちゃ 京都

LAT = data.Lat.iloc[0]
LNG = data.Long.iloc[0]

m = folium.Map(location=[LAT, LNG])
folium.Marker(
    location=[data.Lat.iloc[9], data.Long.iloc[9]],
    popup="This is Simple Marker",
).add_to(m)
m
plt.figure(figsize=(30, 20))
x = df['FirstCategory'][0:50]
sns.boxplot(x, y="TotalRating", data=df)