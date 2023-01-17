import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv("../input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv")

df.head()
df1 = df.copy()
df1.drop(["currency_buyer","has_urgency_banner","urgency_text","origin_country","merchant_info_subtitle","merchant_id","merchant_has_profile_picture","merchant_profile_picture","theme","crawl_month"],axis=1,inplace  =True)

df1
df1.isnull().sum()
df1["product_variation_size_id"].unique()
df1["product_variation_size_id"].replace(["choose a size","One Size","Base & Top & Matte Top Coat","20pcs","Pack of 1","1 pc.","AU plug Low quality","5PAIRS","10 ml","10pcs","first  generation",""], np.NaN, inplace=True)
df1["product_variation_size_id"].unique()
df1.isnull().sum()
df1["product_color"].unique()
df1[df1["rating_five_count"].isnull()==True].head(2)
df1.dropna(inplace=True)
df1.head(3)
df1.describe()
df1.corr()
df1.isnull().sum()
x = df1["rating_five_count"]#,df1["rating_four_count"],df1["rating_three_count"],df1["rating_two_count"],df1["rating_one_count"]

y = df1["units_sold"]

plt.bar(x,y)
df1.hist(bins=30, figsize=(20,20))
df1["tags"]
df["units_sold"].sum()
df["rating_five_count"].sum()
df["rating_four_count"].sum()
df["rating_three_count"].sum()
df["rating_two_count"].sum()
df["rating_one_count"].sum()
x=[675779.0,274428.0,205592.0,97351.0,146284.0]

plt.pie(x)
len(df["merchant_name"].unique())
#x=df1["merchant_name"].head()

#y=df1["units_sold"]

#lt.plot(x,y)
df1["countries_shipped_to"].unique()
df1["product_variation_inventory"].unique()
df1.head(1)
df1.drop(["inventory_total","merchant_title","product_url","product_picture","product_id"],axis=1,inplace=True)
df1.head(1)
df1["shipping_option_name"].unique()
x = df1["shipping_option_name"]

y = df1["units_sold"]

fig = plt.figure()

ax1 = plt.subplot2grid((1,1),(0,0))

for label in ax1.xaxis.get_ticklabels():

  label.set_rotation(90)

plt.bar(x,y)
df1.head(1)
x = df1["product_color"]

y = df1["units_sold"]

fig = plt.figure()

ax1 = plt.subplot2grid((1,1),(0,0))

for label in ax1.xaxis.get_ticklabels():

  label.set_rotation(90)

plt.bar(x,y)
import itertools

import matplotlib.animation as animation
fig = plt.figure()

ax1 = fig.add_subplot(1,1,1)

def animate(i):

  graph_data=open("e_commerce.csv").read()

  x=df1["product_color"]

  y=df1["units_sold"]

#ax1.clear()

ax1.plot(x,y)

#fig, ax = plt.subplots()

#line = ax.plot(df1["product_color"], df1["units_sold"], lw=2)

#ax.grid()

#xdata, ydata = df1["product_color"], df1["units_sold"]



ani = animation.FuncAnimation(fig, animate, interval=10)

plt.show()
x=df1["product_color"]

y=df1["units_sold"]

fig = plt.figure()

ax1 = plt.subplot2grid((1,1),(0,0))

for label in ax1.xaxis.get_ticklabels():

  label.set_rotation(90)

plt.scatter(x,y,linewidths=0.00001)
df1.head(1)
plt.hist(df1["uses_ad_boosts"],bins=2,rwidth=0.5)