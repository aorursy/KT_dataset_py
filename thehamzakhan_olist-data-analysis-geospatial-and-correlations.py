# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from scipy.special import boxcox
import os
from os import path
print(os.listdir("../input"))

%matplotlib inline
import plotly
import plotly.plotly as py
import plotly.offline as offline
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
from plotly.graph_objs import Scatter, Figure, Layout
from plotly import tools
# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/olist_classified_public_dataset.csv")
df2 = pd.read_csv("../input/geolocation_olist_public_dataset.csv")
df.head(10)
translate_df = pd.read_csv("../input/product_category_name_translation.csv")
translate_df.head()
### all of product categories names are translated into english
for i in range(0,len(translate_df)):
    df.product_category_name[df.product_category_name==translate_df.iloc[i,0]] = translate_df.iloc[i,1]
plt.figure(figsize=(50,60))
sns.countplot(y=df.product_category_name,orient="v")
plt.yticks(fontsize=35)
plt.xticks(fontsize=30)
plt.ylabel("Product Category Name", fontsize=40)
plt.xlabel("Product Count",fontsize=40)
plt.title("Product Category Count",fontsize=80)
plt.show()
soup = ' '.join(df.product_category_name)
#wordcloud = WordCloud().generate()

wordcloud = WordCloud(width=5000, height=2500,max_words=300)
wordcloud.generate(soup)
plt.figure(figsize=(20,10),facecolor='k')
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
soup = ' '.join(df.review_comment_message)
#wordcloud = WordCloud().generate()

wordcloud = WordCloud(width=5000, height=2500,max_words=300)
wordcloud.generate(soup)
plt.figure(figsize=(20,10),facecolor='k')
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
df.head()
### convert data types of date columns into pd.datetime
time_columns = ['order_purchase_timestamp','order_aproved_at','order_estimated_delivery_date','order_delivered_customer_date']
for x in time_columns:
    df[x] = pd.to_datetime(df[x])
### calculate time delay between approving purchase
### calculate time gap between estimated delivery date and actual ordered date

df['approved_delay'] = df['order_aproved_at']-df['order_purchase_timestamp']
df['delivery_gap_btw_est_act'] = df['order_delivered_customer_date']-df['order_estimated_delivery_date']
### convert time delay between approving purchase into minutes data

approved_time = pd.DatetimeIndex(df['approved_delay'])
approved_minutes = approved_time.hour*60 + approved_time.minute
### Approved minutes into histogram

plt.figure(figsize=(15,8))
ax = plt.subplot(1,1,1)
sns.distplot(list(approved_minutes),bins=50,color='c')
ax.set_xticks(range(0,1800,60))
plt.title("Approval Time Delay Histogram",fontsize=20)
plt.xlabel("Approval Delay in minutes", fontsize=10)
plt.show()
### convert time delay between approving purchase into minutes data
Deliver_gap = pd.DatetimeIndex(df['delivery_gap_btw_est_act'])
Deliver_gap = Deliver_gap.day-1
CleanedList = [x for x in Deliver_gap if str(x) != 'nan']
### Approved minutes into histogram

plt.figure(figsize=(15,8))
ax = plt.subplot(1,1,1)
sns.distplot(CleanedList,bins=50,color='orange')
#ax.set_xticks(range(0,1800,60))
plt.title("Time gap between Estimated Delivery Date and Actual Date"+"\n (Days Faster than estimated)",fontsize=20)
plt.xlabel("Dates", fontsize=12)
plt.show()
df2 = pd.read_csv("../input/geolocation_olist_public_dataset.csv").sample(n=50000)

mapbox_access_token = 'pk.eyJ1IjoibGVlZG9oeXVuIiwiYSI6ImNqbjl1Y2hmcTB6dTQzcnBiNDZ2cXcwbGEifQ.hcPVtUhnyzXDXZbQQH0nMw'
data = [go.Scattermapbox(
    lon = df2['lng'],
    lat = df2['lat'],
    marker = dict(
        size = 3,
        
    ))]

layout = dict(
        title = 'Geo Locations based on Zip code',
        mapbox = dict(
            accesstoken = mapbox_access_token,
            center= dict(lat=-20,lon=-60),
            bearing=5,
            pitch=5,
            zoom=2.3,
        )
    )
fig = dict( data=data, layout=layout )
iplot( fig, validate=False)
### longitude and latitude are recored into new columns in df

df['customer_latitude'] = pd.Series([df2.loc[df2.zip_code_prefix==df.customer_zip_code_prefix[x],:].lat.mean() for x in range(0,len(df)-1)])
df['customer_longitude'] = pd.Series([df2.loc[df2.zip_code_prefix==df.customer_zip_code_prefix[x],:].lng.mean() for x in range(0,len(df)-1)])

mapbox_access_token = 'pk.eyJ1IjoibGVlZG9oeXVuIiwiYSI6ImNqbjl1Y2hmcTB6dTQzcnBiNDZ2cXcwbGEifQ.hcPVtUhnyzXDXZbQQH0nMw'
data = [go.Scattermapbox(
    lon = df['customer_longitude'],
    lat = df['customer_latitude'],
    marker = dict(
        size = 5,
        color = 'red'
    ))]

layout = dict(
        title = 'Customer Locations',
        mapbox = dict(
            accesstoken = mapbox_access_token,
            center= dict(lat=-20,lon=-60),
            bearing=5,
            pitch=5,
            zoom=2.3,
        )
    )
fig = dict( data=data, layout=layout )
iplot( fig, validate=False)
plt.figure(figsize=(10,10))
corr = df.corr()
mask = np.zeros_like(corr, dtype=np.bool)

cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
heavy_freights = df[df.order_freight_value > df.order_freight_value.quantile(q=0.9)]

mapbox_access_token = 'pk.eyJ1IjoibGVlZG9oeXVuIiwiYSI6ImNqbjl1Y2hmcTB6dTQzcnBiNDZ2cXcwbGEifQ.hcPVtUhnyzXDXZbQQH0nMw'
data = [go.Scattermapbox(
    lon = heavy_freights['customer_longitude'],
    lat = heavy_freights['customer_latitude'],
    marker = dict(
        size = heavy_freights['order_items_qty']*3,
        color = 'blue'
    ))]

layout = dict(
        title = 'High Freight order Customer Locations',
        mapbox = dict(
            accesstoken = mapbox_access_token,
            center= dict(lat=-20,lon=-60),
            bearing=5,
            pitch=5,
            zoom=2.3,
        )
    )
fig = dict( data=data, layout=layout )
iplot( fig, validate=False)
### import olist public dataset which is including the data for customer_id

df3 = pd.read_csv("../input/olist_public_dataset_v2.csv")
df3.head()
### making DataFrame based on df3.customer_id.value_counts()
### first, making index list
### second, making counts value list
### zipping into dictionary and make new df 'purchase_df'

value = df3.customer_id.value_counts().index.tolist()
counts= df3.customer_id.value_counts().tolist()
purchase_df = pd.DataFrame({'customer_id':value,'purchase_counts':counts})
### purchase counts number df is concated into df3 as new column
### pd.merge is very effective

df3 = pd.merge(df3,purchase_df)
df3.head()
def normalize(x):
    return(x-x.mean())/(x.max()-x.min())
df3['normalized_opv']=normalize(df3.order_products_value)
df3['normalized_ofv']=normalize(df3.order_freight_value)
from sklearn.cluster import KMeans
feature = df3[['normalized_opv','normalized_ofv']]
model = KMeans(n_clusters=3,algorithm='auto')
model.fit(feature)
predict = pd.DataFrame(model.predict(feature))
predict.columns=['predict']
r = pd.concat([feature,predict],axis=1)
ks = range(1,10)
inertias = []


for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(feature)
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.title("Intertias Tilting Graph")
plt.xticks(ks)
plt.show()
model = KMeans(n_clusters=3,algorithm='auto')
model.fit(feature)
predict = pd.DataFrame(model.predict(feature))
predict.columns=['predict']

plt.figure(figsize=(8,7))
plt.scatter(r['normalized_opv'],r['normalized_ofv'],c=r['predict'],alpha=0.5)
plt.xlim(xmax=0.4)
plt.ylim(ymax=0.4)
plt.xlabel("Order Product Values",fontsize=15)
plt.ylabel("Order Freight Values",fontsize=15)
plt.title("Clustering from Product Value & Freight Value",fontsize=20)
centers = pd.DataFrame(model.cluster_centers_,columns=['normalized_opv','normalized_ofv'])
center_x = centers['normalized_opv']
center_y = centers['normalized_ofv']
plt.scatter(center_x,center_y,s=50,marker='D',c='r')
plt.show()