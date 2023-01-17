# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
onion_data=pd.read_csv('/kaggle/input/market-price-of-onion-2020/Onion Prices 2020.csv')
onion_data.info()
onion_data.head()
print("Number of Onion Markets:", len(onion_data['market'].unique()))
print("Number of Districts:", len(onion_data['district'].unique()))
print("Number of States:", len(onion_data['state'].unique()))
print("Number of Varieties:", len(onion_data['variety'].unique()))
min_price=list(onion_data.min_price)
max_price =list(onion_data.max_price)
model_price =list(onion_data.modal_price)
arr =[list(onion_data.state),list(onion_data.market), list(onion_data.variety)]
index =pd.MultiIndex.from_arrays(arr, names=('state','market','variety'))
onion_df = pd.DataFrame({'minimum_price':min_price,'maximum_price':max_price,'modal_price':model_price}, index=index)
onion_df.head()
### Price Vs Variety
onion_df2=onion_df.groupby(level=2,sort=False).mean().reset_index()
onion_df2.head()
print("national average modal price of onion:", round(onion_df2['modal_price'].mean()))
plt.figure(figsize=(30,5))
sns.set_context("notebook", font_scale=1)
plt.plot(onion_df2['variety'],onion_df2['modal_price'],color='green', marker='o', linestyle='dashed',linewidth=2, markersize=12)
plt.plot(onion_df2['variety'],onion_df2['minimum_price'])
plt.plot(onion_df2['variety'],onion_df2['maximum_price'])
plt.title("Onion variety VS Price")
plt.xlabel("Onion Variety")
plt.ylabel("Average Market Price")
onion_df2[onion_df2['modal_price']>2327]
onion_df2[onion_df2['modal_price']<2327].sort_values('modal_price',ascending=True)[0:5]
onion_df3=onion_df.groupby(level=0,sort=False).mean().reset_index()

plt.figure(figsize=(30,5))
sns.set_context("notebook", font_scale=1)
plt.plot(onion_df3['state'],onion_df3['modal_price'],color='green', marker='o', linestyle='dashed',linewidth=2, markersize=12)
plt.plot(onion_df3['state'],onion_df3['minimum_price'])
plt.plot(onion_df3['state'],onion_df3['maximum_price'])
plt.title("State VS Price")
plt.xlabel("State")
plt.ylabel("Average Market Price")
onion_df3.sort_values('modal_price',ascending=False)[0:5]
onion_df3.sort_values('modal_price')[0:5]
print("Expensive onion producing states:",onion_data[(onion_data.variety=='Bellary')|(onion_data.variety=='Small')|(onion_data.variety=='Dry F.A.Q.')|(onion_data.variety=='Big')|(onion_data.variety=='Bombay (U.P.)')|(onion_data.variety=='Hybrid')].state.unique())
print("Low cost onion producing states:",onion_data[(onion_data.variety=='Puna')|(onion_data.variety=='2nd Sort')|(onion_data.variety=='White')|(onion_data.variety=='Telagi')|(onion_data.variety=='Bangalore-Samall')].state.unique())
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_arr3 = scaler.fit_transform(onion_data[['modal_price','min_price','max_price']])

clusters = KMeans(5,random_state=42)# we've got k=5 using elbow plot
kmeans_model= clusters.fit(scaled_arr3)
onion_data['cluster_id']= kmeans_model.labels_
onion_data1 = onion_data.sort_values(['modal_price','cluster_id'])
onion_data1.head()
onion_clusters = onion_data1.groupby('cluster_id')[['modal_price','min_price','max_price']].agg(['mean','std']).reset_index()
onion_clusters
plt.figure(figsize=(20,10))
sns.set_context("notebook", font_scale=1)
sns.barplot(onion_data1['cluster_id'],onion_data1['modal_price'])