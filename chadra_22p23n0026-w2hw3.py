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
!pip install kora -q
#libraly importing

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize

df = pd.read_csv('../input/ab-nyc-2019/AB_NYC_2019.csv')
    
df.shape
df.head()
df[df['reviews_per_month'].isnull()].head()
df.info()
df.isnull().sum()
df = df.drop(['id','name','host_name','last_review','calculated_host_listings_count','availability_365'], axis=1)
df = df.fillna(0)
df.isnull().sum()
df.head()
lb_en_1 = preprocessing.LabelEncoder()
lb_en_1.fit(df['neighbourhood_group'])
#lb_en_1.classes_
df['neighbourhood_group'] = list(lb_en_1.transform(df['neighbourhood_group']))
#lb_en_1.inverse_transform(list_encode) #

lb_en_2 = preprocessing.LabelEncoder()
lb_en_2.fit(df['neighbourhood'])
#lb_en_2.classes_
df['neighbourhood'] = list(lb_en_2.transform(df['neighbourhood']))
#lb_en_2.inverse_transform(list_encode) #
lb_en_3 = preprocessing.LabelEncoder()
lb_en_3.fit(df['room_type'])
#lb_en_3.classes_
df['room_type'] = list(lb_en_3.transform(df['room_type']))
#lb_en_3.inverse_transform(list_encode) #
lb_en_3.classes_
df.head()
corr = df.corr(method="pearson")
plt.figure(figsize=(20,10))
sns.heatmap(corr, annot=True)
#ทดสอบ plot ชนิดห้องเช่า เทียบกับตำแน่ง
plt.style.use('ggplot')
plt.figure(figsize=(10,10))
sns.scatterplot(x='longitude', y='latitude', hue='room_type',s=20, data=df,palette="deep")
#กำหนดแค่ 2000 records เนื่องจากไม่สามารถ process ทั้งหมดบน server ได้(server ตัดการทำงานเนื่องจากใช้ ram มากเกินไป)
data_range = 2000  
#ทำการ rescale data
data_scaled = normalize(df.iloc[0:data_range,:])
data_scaled = pd.DataFrame(data_scaled, columns=df.columns)
data_scaled.head()
#Hierarchical Clustering และ plot
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
#จะเห็นได้ว่าข้อมูลถูกแยกป็น 2 clusters ที่ threshold 0.15 ซึ้งมีเส้นแนวตั้งสูงสุด
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
plt.axhline(y=0.15, color='r', linestyle='--')
#ใส่ label ให้กับ 2 cluster ด้านบน
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
cluster.fit_predict(data_scaled)
#plot เพื่อหา insight ของข้อมูล
df_for_ploting = df.iloc[0:data_range,:]
plt.figure(figsize=(10, 7))  
plt.scatter(df_for_ploting['longitude'], df_for_ploting['latitude'], c=cluster.labels_,) 