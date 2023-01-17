#การนำเข้าข้อมูล โดย import และประกาสค่าตัวแปรต่างๆ

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.core.interactiveshell import InteractiveShell

from sklearn.cluster import KMeans



from sklearn.manifold import TSNE

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import linkage, dendrogram

from sklearn.cluster import DBSCAN

from sklearn.cluster import AgglomerativeClustering

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import silhouette_score

InteractiveShell.ast_node_interactivity = "all"

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# import ข้อมูลเข้า notebook

filename='/kaggle/input/top-spotify-songs-from-20102019-by-year/top10s.csv'

df=pd.read_csv(filename,encoding='ISO-8859-1')

# แสดงข้อมูล 603 ข้อ จาก 0 - 602

df.head(603) 
df = df.rename(columns = {'Unnamed: 0': 'id'})

df.head()
#จากการทำการตรวจสอบถ้าข้อมูลขึ้น false จะแสดงได้ว่าข้อมูลถูกเคลียร์เรียบร้อยแล้วสามารถนำข้อมูลมาใช้ได้ต่อ

df.isnull().any()
#จำนวนข้อมูลของ columns 

df = df.drop_duplicates()

df.shape
#การ drop ข้อมูลทิ้ง ใน columns id

df = df.drop(['id'], axis=1)

df.head()
#ศิลปินที่ร้องเพลงที่คนนิยมในช่วง 10 ปี 

df['artist'].value_counts()
#เพลงของศิลปิน Justin Bieber ที่ติดอันดับภายในปี 2010 - 2019 ติด top ในปี 2015-2016

df[df.title == 'Company']
#แนวเพลงที่มีการเข้าฟังมากที่สุด  

df['top genre'].value_counts()
#จะได้ตารางที่เหลือแต่ข้อมูลที่เราจะนำไปเทรนโมเดล

yearless_df = df.drop(['year', 'title', 'top genre', 'artist'], axis=1)

yearless_df.drop_duplicates()
#การทำ clustering ด้วย KMeans

#k = กลุ่มที่เราต้องการจะจับ

#sse = ระยะห่างของ k หรือความห่างของแต่ละกลุ่ม

from sklearn.cluster import KMeans



sse = {}

for k in range(1, 11):

    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(yearless_df)

    yearless_df["clusters"] = kmeans.labels_

    #print(data["clusters"])

    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

plt.figure()

plt.plot(list(sse.keys()), list(sse.values()))

plt.xlabel("Number of cluster")

plt.ylabel("SSE")

plt.show()
#การทำ kmeans โดยแบ่งข้อมูลเป็น 10 กลุ่ม ตั้งแต่ 0 - 9

kmeans = KMeans(n_clusters=10, max_iter=1000).fit(yearless_df)

df["clusters"] = kmeans.labels_

df.groupby(by=['clusters']).mean()
df.head(603)
sns.pairplot(df, hue = 'clusters', diag_kind = 'kde',

             plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},

             size = 4)

 

sns.pairplot(df, vars=['bpm','nrgy','dnce','dB', 'live', 'dur', 'val','acous','spch','pop'],

            hue = 'clusters');