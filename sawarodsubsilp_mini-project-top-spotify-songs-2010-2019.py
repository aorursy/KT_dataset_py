
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
# การนำข้อมูลเข้า โดยใช้ไฟล์ top10s.csv
filename='/kaggle/input/top-spotify-songs-from-20102019-by-year/top10s.csv'
df=pd.read_csv(filename,encoding='ISO-8859-1')
# การแสดงข้อมูลทั้งหมด 603 ข้อมูล ที่เป็นเพลงฮิตตลอด 10 ปี แต่เป็นการขึ้นต้นอันดับด้วย 0 จนถึง 602
df.head(603) 

#เปลี่ยนชื่อ columns จาก 'Unnamed: 0'เป็น 'id'
df = df.rename(columns = {'Unnamed: 0': 'id'})
df.head()
#จากการทำการตรวจสอบถ้าข้อมูลขึ้น false จะแสดงได้ว่าข้อมูลถูกเคลียร์เรียบร้อยแล้วสามารถนำข้อมูลมาใช้ได้ต่อ
df.isnull().any()
#จำนวนข้อมูลของ columns ก่อน drop ข้อมูลทิ้ง
df = df.drop_duplicates()
df.shape
#การ drop ข้อมูลทิ้ง ใน columns id
df = df.drop(['id'], axis=1)
df.head()
#การนับจำนวน เพลงที่คนนิยมในช่วง 10 ปี ว่าแต่ละแนวเพลงจำนวนเท่าไหร่
df['top genre'].value_counts().head()

#การนับจำนวน ศิลปินที่ร้องเพลงที่คนนิยมในช่วง 10 ปี ว่าแต่คนร้องเพลงจำนวนเท่าไหร่
df['artist'].value_counts().head()
#การนับจำนวน ชื่อเพลงที่คนนิยมในช่วง 10 ปี ว่ามีเพลงที่ซ้ำกันกี่เพลง
df['title'].value_counts().head()
#แสดงค่าว่าตอนนี้เพลง Company และ deduplicate rows ที่ค่าซ้ำกัน 
df[df.title == 'Company']
#นับค่าแนวเพลงทั้งหมด ว่าแต่ละแนวเพลงมีกี่เพลง
df['top genre'].value_counts()
#ทำการ drop หรือทิ้ง  columns  year, title , pop , artist , top genre จาก data df และแทนค่า ด้วย data = yearless_df
#จะได้ตารางที่เหลือแต่ข้อมูลที่เราจะนำไปเทรนโมเดล
yearless_df = df.drop(['year', 'title', 'pop','artist' ,'top genre'], axis=1)
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

#การทำ kmeans โดยแบ่งข้อมูลเป็น 10 กลุ่ม กลุ่มที่ 0-9
kmeans = KMeans(n_clusters=10, max_iter=1000).fit(yearless_df)
df["clusters"] = kmeans.labels_
df.groupby(by=['clusters']).mean()
#การแสดงข้อมูลทั้งหมด และบ่งบอกว่าแต่ละเพลง จัดอยู่ในกลุ่มที่เท่าไหร่ ใน clusters
df.head(603)
#การแสดงข้อมูลโดย pairplot เพื่อแสดงข้อมูลทั้งหมด ว่าสีนี้อยู่กลุ่มอะไร และกราฟมีการพล็อตจุดเป็นสีช่วงไหน
sns.pairplot(df, hue="clusters")
plt.figure(figsize=(10,5))
sns.countplot(df['clusters'])
plt.show()
sns.catplot(x="clusters", y="bpm", data=df)
sns.catplot(x = "clusters", y  ="nrgy" , data=df)

sns.catplot(x="clusters",y="dnce",data=df)
sns.catplot(x="clusters", y="dB", data=df)
sns.catplot(x="clusters", y="live", data=df)
sns.catplot(x="clusters", y="val", data=df)
sns.catplot(x="clusters", y="dur", data=df)
sns.catplot(x="clusters", y="acous", data=df)
sns.catplot(x="clusters", y="spch", data=df)