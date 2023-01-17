import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
!unzip /kaggle/input/expedia-personalized-sort/data.zip 
### ZipFile can't read this proprietary format 
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
print(train.shape)
train=train.dropna(axis=1,how="any")
train
test
train.isnull().sum()
train.info()
import seaborn as sns
df = train.corr()
print(df)
train[["promotion_flag","position"]].groupby("promotion_flag").sum().position.plot(kind="pie",shadow=True,autopct="%1.1f%%",radius=1.2,startangle=120)
plt.title("Promotion Graph")
plt.show()
train.columns
train=train.drop(["srch_id","date_time","site_id","visitor_location_country_id","prop_country_id","prop_id","srch_destination_id"],axis=1)
test=test.dropna(axis=1,how="any")
test.columns
test=test.drop(["srch_id","date_time","site_id","visitor_location_country_id","prop_country_id","prop_id","srch_destination_id"],axis=1)
test.columns
train=train[['prop_starrating', 'prop_brand_bool', 'prop_location_score1',
       'prop_log_historical_price', 'price_usd', 'promotion_flag',
       'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',
       'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool',
       'random_bool']]

from sklearn.cluster import KMeans
data = train
n_cluster = range(1, 20)

kmeans = [KMeans(n_clusters = i).fit(data) for i in n_cluster]
scores = [kmeans[i].score(data) for i in range(len(kmeans))]

fig, ax = plt.subplots(figsize = (16, 8))
ax.plot(n_cluster, scores, color = 'orange')

plt.xlabel('clusters num')
plt.ylabel('score')
plt.title('Elbow curve for K-Means')
plt.show();

from sklearn.cluster import KMeans
km = KMeans(n_clusters = 8)
kmean=km.fit(train)
y_kmeans = km.predict(train)
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
fig = plt.figure(1, figsize = (7, 7))

ax = Axes3D(fig, rect = [0, 0, 0.95, 1], 
            elev = 48, azim = 134)

ax.scatter(train.iloc[:, 4:5],
           train.iloc[:, 7:8], 
           train.iloc[:, 11:12],
           c = km.labels_.astype(np.float), edgecolor = 'm')

ax.set_xlabel('USD')
ax.set_ylabel('srch_booking_window')
ax.set_zlabel('srch_saturday_night_bool')

plt.title('K Means', fontsize = 10)
train.columns
km.predict(test)
