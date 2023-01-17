!pip install hdbscan


import matplotlib.pyplot as plt





import pandas as pd

import numpy as np



from sklearn.cluster import KMeans, DBSCAN

from sklearn.metrics import silhouette_score

from sklearn.neighbors import KNeighborsClassifier



import hdbscan

import folium

import re





cols = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',

        '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', 

        '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', 

        '#000075', '#808080']*10
data=pd.read_csv("../input/taxi_data.csv")

data.head()
data.duplicated(subset=['LON','LAT']).values.any()
data.isnull().values.any()
print(f'Before dropping NaNs and dupes\t:\tdf.shape = {data.shape}')

data.dropna(inplace=True)

data.drop_duplicates(subset=['LON','LAT'],keep='first',inplace=True)

print(f'After dropping NaNs and dupes\t:\tdf.shape = {data.shape}')
X=np.array(data[['LON','LAT']],dtype='float64')

plt.scatter(X[:,0],X[:,1],alpha=0.2,s=50)

plt.grid(True)
plt_map=folium.Map(location=[data.LAT.mean(),data.LON.mean()],zoom_start=10,

             tiles='OpenStreetMap')

for _,row in data.iterrows():

    folium.CircleMarker(

        location=[row.LAT,row.LON],

        radius=5,

        popup=re.sub(r'[^a-zA-Z ]+','',row.NAME),

        color='blue',

        fill=True

    ).add_to(plt_map)

plt_map
k_range=range(5,150,5)

kmeans_per_k=[]

for k in k_range:

    kmeans=KMeans(n_clusters=k,random_state=42).fit(X)

    kmeans_per_k.append(kmeans)
silh_scores=[silhouette_score(X,model.labels_) for model in kmeans_per_k]

best_index = np.argmax(silh_scores)

best_k = k_range[best_index]

best_score = silh_scores[best_index]

print("best k value:",best_k)

print("silhouette score:",best_score)



plt.figure(figsize=(8, 3))

plt.grid(True)

plt.plot(k_range, silh_scores, "bo-")

plt.xlabel("k", fontsize=14)

plt.ylabel("Silhouette score", fontsize=14)

plt.plot(best_k, best_score, "rs")

plt.show()

inertias = [model.inertia_ for model in kmeans_per_k]

best_inertia = inertias[best_index]



plt.figure(figsize=(8, 3.5))

plt.grid(True)

plt.plot(k_range, inertias, "bo-")

plt.xlabel("$k$", fontsize=14)

plt.ylabel("Inertia", fontsize=14)

plt.plot(best_k, best_inertia, "rs")

plt.show()

k=70

model=KMeans(n_clusters=k,random_state=17).fit(X)

pred=model.predict(X)

data[f'CLUSTER_kmeans']=pred
def create_map(data,cluster_col):

    m = folium.Map(location=[data.LAT.mean(), data.LON.mean()], zoom_start=9, tiles='openstreetmap')



    for _, row in data.iterrows():



        # get a colour

        if row[cluster_col]==-1:

            cluster_colour='black'

        else:

            cluster_colour = cols[row[cluster_col]]



        folium.CircleMarker(

            location=[row.LAT,row.LON],

            radius=5,

            popup= row[cluster_col],

            color=cluster_colour,

            fill=True,

            fill_color=cluster_colour

        ).add_to(m)

    return m



plt_map=create_map(data,'CLUSTER_kmeans')   

print(f'Silhouette Score: {silhouette_score(X, pred)}')

plt_map.save('kmeans_map.html')

plt_map


dummy = np.array([-1, -1, -1, 2, 3, 4, 5, -1])

new=np.array([(counter+2)*x if x==-1 else x for counter,x in enumerate(dummy)])

new
model=DBSCAN(eps=0.01,min_samples=5).fit(X)

class_predictions=model.labels_

data['Clusters_dbscan']=class_predictions

data.head()


print(f'Number of clusters found: {len(np.unique(class_predictions))}')

print(f'Number of outliers found: {len(class_predictions[class_predictions==-1])}')



print(f'Silhouette ignoring outliers: {silhouette_score(X[class_predictions!=-1], class_predictions[class_predictions!=-1])}')



no_outliers = 0

no_outliers = np.array([(counter+2)*x if x==-1 else x for counter, x in enumerate(class_predictions)])

print(f'Silhouette outliers as singletons: {silhouette_score(X, no_outliers)}')

plt_map=create_map(data,'Clusters_dbscan')

plt_map.save('DBSCAN_map.html')

plt_map
model=hdbscan.HDBSCAN(min_cluster_size=5,min_samples=2,cluster_selection_epsilon=0.01)

class_predictions=model.fit_predict(X)

data['CLUSTER_hdbscan']=class_predictions

data.head()


print(f'Number of clusters found: {len(np.unique(class_predictions))-1}')

print(f'Number of outliers found: {len(class_predictions[class_predictions==-1])}')



print(f'Silhouette ignoring outliers: {silhouette_score(X[class_predictions!=-1], class_predictions[class_predictions!=-1])}')



no_outliers = np.array([(counter+2)*x if x==-1 else x for counter, x in enumerate(class_predictions)])

print(f'Silhouette outliers as singletons: {silhouette_score(X, no_outliers)}')



m=create_map(data,'CLUSTER_hdbscan')

plt_map.save("HDBSCAN_map.html")

plt_map
classifier=KNeighborsClassifier(n_neighbors=3)
data_train=data[data.CLUSTER_hdbscan!=-1]

data_predict=data[data.CLUSTER_hdbscan==-1]
X_train=np.array(data_train[['LON','LAT']],dtype='float64')

y_train=np.array(data_train['CLUSTER_hdbscan'])



X_predict=np.array(data_predict[['LON',"LAT"]],dtype='float64')
classifier.fit(X_train,y_train)

pred=classifier.predict(X_predict)

pred
data['CLUSTER_hybrid']=data['CLUSTER_hdbscan']

data.loc[data.CLUSTER_hdbscan==-1,'CLUSTER_hybrid']=pred

data.head()
class_predictions=data.CLUSTER_hybrid

print(f'Number of clusters found: {len(np.unique(class_predictions))}')

print(f'Silhouette: {silhouette_score(X, class_predictions)}')

plt_map=create_map(data,'CLUSTER_hybrid')

plt_map.save('hybrid_map.html')

plt_map
data['CLUSTER_hybrid'].value_counts().plot.hist(bins=70,alpha=0.5,label='hybrid')

data['CLUSTER_kmeans'].value_counts().plot.hist(bins=70,alpha=0.5,label='kmeans')



plt.legend()



plt.grid(True)

plt.title('Comparing Hybrid and K-Means Approaches')

plt.xlabel('cluster sizes')