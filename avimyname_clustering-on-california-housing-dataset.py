import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import folium
import re

cols = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
        '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', 
        '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', 
        '#000075', '#808080']*10

df = pd.read_csv('/kaggle/input/housing/housing.csv')
df.head()
df.duplicated(subset=['longitude', 'latitude']).values.any()
df.isna().values.any()
print(f'Before dropping NaNs and dupes\t:\tdf.shape = {df.shape}')
df.dropna(inplace=True)
df.drop_duplicates(subset=['longitude', 'latitude'], keep='first', inplace=True)
print(f'After dropping NaNs and dupes\t:\tdf.shape = {df.shape}')

df.isna().values.any()
df['index'] = range(len(df))
df.set_index('index')
df.head(5)
#Plot all the latitudes and longitudes
X = np.array(df[['longitude', 'latitude']], dtype='float64')
plt.scatter(X[:,0], X[:,1], alpha=0.3, s=90)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
def converter(cluster):
    if cluster=='NEAR BAY':
        return 1
    else:
        return 0
df['bay'] = df['ocean_proximity'].apply(converter)
df.head(3)
samples = df
samples = samples.drop(columns=['longitude','latitude','ocean_proximity','population'])
samples.head(3)
pca = PCA()
scaler = StandardScaler()
df_features = StandardScaler().fit_transform(samples)
pca.fit(df_features)
features = samples.columns
print(pca.n_components)
plt.bar(features,pca.explained_variance_)
plt.xticks(features,rotation=90)
plt.ylabel("variance")
plt.xlabel("features")
plt.show()
samples_new = samples[['housing_median_age' , 'total_rooms','total_bedrooms','population']]
samples_new.head(3)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score,v_measure_score
X
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
print(kmeans.cluster_centers_)
print(kmeans.labels_)
preds = kmeans.predict(X)
results = df[['index','longitude', 'latitude','housing_median_age','median_house_value','population','households']]
results['cluster_region'] = preds
results.head(10)
results.cluster_region.value_counts()
vals = results.values
vals
l = vals[:,7]
colors = [cols[int(i)] for i in l]
#print(colors)
#l = vals[:,2]
#for i in l:
plt.scatter(vals[:,1], vals[:,2], alpha=0.3, color=colors)
results_A = results[results['cluster_region']==1.0]
results_B = results[results['cluster_region']==0.0]

vals_A = results_A.values
vals_B = results_B.values

plt.scatter(vals_A[:,1], vals_A[:,2], alpha=0.3, color=cols[1])
results_A.head(4)

pop_data = np.array(results_A[['population','households']],dtype='float64')
pop_data

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
pop_data_scaled = scaler.fit_transform(pop_data)
pop_data_scaled

kmeans_pop = KMeans(n_clusters=2)
kmeans_pop.fit(pop_data_scaled)
preds = kmeans_pop.predict(pop_data_scaled)
results_A['cluster_population'] = preds
results_A.head(4)
results_A.cluster_population.value_counts()

vals = results_A.values
l = vals[:,8]
colors = [cols[int(i)+3] for i in l]
#print(colors)
plt.scatter(vals_A[:,1], vals_A[:,2], alpha=0.3, color=colors)

house_data = np.array(results_A[['median_house_value']],dtype='float64')
house_data
scaler = StandardScaler()
house_data_scaled = scaler.fit_transform(house_data)
house_data_scaled
kmeans_house = KMeans(n_clusters=2)
kmeans_house.fit(house_data_scaled)
preds = kmeans_house.predict(house_data_scaled)
results_A['cluster_house_price'] = preds
results_A.head(5)
results_A.cluster_house_price.value_counts()

vals = results_A.values
l = vals[:,9]
colors = [cols[int(i)+6] for i in l]
#print(colors)
plt.scatter(vals_A[:,1], vals_A[:,2], alpha=0.3, color=colors)
plt.scatter(vals_B[:,1], vals_B[:,2], alpha=0.3, color=cols[0])
results_B.head(4)

pop_data = np.array(results_B[['population','households']],dtype='float64')
pop_data
scaler = StandardScaler()
pop_data_scaled = scaler.fit_transform(pop_data)
pop_data_scaled

kmeans_pop = KMeans(n_clusters=2)
kmeans_pop.fit(pop_data_scaled)
preds = kmeans_pop.predict(pop_data_scaled)
results_B['cluster_population'] = preds
results_B.head(4)
results_B.cluster_population.value_counts()

vals = results_B.values
l = vals[:,8]
colors = [cols[int(i)+9] for i in l]
#print(colors)
plt.scatter(vals[:,1], vals[:,2], alpha=0.3, color=colors)

house_data = np.array(results_B[['median_house_value']],dtype='float64')
house_data
scaler = StandardScaler()
house_data_scaled = scaler.fit_transform(house_data)
house_data_scaled
kmeans_house = KMeans(n_clusters=2)
kmeans_house.fit(house_data_scaled)
preds = kmeans_house.predict(house_data_scaled)
results_B['cluster_house_price'] = preds
results_B.head(5)
results_B.cluster_house_price.value_counts()
vals = results_B.values
l = vals[:,9]
colors = [cols[int(i)+10] for i in l]
#print(colors)
plt.scatter(vals[:,1], vals[:,2], alpha=0.3, color=colors)


results_A.head(5)
results_B.head(5)

pop_data = np.array(results_B[['population','households']],dtype='float64')
#pop_data
scaler = StandardScaler()
pop_data_scaled = scaler.fit_transform(pop_data)
#pop_data_scaled
km_scores= []
km_silhouette = []
vmeasure_score =[]
db_score = []
for i in range(2,20):
    km = KMeans(n_clusters=i, random_state=0).fit(pop_data_scaled)
    preds = km.predict(pop_data_scaled)
    
    print("Score for number of cluster(s) {}: {}".format(i,km.score(pop_data_scaled)))
    km_scores.append(-km.score(pop_data_scaled))
    
    silhouette = silhouette_score(pop_data_scaled,preds)
    km_silhouette.append(silhouette)
    print("Silhouette score for number of cluster(s) {}: {}".format(i,silhouette))
    
    db = davies_bouldin_score(pop_data_scaled,preds)
    db_score.append(db)
    print("Davies Bouldin score for number of cluster(s) {}: {}".format(i,db))
    
    print('\n')
    
#Plotting the Elbow Graph
plt.title("The elbow method for determining number of clusters\n")
plt.scatter(x=[i for i in range(2,20)],y=km_scores,s=150,edgecolor='k')
plt.grid(True)
plt.xlabel("Number of clusters")
plt.ylabel("K-means score")
plt.xticks([i for i in range(2,12)])
plt.yticks(fontsize=15)
plt.show()

print("Hello")
data = results_A
m = folium.Map(location=[data.latitude.mean(), data.longitude.mean()], zoom_start=9, 
               tiles='Stamen Toner')

for _, row in data.iterrows():
    folium.CircleMarker(
        location=[row.latitude, row.longitude],
        radius=5,
        popup={'Region':row.cluster_region,'Population':row.population,'Price':row.median_house_value},
        #popup=re.sub(r'[^a-zA-Z0-9 ]+', '', row.cluster_name),
        color='#1787FE',
        fill=True,
        fill_colour='#1787FE'
    ).add_to(m)
m

def create_map(m,df, cluster_column):
    
    for _, row in df.iterrows():

        if row.cluster_region == -1:
            cluster_colour = '#000000'
        else:
            cluster_colour = cols[int(row.cluster_region)]
            
        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=5,
            popup={'Region':row.cluster_region,'Population_Cluster':row.cluster_population,'Population':row.population,'Price_Cluster':row.cluster_house_price,'Price':row.median_house_value},
        color=cluster_colour,
            fill=True,
            fill_color=cluster_colour
        ).add_to(m)
        
        
    return m


m1 = folium.Map(location=[df.latitude.mean(), df.longitude.mean()], zoom_start=9, tiles='Stamen Toner')

m1 = create_map(m1, results_A,'CLUSTER_kmeans')
m1 = create_map(m1, results_B,'CLUSTER_kmeans')

#print(f'K={k}')
#print(f'Silhouette Score: {silhouette_score(X, class_predictions)}')

#m1.save('kmeans_70.html')
m1

def add_points(m,df, cluster_column):
    
    for _, row in df.iterrows():

        if row.cluster_house_price == -1:
            cluster_colour = '#000000'
        elif row.cluster_house_price == 0.0:
            cluster_colour = '#CCCC00'
            
            folium.CircleMarker(
                location=[row.latitude, row.longitude],
                radius=5,
                popup={'Region':row.cluster_region,'Population_Cluster':row.cluster_population,'Population':row.population,'Price_Cluster':row.cluster_house_price,'Price':row.median_house_value},
                color=cluster_colour,
                fill=True,
                fill_color=cluster_colour
            ).add_to(m)
        
        
    return m


#m1 = add_points(m1, results_A,'CLUSTER_kmeans')
m1 = add_points(m1, results_B,'CLUSTER_kmeans')

m1
m1.save('output.html')

