import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

data = pd.read_csv('../input/dataset/dataset.csv')

Data = data
data.head()
#finding categorical and continuous features

for i in data.columns:

    print(i,':',data[i].unique())
#replacing all values that have multiple representations

data = data.replace('?',np.nan)

numeric = ['Monthly Period','Credit1', 'InstallmentRate','Tenancy Period','Age', '#Credits', '#Authorities','InstallmentCredit', 'Yearly Period']

data['Plotsize'] = data['Plotsize'].replace('M.E.','ME')

data['Plotsize'] = data['Plotsize'].replace('me', 'ME')

data['Plotsize'] = data['Plotsize'].replace('sm', 'SM')

data['Plotsize'] = data['Plotsize'].replace('la', 'LA')

data['Account2'] = data['Account2'].replace('Sacc4','sacc4')

data['Sponsors'] = data['Sponsors'].replace('g1','G1')

data = data.drop(['id'],1)
#converting all object type attributes that are numeric to integers and floats

for i in numeric:

    data[i] = pd.to_numeric(data[i])

data.info()
categorical = []

categorical
#filling in missing values present in the data

null_columns = data.columns[data.isnull().any()]

print('Features with null values are :', null_columns)



print("Before removing categorical features we have")

print('Categorical :', len(categorical))



for i in null_columns:

    if i in numeric:

        data[i] = data[i].fillna(data[i].mean())

    elif i in categorical:

        categorical.remove(i)

        

print(data.columns[data.isnull().any()])

print("After removing categorical features having null values")

print('Categorical :', categorical)
#choosing which columns to drop

drop = []

for i in data.columns:

    if i not in numeric and i not in categorical:

        drop.append(i)

#We are not dropping class yet because we will see the correlation between it and other attributes 

drop.remove('Class')



drop
#dropping the coloumns we chose

new_data = data.drop(columns=drop)
#plotting the correlation matrix to visualise correlation between the class and features

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

#corelation heat map

df = new_data

corr = df.corr(method="kendall")



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(25, 25))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0.5,

            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)



plt.show()
#After judging from correlation matrix we can drop columns which are not highly correlated wth Class label

#drop.append('Monthly Period')

drop = []

drop.append('Employment Period')

drop.append('Motive')

drop.append('History')

drop.append('Post')

drop.append('Phone')

drop.append('Account1')

drop.append('Account2')

drop.append('Gender&Type')

drop.append('Plotsize')

drop.append('Class')

drop.append('Monthly Period')

drop.append('Credit1')

data = data.drop(drop,1)

print('Columns we dropped are :', drop)
#counting unique values of attributes

for i in categorical:

    print(i,':',len(data[i].unique()))
#encoding categorical variables

null_columns=data.columns[data.isnull().any()]

for i in null_columns:

    data[i]=data[i].fillna(data[i].mean())

data.columns[data.isnull().any()]

new_data = data

new_data['Expatriate']=new_data['Expatriate'].replace({True: 1, False: 0})

for i in new_data.columns:

    new_data[i].replace(np.nan,None)

categorical=['Housing','Sponsors','Plan']

data_encoded = pd.get_dummies(new_data, columns = categorical)

#data_encoded.head()

data_encoded.info()

#min max scaling of variables

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(data_encoded)

dataN = pd.DataFrame(np_scaled)

dataN.head()
#plotting the elbow method to find optimum number of clusters

from sklearn.cluster import KMeans



wcss = []

for i in range(2, 50):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(dataN)

    wcss.append(kmean.inertia_)

    

plt.plot(range(2,50),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
#pca transformation to plot 2d clusters

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(dataN)

T = pca.transform(dataN)
#plotting of the clusters

plt.figure(figsize=(16, 8))

preds1 = []

for i in range(2, 11):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(dataN)

    pred = kmean.predict(dataN)

    preds1.append(pred)

    

    plt.subplot(2, 5, i - 1)

    plt.title(str(i)+" clusters")

    plt.scatter(T[:, 0], T[:, 1], c=pred)

    

    centroids = kmean.cluster_centers_

    centroids = pca.transform(centroids)

    plt.plot(centroids[:, 0], centroids[:, 1], 'b+', markersize=30, color = 'brown', markeredgewidth = 3)
#plotting the clusters after running KMeans

colors = ['red','green','blue','yellow','purple','pink']

plt.figure(figsize=(16, 8))



kmean = KMeans(n_clusters = 5, random_state = 42)

kmean.fit(dataN)

pred = kmean.predict(dataN)

pred_pd = pd.DataFrame(pred)

arr = pred_pd[0].unique()



for i in arr:

    meanx = 0

    meany = 0

    count = 0

    

    c0 = 0

    c1 = 0

    c2 = 0

    

    for j in range(len(pred)):

        if i == pred[j]:

            count+=1

            meanx+=T[j,0]

            meany+=T[j,1]

            plt.scatter(T[j, 0], T[j, 1], c=colors[i])

        

            if j<=175:

                if Data['Class'][j]==0:

                    c0 += 1

                if Data['Class'][j]==1:

                    c1 += 1

                if Data['Class'][j]==2:

                    c2 += 1

                

                                                                                                                                          

    if max(c0, c1, c2) == 0:

        c_final = 0

    elif max(c0, c1, c2) == 1:

        c_final = 1

    else: c_final = 2

                

            

    meanx = meanx/count

    meany = meany/count

    plt.annotate(i,(meanx, meany),size=30, weight='bold', color='black', backgroundcolor=colors[i])

    plt.annotate(c_final,(meanx+0.1, meany+0.1),size=30, weight='bold', color='gray', backgroundcolor=colors[i])
#assigning predicted clusters to actual class values

ids=Data['id'].values[175:1031]

final_classes=[]

for i in range(175,1031):

    if pred[i] == 4 or pred[i] == 0:

        final_classes.append(0)

    if pred[i] == 1 or pred[i] == 3:

        final_classes.append(2)

    if pred[i] == 2:

        final_classes.append(1)



final_results=pd.concat([pd.DataFrame(ids),pd.DataFrame(final_classes)], axis=1).reindex()

final_results.columns=['id','Class']

final_results.head()
#saving as csv file for submission

final_results.to_csv('final4.csv', index=False)


from sklearn.neighbors import NearestNeighbors



ns = 74

nbrs = NearestNeighbors(n_neighbors = ns).fit(dataN)

distances, indices = nbrs.kneighbors(dataN)



kdist = []



for i in distances:

    avg = 0.0

    for j in i:

        avg += j

    avg = avg/(ns-1)

    kdist.append(avg)



kdist = sorted(kdist)

plt.plot(indices[:,0], kdist)
#running DBSCAN

from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=2, min_samples=10)

predD = dbscan.fit_predict(dataN)

plt.scatter(T[:, 0], T[:, 1], c=predD)