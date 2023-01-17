import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data_orig = pd.read_csv("../input/dataset.csv", sep=',')

data = data_orig.copy()

numericals=['Monthly Period','Credit1','InstallmentRate','Tenancy Period','Age','#Credits','#Authorities']

floats=['InstallmentCredit','Yearly Period']
data.info()
dnn=data.loc[np.sum(data == '?' , axis = 1) == 0, :] #Removing Tuples with missing values
#Converting string to int/float for numerical values

for col in numericals:

    dnn[col]=dnn[col].astype(np.int64)

for col in floats:

    dnn[col]=dnn[col].astype(np.float64)
#Correcting the spelling errors

dnn.loc[dnn['Plotsize']=='M.E.', 'Plotsize'] = 'ME'

dnn.loc[dnn['Plotsize']=='me', 'Plotsize'] = 'ME'

dnn.loc[dnn['Plotsize']=='la', 'Plotsize'] = 'LA'

dnn.loc[dnn['Plotsize']=='sm', 'Plotsize'] = 'SM'

dnn.loc[dnn['Sponsors']=='g1', 'Sponsors'] = 'G1'

dnn.loc[dnn['Account2']=='Sacc4', 'Account2'] = 'sacc4'
dnn.info()
dnn.groupby('Housing')['Tenancy Period'].mean()
#Merginng Housing and Tenancy Period due to correlation

dnn['house_ten'] = dnn.Housing

dnn.house_ten.replace(dnn.groupby('Housing')['Tenancy Period'].mean(), inplace=True)
dnn.groupby('Account1')['Monthly Period'].mean()
#Merging Account1 and Monthly Period due to correlation

dnn['acc1_mp'] = dnn.Account1

dnn.acc1_mp.replace(dnn.groupby('Account1')['Monthly Period'].mean(), inplace=True)
dnn.groupby('Gender&Type')['Age'].mean()
#Merging Gender&Type with Age due to correlation

dnn['gta'] = dnn['Gender&Type']

dnn.gta.replace(dnn.groupby('Gender&Type')['Age'].mean(), inplace=True)
dnn.groupby('Plotsize')['Credit1'].mean()
#Merginng Plotsize and Credit1 due to correlation

dnn['plc'] = dnn['Plotsize']

dnn.plc.replace(dnn.groupby('Plotsize')['Credit1'].mean(), inplace=True)
dnn.groupby('Sponsors')['#Authorities'].mean()
#Merginng Sponsors and #Authorities due to correlation

dnn['spon_auth'] = dnn['Sponsors']

dnn.spon_auth.replace(dnn.groupby('Sponsors')['#Authorities'].mean(), inplace=True)
dnn.columns
dataOH= pd.get_dummies(dnn, columns=["Gender&Type", 'Post','Phone','History','Account2','Employment Period','Plan'])

dataOH.head()
dataOH=dataOH.drop(['Gender&Type_F1','Gender&Type_M1','Phone_no'],1)
dataOH.info()
import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))

corr = dataOH.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax, annot = True);
dataOH=dataOH.drop(['id'],1) #Dropping ID
#Dropping Columns with high covariance

cols=['Tenancy Period','Monthly Period','Account1','Motive','Credit1','InstallmentCredit','Sponsors','Plotsize','Age','Housing','#Authorities']
preDf=dataOH.drop(cols,1)
preDf=preDf.drop(['Class'],1)
preDf['Expatriate']=preDf['Expatriate'].astype(np.int64)
num_col=['Yearly Period','house_ten','acc1_mp','gta','plc','spon_auth']
from sklearn import preprocessing

#Performing Min_Max Normalization

min_max_scaler = preprocessing.MinMaxScaler()

preDf[['InstallmentRate','#Credits']] = min_max_scaler.fit_transform(preDf[['InstallmentRate','#Credits']])



#Performing Z-score Normalization

std_scaler = preprocessing.StandardScaler()

preDf[num_col] = std_scaler.fit_transform(preDf[num_col])

preDf.head()
finalDF=preDf.copy() #To be used during final Clustering
from sklearn.decomposition import PCA

pca1 = PCA(n_components=2)

pca1.fit(preDf)

T1 = pca1.transform(preDf)

preDf.info()
from sklearn.cluster import KMeans



wcss = []

for i in range(2, 20):

    kmean = KMeans(n_clusters = i, random_state = 47)

    kmean.fit(preDf)

    wcss.append(kmean.inertia_)

    

plt.plot(range(2,20),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
plt.figure(figsize=(16, 8))

preds1 = []

for i in range(2, 11):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(preDf)

    pred = kmean.predict(preDf)

    preds1.append(pred)

    

    plt.subplot(2, 5, i - 1)

    plt.title(str(i)+" clusters")

    plt.scatter(T1[:, 0], T1[:, 1], c=pred)

    

    centroids = kmean.cluster_centers_

    centroids = pca1.transform(centroids)

    plt.plot(centroids[:, 0], centroids[:, 1], 'b+', markersize=30, color = 'brown', markeredgewidth = 3)
tempDf=preDf

tempDf['predK']=preds1[4]

tempJoin = data_orig.join(tempDf['predK'], how = 'left')

tempJoin.info()
#Finding Cross Validation Score by trying all possible permutations

cross_ma=0

tempJoin['prediction_map'] = tempJoin['predK']

for i in range(3**6):

    j=i

    lis=[]

    dic={}

    for k in range(6):

        lis.append(j%3)

        dic[k]=j%3

        j=j//3

    tempJoin['temp_map'] = tempJoin['predK'].map(dic)

    val_score=sum(tempJoin['Class'].values[0:175] == tempJoin['temp_map'].values[0:175])/175

    if (val_score>cross_ma):

        cross_ma=val_score

        tempJoin['prediction_map'] = tempJoin['temp_map']

cross_ma
from sklearn.neighbors import NearestNeighbors

ns = 36

nbrs = NearestNeighbors(n_neighbors = ns).fit(preDf)

distances, indices = nbrs.kneighbors(preDf)



kdist = []



for i in distances:

    avg = 0.0

    for j in i:

        avg += j

    avg = avg/(ns-1)

    kdist.append(avg)



kdist = sorted(kdist)

plt.plot(indices[:,0], kdist)
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=3.0, min_samples=10)

pred = dbscan.fit_predict(preDf)

plt.scatter(T1[:, 0], T1[:, 1], c=pred)
tempDf=preDf

tempDf['predDB']=pred

tempDf.groupby('predDB').count()
tempDf=tempDf.loc[tempDf.predDB!=-1,:]

tempJoin = data_orig.join(tempDf['predDB'], how = 'left')

tempJoin.info()
#Finding Cross Validation Score by trying all possible permutations

cross_ma=0

tempJoin['prediction_map'] = tempJoin['predDB']

for i in range(3**3):

    j=i

    lis=[]

    dic={}

    for k in range(3):

        lis.append(j%3)

        dic[k]=j%3

        j=j//3

    tempJoin['temp_map'] = tempJoin['predDB'].map(dic)

    val_score=sum(tempJoin['Class'].values[0:175] == tempJoin['temp_map'].values[0:175])/175

    if (val_score>cross_ma):

        cross_ma=val_score

        tempJoin['prediction_map'] = tempJoin['temp_map']

cross_ma
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree

from scipy.cluster.hierarchy import fcluster

linkage_matrix1 = linkage(preDf, "ward",metric="euclidean")

ddata1 = dendrogram(linkage_matrix1,color_threshold=10)
from sklearn.cluster import AgglomerativeClustering as AC

aggclus = AC(n_clusters = 6,affinity='euclidean',linkage='ward',compute_full_tree='auto')

y_aggclus= aggclus.fit_predict(preDf)

plt.scatter(T1[:, 0], T1[:, 1], c=y_aggclus)
tempDf=preDf

tempDf['pred_ac']=y_aggclus

tempJoin = data_orig.join(tempDf['pred_ac'], how = 'left')

tempJoin.info()
#Finding Cross Validation Score by trying all possible permutations

cross_ma=0

tempJoin['prediction_map'] = tempJoin['pred_ac']

for i in range(3**6):

    j=i

    lis=[]

    dic={}

    for k in range(6):

        lis.append(j%3)

        dic[k]=j%3

        j=j//3

    tempJoin['temp_map'] = tempJoin['pred_ac'].map(dic)

    val_score=sum(tempJoin['Class'].values[0:175] == tempJoin['temp_map'].values[0:175])/175

    if (val_score>cross_ma):

        cross_ma=val_score

        tempJoin['prediction_map'] = tempJoin['temp_map']

cross_ma
data2=finalDF.copy()
data2.info()
#Incorporating Results of Agglomerative Clustering

from sklearn.cluster import AgglomerativeClustering as AC

aggclus = AC(n_clusters = 6,affinity='euclidean',linkage='ward',compute_full_tree='auto')

y_aggclus= aggclus.fit_predict(data2)

plt.scatter(T1[:, 0], T1[:, 1], c=y_aggclus)
data2['predAC']=y_aggclus
#Scaling the results from AgglomerativeClustering

min_max_scaler = preprocessing.MinMaxScaler()

data2[['predAC']] = min_max_scaler.fit_transform(data2[['predAC']])
pca1 = PCA(n_components=2)

pca1.fit(data2)

T1 = pca1.transform(data2)

data2.info()
from sklearn.neighbors import NearestNeighbors

ns = 36

nbrs = NearestNeighbors(n_neighbors = ns).fit(data2)

distances, indices = nbrs.kneighbors(data2)



kdist = []



for i in distances:

    avg = 0.0

    for j in i:

        avg += j

    avg = avg/(ns-1)

    kdist.append(avg)



kdist = sorted(kdist)

plt.plot(indices[:,0], kdist)
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=3.0, min_samples=10)

pred = dbscan.fit_predict(data2)

plt.scatter(T1[:, 0], T1[:, 1], c=pred)
#Incorporating Results from DBSCAN

data2['pred_dbscan']=pred

data2.groupby('pred_dbscan').count()
#Taking tuples which are not outliers

datas2=data2.loc[(data2.pred_dbscan!=-1),:]
datas2.info()
#Scaling Results from DBSCAN

min_max_scaler = preprocessing.MinMaxScaler()

datas2[['pred_dbscan']] = min_max_scaler.fit_transform(datas2[['pred_dbscan']])
pca1 = PCA(n_components=2)

pca1.fit(datas2)

T1 = pca1.transform(datas2)

datas2.info()
from sklearn.cluster import KMeans



wcss = []

for i in range(2, 20):

    kmean = KMeans(n_clusters = i, random_state = 47)

    kmean.fit(datas2)

    wcss.append(kmean.inertia_)

    

plt.plot(range(2,20),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
import matplotlib

plt.figure(figsize=(16, 8))

preds1 = []

#colors = ['red','green','blue']

for i in range(2, 11):

    kmean = KMeans(n_clusters = i, random_state = 47)

    kmean.fit(datas2)

    pred = kmean.predict(datas2)

    preds1.append(pred)

    

    plt.subplot(2, 5, i - 1)

    plt.title(str(i)+" clusters")

    plt.scatter(T1[:, 0], T1[:, 1], c = pred )

    #, cmap=matplotlib.colors.ListedColormap(colors))

    

    centroids = kmean.cluster_centers_

    centroids = pca1.transform(centroids)

    plt.plot(centroids[:, 0], centroids[:, 1], 'b+', markersize=30, color = 'brown', markeredgewidth = 3)

    #plt.scatter(T1[:, 0], T1[:, 1], c = pred , cmap=matplotlib.colors.ListedColormap(colors))
#Incorporating results from K-means

datas2['predK']=preds1[6]

datas2['predK']=datas2['predK'].astype(np.int64)

datas2.groupby('predK').count()
datas2.info()
#Scaling results from K-means

min_max_scaler = preprocessing.MinMaxScaler()

datas2[['predK']] = min_max_scaler.fit_transform(datas2[['predK']])
pca1 = PCA(n_components=2)

pca1.fit(datas2)

T1 = pca1.transform(datas2)

datas2.info()
from sklearn.cluster import KMeans



wcss = []

for i in range(2, 15):

    kmean = KMeans(n_clusters = i, random_state = 47)

    kmean.fit(datas2)

    wcss.append(kmean.inertia_)

    

plt.plot(range(2,15),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
#Applying K-means on results from Agglomerative Clustering, DBSCAN, K-means

import matplotlib

plt.figure(figsize=(16, 8))

preds1 = []

#colors = ['red','green','blue']

for i in range(2, 11):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(datas2)

    pred = kmean.predict(datas2)

    preds1.append(pred)

    

    plt.subplot(2, 5, i - 1)

    plt.title(str(i)+" clusters")

    plt.scatter(T1[:, 0], T1[:, 1], c = pred )

    #, cmap=matplotlib.colors.ListedColormap(colors))

    

    centroids = kmean.cluster_centers_

    centroids = pca1.transform(centroids)

    plt.plot(centroids[:, 0], centroids[:, 1], 'b+', markersize=30, color = 'brown', markeredgewidth = 3)

    #plt.scatter(T1[:, 0], T1[:, 1], c = pred , cmap=matplotlib.colors.ListedColormap(colors))
datas2['final_pred']=preds1[6]

datas2.groupby('final_pred').count()
temp_join = data_orig.join(datas2['final_pred'], how = 'left')
temp_join= temp_join.loc[(pd.isnull(temp_join.final_pred)==False),:]

temp_join.info()
#Finding Cross Validation Score by trying all possible permutations

cross_ma=0



datas2['prediction_map'] = datas2['final_pred']

data_final = data_orig.join(datas2['prediction_map'], how = 'left')



for i in range(3**8):

    j=i

    lis=[]

    dic={}

    for k in range(8):

        lis.append(j%3)

        dic[k]=j%3

        j=j//3

    datas2['temp_map'] = datas2['final_pred'].map(dic)

    val_score=sum(temp_join['Class'].values[0:168] == datas2['temp_map'].values[0:168])/168

    if (val_score>cross_ma):

        cross_ma=val_score

        data_final['prediction_map'] = datas2['temp_map']

cross_ma
data_final.info()
sub=data_final[['id','prediction_map']]

sub.groupby('prediction_map').count()
sub.loc[pd.isnull(sub['prediction_map'])==True,'prediction_map']=1

sub['prediction_map']=sub['prediction_map'].astype(np.int64)

sub.rename(columns={'prediction_map':'Class'},inplace=True)

sub.info()
sub=sub[175:]

sub.head()
sub.to_csv('Final_Sub.csv',index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(sub)