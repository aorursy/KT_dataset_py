#Import Important Functions

import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

np.random.seed(42)

from sklearn.preprocessing import StandardScaler,LabelEncoder

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN

from sklearn.metrics import accuracy_score
#Read the dataframe

df=pd.read_csv("../input/dataset.csv", sep=",")
#Split the names of variables depending on type

categorical_variables = ['Account1','History','Motive','Account2','Employment Period','Gender&Type','Sponsors','Plotsize','Plan','Housing','Post','Phone','Expatriate']

numerical_variables = ['Monthly Period','Credit1','InstallmentRate','Tenancy Period','Age','#Credits','#Authorities','InstallmentCredit','Yearly Period']
#Checking for absurd values in the columns

for i in df.columns:

    print(i,df[i].unique())
#Changing the column values as desired to remove visible discrepancies

df = df.replace('?',np.NaN)

df[numerical_variables] = df[numerical_variables].apply(pd.to_numeric)

df = df.replace('M.E.','ME')

df = df.replace('sm','SM')

df = df.replace('me','ME')

df = df.replace('la','LA')

df = df.replace('Sacc4','sacc4')

df = df.replace('g1','G1')
#Filling Null Values in the DataFrame

#Median for Numerical

#Mode for Categorical

df[numerical_variables]= df[numerical_variables].fillna(df[numerical_variables].median())

df[categorical_variables] = df[categorical_variables].apply(lambda x:x.fillna(x.value_counts().index[0]))
#Storing a separate df for tuples which have classes

class_df = df.dropna()
class_df.shape
#Plotting a heatmap of correlation

f, ax = plt.subplots(figsize=(30,20))

corr = df[numerical_variables+['Class']].corr()

plt.title('Fig 1. Heatmap of Correlation Matrix')

sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax, annot = True);
#Dropping Columns with Correlation < 0.2

dropcol = (corr['Class']<0.2) & (corr['Class']>-0.2)

todrop = list(dropcol[dropcol==True].keys())
#Converting Motive to One Hot Encoding will create 10 columns. Drop such attributes.

todrop += ['Account1','History','Motive']
#Removing those attributes from the lists

for i in todrop:

    if i in numerical_variables:

        numerical_variables.remove(i)

    elif i in categorical_variables:

        categorical_variables.remove(i)
#Drop from respective dataframes

label_drop = df.drop(todrop,axis=1)

class_drop = class_df.drop(todrop,axis=1)
label_drop.shape
label_drop.columns
class_fin = class_drop.drop('id',axis=1)
cat2 = []

for c in class_fin.columns:

    if c in categorical_variables:

        cat2.append(c)
#One-Hot Encode the Categorical Variables Present for distance based algorithms

oh_data = pd.get_dummies(label_drop,columns = cat2)

oh_class = pd.get_dummies(class_fin,columns=cat2)
#Drop the duplicate values in the data

drop_duplicates = oh_data.drop(['id'],axis=1).drop_duplicates(keep='first')
drop_duplicates.columns
oh_data.columns
numerical_variables
#Scaling might be required if any numerical attributes are left

# scaled_df = oh_data.copy()

# scaled_df[numerical_variables] = StandardScaler().fit_transform(scaled_df[numerical_variables])
#Visualizing the given classes using PCA

model=PCA(n_components=2)

model_data=model.fit(oh_class.drop(['Class'],axis=1)).transform(oh_class.drop('Class',axis=1))
plt.figure(figsize=(8,6))

plt.xlabel('X')

plt.ylabel('Y')

plt.title('Fig 2. PCA Representation of Given Classes')

plt.scatter(model_data[:,0],model_data[:,1],c=oh_class['Class'],cmap = plt.get_cmap('rainbow_r'))
#Visualizing the given classes using TSNE

model=TSNE(n_iter=10000,n_components=2,perplexity=100)

model_data=model.fit_transform(oh_class.drop('Class',axis=1))

model_data.shape
plt.figure(figsize=(8,6))

plt.xlabel('X')

plt.ylabel('Y')

plt.title('Fig 3. TSNE Representation of Given Classes')

plt.scatter(model_data[:,0],model_data[:,1],c=oh_class['Class'],cmap = plt.get_cmap('rainbow'))
#Make a dataframe to store all the predictions

comb_data = oh_data.copy()
#Dataframe for final submission

suball_csv = oh_data[['id','Class']]
#Import dendogram and linkage to get predictions based on a dendogram

from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree

from scipy.cluster.hierarchy import fcluster

linkage_matrix1 = linkage(oh_data.drop(['id','Class'],axis=1), "ward",metric="euclidean")

plt.title('Fig 4. Dendogram')

ddata1 = dendrogram(linkage_matrix1,color_threshold=10)

#Make a cut at 20 clusters to prevent underfitting and not more to prevent overfitting

y_ac=cut_tree(linkage_matrix1, n_clusters = 20).T

y_ac = y_ac.reshape((1031,))
#Making predictions and storing in submission data

wpred = 'predDend'

suball_csv[wpred]=y_ac

test_df = class_drop.merge(suball_csv.drop('Class',axis=1),how = 'inner',on = ['id'])

#MAJORITY VOTING

#Making a map of MAJORITY CLASS for each predicted cluster

dic = {}

for i in sorted(test_df[wpred].unique()):

    dic[i] = int(test_df[test_df[wpred]==i]['Class'].value_counts().index[0])





#Mapping predictions in submission data and testing data to majority class

test_df[wpred] = test_df[wpred].map(dic)

#Adding predictions to combined data

comb_data[wpred]=suball_csv[wpred]

suball_csv[wpred] = suball_csv[wpred].map(dic)

#Testing accuracy for Dendogram

accuracy_score(test_df['Class'],test_df[wpred])
#Plotting Elbow Method Graph for KMeans

wcss = []

for i in range(2,25):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(drop_duplicates.drop(['Class'],axis=1))

    wcss.append(kmean.inertia_)

plt.figure(figsize=(10,8))    

plt.plot(range(2,25),wcss)

plt.title('Fig 5. The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
#Transforming One Hot Data using PCA for Representation

pca=PCA(n_components=2)

T=pca.fit(drop_duplicates.drop(['Class'],axis=1)).transform(oh_data.drop(['Class','id'],axis=1))

T.shape
#Plotting Clusters with different K values

preds1 = []

for i in range(2, 18):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(drop_duplicates.drop(['Class'],axis=1))

    #TO copy predictions on non-duplicated data to duplicated data

    pred = kmean.predict(oh_data.drop(['Class','id'],axis=1))

    preds1.append(pred)

    

    plt.subplot(2,8, i - 1)

    plt.title(str(i)+" clusters")

    plt.scatter(T[:, 0], T[:, 1], c=pred)

    

    centroids = kmean.cluster_centers_

    centroids = pca.transform(centroids)

    plt.plot(centroids[:, 0], centroids[:, 1], 'b+', markersize=30, color = 'brown', markeredgewidth = 3)

    

fig = plt.gcf()

fig.set_size_inches((20,10))

fig.suptitle("Fig 6. Multiple K Clusters")

plt.show()
#The elbow can be seen somewhere around 15.

#Choose K = 15 

colors = ['red','green','blue','yellow','purple','pink','palegreen','violet','cyan','black']

colors = colors + colors

plt.figure(figsize=(16, 8))



kmean = KMeans(n_clusters = 15, random_state = 42)

kmean.fit(drop_duplicates.drop(['Class'],axis=1))

#TO copy predictions on non-duplicated data to duplicated data

pred = kmean.predict(oh_data.drop(['Class','id'],axis=1))

pred_pd = pd.DataFrame(pred)

arr = pred_pd[0].unique()

plt.title('Fig 7. K-Means Results')

for i in arr:

    meanx = 0

    meany = 0

    count = 0

    for j in range(len(pred)):

        if i == pred[j]:

            count+=1

            meanx+=T[j,0]

            meany+=T[j,1]

            plt.scatter(T[j, 0], T[j, 1], c=colors[i])

    meanx = meanx/count

    meany = meany/count

    plt.annotate(i,(meanx, meany),size=30, weight='bold', color='black', backgroundcolor=colors[i])
#Making predictions and storing in submission data

wpred = 'predK'

suball_csv[wpred]=pred

test_df = class_drop.merge(suball_csv.drop('Class',axis=1),how = 'inner',on = ['id'])

#MAJORITY VOTING

#Making a map of MAJORITY CLASS for each predicted cluster

dic = {}

for i in sorted(test_df[wpred].unique()):

    dic[i] = int(test_df[test_df[wpred]==i]['Class'].value_counts().index[0])





#Mapping predictions in submission data and testing data to majority class

test_df[wpred] = test_df[wpred].map(dic)

#Adding predictions to combined data

comb_data[wpred]=suball_csv[wpred]

suball_csv[wpred] = suball_csv[wpred].map(dic)

#Testing accuracy for KMeans

accuracy_score(test_df['Class'],test_df[wpred])
#Choose n = 15 as decided using KMeans

agg = AgglomerativeClustering(n_clusters=15,affinity='euclidean',linkage='ward')

pred_agg = agg.fit_predict(drop_duplicates.drop(['Class'],axis=1))



drop_duplicates['predAgg'] = pred_agg



#TO copy predictions on non-duplicated data to duplicated data

temp_data = oh_data.merge(drop_duplicates,on=list(drop_duplicates.drop('predAgg',axis=1).columns),how='left')

pred_agg = temp_data['predAgg']

pred_agg = np.array(pred_agg)

plt.title('Fig 8. Agglomerative Clustering Results')

plt.scatter(T[:, 0], T[:, 1], c=pred_agg)
#Making predictions and storing in submission data

wpred = 'predAgg'

suball_csv[wpred]=pred_agg

test_df = class_drop.merge(suball_csv.drop('Class',axis=1),how = 'inner',on = ['id'])

#MAJORITY VOTING

#Making a map of MAJORITY CLASS for each predicted cluster

dic = {}

for i in sorted(test_df[wpred].unique()):

    dic[i] = int(test_df[test_df[wpred]==i]['Class'].value_counts().index[0])





#Mapping predictions in submission data and testing data to majority class

test_df[wpred] = test_df[wpred].map(dic)

#Adding predictions to combined data

comb_data[wpred]=suball_csv[wpred]

suball_csv[wpred] = suball_csv[wpred].map(dic)

#Testing accuracy for Agglomerative Clustering

accuracy_score(test_df['Class'],test_df[wpred])
#Making the Neighbors vs Distance Plot to select EPS

from sklearn.neighbors import NearestNeighbors

ns = 17

dat = drop_duplicates.drop(['Class'],axis=1)



nbrs = NearestNeighbors(n_neighbors = ns).fit(dat)

distances, indices = nbrs.kneighbors(dat)



kdist = []

for i in distances:

    avg = 0.0

    for j in i:

        avg += j

    avg = avg/(ns-1)

    kdist.append(avg)



f = plt.gcf()

f.set_size_inches(10,6)

kdist = sorted(kdist)

plt.xlabel('Number of Neighbors')

plt.ylabel('Distances')

plt.title('Fig 9. K-Distance Graph')

plt.plot(sorted(indices[:,0]),kdist)
#Select eps = 1.8 for DBSCAN as suggested by the graph

dbs =  DBSCAN(eps=1.8,min_samples=14,metric='euclidean')

pred_dbs = dbs.fit_predict(drop_duplicates.drop(['Class'],axis=1))



#TO copy predictions on non-duplicated data to duplicated data

drop_duplicates['preddbs'] = pred_dbs

temp_data = oh_data.merge(drop_duplicates,on=list(drop_duplicates.drop(['preddbs','predAgg'],axis=1).columns),how='left')

pred_dbs = temp_data['preddbs']

pred_dbs = np.array(pred_dbs)

#Plotting the graph

plt.title('Fig 10. DBSCAN Results')

plt.scatter(T[:, 0], T[:, 1], c=pred_dbs)
np.unique(pred_dbs)
#Making predictions and storing in submission data

wpred = 'preddbs'

suball_csv[wpred]=pred_dbs

test_df = class_drop.merge(suball_csv.drop('Class',axis=1),how = 'inner',on = ['id'])

#MAJORITY VOTING

#Making a map of MAJORITY CLASS for each predicted cluster

dic = {}

for i in sorted(test_df[wpred].unique()):

    dic[i] = int(test_df[test_df[wpred]==i]['Class'].value_counts().index[0])





#Mapping predictions in submission data and testing data to majority class

test_df[wpred] = test_df[wpred].map(dic)

#Adding predictions to combined data

comb_data[wpred]=suball_csv[wpred]

suball_csv[wpred] = suball_csv[wpred].map(dic)

#Testing accuracy for Agglomerative Clustering

accuracy_score(test_df['Class'],test_df[wpred])
accuracy_score(test_df['Class'],test_df['preddbs'])
suball_csv[wpred].value_counts()
comb_data.columns
#Making a correlation heatmap

corr = comb_data.corr()

plt.title('Fig 11. Final Data Correlation')

sns.heatmap(corr)
dropcol =  (corr['Class']>0.075) | (corr['Class']<-0.075)

sum(dropcol)

todrop = list(dropcol[dropcol==False].keys())

todrop.remove('preddbs')
fin_comb_data = comb_data.drop(todrop,axis=1)
fin_comb_data.columns
#fin_comb_data.drop(['predComb'],axis=1,inplace=True)
fin_comb_data =  pd.get_dummies(fin_comb_data,columns = ['predK','predDend','preddbs'] )
fin_comb_data.columns
#Plotting the Elbow Method Graph for Final Data

wcss = []

for i in range(2,25):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(fin_comb_data.drop(['Class','id'],axis=1))

    wcss.append(kmean.inertia_)

plt.figure(figsize=(10,8))    

plt.plot(range(2,25),wcss)

plt.title('Fig 12. The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
pca=PCA(n_components=2)

T=pca.fit(fin_comb_data.drop(['Class','id'],axis=1)).transform(fin_comb_data.drop(['Class','id'],axis=1))

T.shape
#Plotting Clusters for Different K-values

preds1 = []

for i in range(2, 17):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(fin_comb_data.drop(['Class','id'],axis=1))

    pred = kmean.predict(fin_comb_data.drop(['Class','id'],axis=1))

    preds1.append(pred)

    

    plt.subplot(2,8, i - 1)

    plt.title(str(i)+" clusters")

    plt.scatter(T[:, 0], T[:, 1], c=pred)

    

    centroids = kmean.cluster_centers_

    centroids = pca.transform(centroids)

    plt.plot(centroids[:, 0], centroids[:, 1], 'b+', markersize=30, color = 'brown', markeredgewidth = 3)

fig = plt.gcf()

fig.set_size_inches((20,10))

fig.suptitle("Fig 13. Multiple K Clusters")

plt.show()
colors = ['red','green','blue','yellow','purple','pink','palegreen','violet','cyan','black']

colors = colors + colors

plt.figure(figsize=(16, 8))

plt.title("Fig 14. K-Means on Combined Data")

#Choosing 13 clusters as shown by elbow method graph

kmean = KMeans(n_clusters =13, random_state = 42)

kmean.fit(fin_comb_data.drop(['Class','id'],axis=1))

pred = kmean.predict(fin_comb_data.drop(['Class','id'],axis=1))

pred_pd = pd.DataFrame(pred)

arr = pred_pd[0].unique()



for i in arr:

    meanx = 0

    meany = 0

    count = 0

    for j in range(len(pred)):

        if i == pred[j]:

            count+=1

            meanx+=T[j,0]

            meany+=T[j,1]

            plt.scatter(T[j, 0], T[j, 1], c=colors[i])

    meanx = meanx/count

    meany = meany/count

    plt.annotate(i,(meanx, meany),size=30, weight='bold', color='black', backgroundcolor=colors[i])
#Making predictions and storing in submission data

wpred = 'predComb'

suball_csv[wpred]=pred

test_df = class_drop.merge(suball_csv.drop('Class',axis=1),how = 'inner',on = ['id'])

#MAJORITY VOTING

#Making a map of MAJORITY CLASS for each predicted cluster

dic = {}

for i in sorted(test_df[wpred].unique()):

    dic[i] = int(test_df[test_df[wpred]==i]['Class'].value_counts().index[0])





#Mapping predictions in submission data and testing data to majority class

test_df[wpred] = test_df[wpred].map(dic)

#Adding predictions to combined data

comb_data[wpred]=suball_csv[wpred]

suball_csv[wpred] = suball_csv[wpred].map(dic)

#Testing accuracy for Agglomerative Clustering

accuracy_score(test_df['Class'],test_df[wpred])
#Check if null

spred = 'predComb'

sum(test_df['Class']==test_df[spred])

suball_csv[spred].isnull().sum()
#Get Count total

suball_csv[['id',spred]].groupby(spred).count()
#Renaming the saving dataframe

sub_csv = suball_csv[['id',spred]]

sub_csv[spred] = sub_csv[spred].apply(int)

sub_csv = sub_csv.rename(columns = {spred:"Class"})

sub_csv = sub_csv[~sub_csv['id'].isin(test_df['id'])]
#Saving the submission

sub_csv.to_csv('final-submission.csv',index=False)
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



create_download_link(sub_csv)