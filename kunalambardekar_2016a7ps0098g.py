import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data_og=pd.read_csv("../input/dataset/dataset.csv",sep=',')

DATA=data_og
DATA.head()
DATA.info()
#PREPROCESSING:

DATA['Expatriate']=DATA['Expatriate'].replace({False: 0, True: 1})

DATA['Phone']=DATA['Phone'].replace('Yes',1)

DATA['Phone']=DATA['Phone'].replace('No',0)

DATA['Account2']=DATA['Account2'].replace('Sacc4','sacc4')

DATA['Sponsors']=DATA['Sponsors'].replace('g1','G1')

DATA['Phone']=DATA['Phone'].replace('yes',1)

DATA['Phone']=DATA['Phone'].replace('no',0)

DATA['Plotsize']=DATA['Plotsize'].replace('me','ME')

DATA['Plotsize']=DATA['Plotsize'].replace('sm','SM')

DATA['Plotsize']=DATA['Plotsize'].replace('la','LA')

DATA['Plotsize']=DATA['Plotsize'].replace('M.E.','ME')
for x in DATA.columns:

    DATA[x]=DATA[x].replace('?',np.nan)

DATA
DATA['Tenancy Period']=DATA['Tenancy Period'].astype(float)

DATA['Monthly Period']=DATA['Monthly Period'].astype(float)

DATA['Age']=DATA['Age'].astype(float)

DATA['Credit1']=DATA['Credit1'].astype(float)

DATA['InstallmentRate']=DATA['InstallmentRate'].astype(float)

DATA['InstallmentCredit']=DATA['InstallmentCredit'].astype(float)

DATA['Yearly Period']=DATA['Yearly Period'].astype(float)



DATA.info()
null_col=DATA.columns[DATA.isnull().any()]

null_col
DATA['Monthly Period']=DATA['Monthly Period'].fillna(DATA['Monthly Period'].mean())

DATA['Credit1']=DATA['Credit1'].fillna(DATA['Credit1'].mean())

DATA['InstallmentRate']=DATA['InstallmentRate'].fillna(DATA['InstallmentRate'].mean())

DATA['Tenancy Period']=DATA['Tenancy Period'].fillna(DATA['Tenancy Period'].mean())

DATA['InstallmentCredit']=DATA['InstallmentCredit'].fillna(DATA['InstallmentCredit'].mean())

DATA['Age']=DATA['Age'].fillna(DATA['Age'].mean())

DATA['Yearly Period']=DATA['Yearly Period'].fillna(DATA['Yearly Period'].mean())



null_col=DATA.columns[DATA.isnull().any()]

null_col
DATA=DATA.drop(['Account1', 'History', 'Motive', 'Class'],1)	#As they have null values
import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))

corr = DATA.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax, annot = True);
DATA=DATA.drop(['Credit1', 'Yearly Period'],1) #Yearly & Monthly have corr=1, Credit1 & InstallmentCredit have corr=0.96

DATA=DATA.drop(['id'],1)

DATA=DATA.drop(['Employment Period','Gender&Type','Plotsize','Account2','Post'],1)	#dimension is >=4

DATA=DATA.drop(['Phone'],1)		#To simplify clustered result

expDATA = pd.get_dummies(DATA, columns=["Sponsors","Housing","Plan"])	#CategoricalVariables

expDATA.info()
#Min-Max Normalisation:



from sklearn import preprocessing

#Performing Min_Max Normalization

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(expDATA)

sclDATA = pd.DataFrame(np_scaled)

sclDATA.head()
#The Elbow Method:



from sklearn.decomposition import PCA

pca1 = PCA(n_components=2)

pca1.fit(sclDATA)

T1 = pca1.transform(sclDATA)



from sklearn.cluster import KMeans



wcss = []

for i in range(2, 19):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(sclDATA)

    wcss.append(kmean.inertia_)

    

plt.plot(range(2,19),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
#Multiple no. of clusters

from sklearn.cluster import KMeans

plt.figure(figsize=(16, 8))

preds1 = []

for i in range(2, 11):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(sclDATA)

    pred = kmean.predict(sclDATA)

    preds1.append(pred)

    

    plt.subplot(2, 5, i - 1)

    plt.title(str(i)+" clusters")

    plt.scatter(T1[:, 0], T1[:, 1], c=pred)

    

    centroids = kmean.cluster_centers_

    centroids = pca1.transform(centroids)

    plt.plot(centroids[:, 0], centroids[:, 1], 'b+', markersize=30, color = 'brown', markeredgewidth = 3)
colors = ['red','green','blue','yellow','purple','pink','palegreen','violet','cyan']



#Using 9 clusters

plt.figure(figsize=(16, 8))



kmean = KMeans(n_clusters = 9, random_state = 42)

kmean.fit(sclDATA)

pred = kmean.predict(sclDATA)

pred_pd = pd.DataFrame(pred)

arr = pred_pd[0].unique()



for i in arr:

    meanx = 0

    meany = 0

    count = 0

    for j in range(len(pred)):

        if i == pred[j]:

            count+=1

            meanx+=T1[j,0]

            meany+=T1[j,1]

            plt.scatter(T1[j, 0], T1[j, 1], c=colors[i])

    meanx = meanx/count

    meany = meany/count

    plt.annotate(i,(meanx, meany),size=30, weight='bold', color='black', backgroundcolor=colors[i])

    

for i in range(176):

    plt.annotate(str(data_og['Class'][i]),(T1[i,0], T1[i,1]),size=10, weight='bold', color='black')
pred
#To obtain resulting dataframe

res = []

ids = [] 

for i in range(len(pred)): 

    if(i<=174): 

        continue 

    else: 

        if pred[i] == 0 or pred[i]==4 or pred[i]==5: 

            res.append(2) 

        elif pred[i] ==3 or pred[i]==8 or pred[i]==7 or pred[i]==1:

            res.append(0) 

        elif pred[i] == 2 or pred[i]==6: 

            res.append(1) 

        ids.append(data_og['id'][i])

res1 = pd.DataFrame(res) 

final = pd.concat([pd.DataFrame(ids), pd.DataFrame(res1)], axis=1).reindex() 

final = final.rename(columns={0: "id"}) 

final.columns=['id','Class'] 

final.head()
final.to_csv('sub4.csv', index = False)

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



create_download_link(final)