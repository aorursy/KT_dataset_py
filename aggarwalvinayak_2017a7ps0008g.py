import numpy as np

import pandas as pd



%matplotlib inline

import matplotlib.pyplot as plt



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler

import seaborn as sns

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans

import math

import copy

from scipy import stats as stat

inpdf=pd.read_csv("../input/dmassign1/data.csv",sep=",")

inpdf.info(verbose=True)
datahigh=inpdf.replace("M.E.","ME")

datahigh=datahigh.replace("me","ME")

datahigh=datahigh.replace("la","LA")

datahigh=datahigh.replace("sm","SM")

datahigh = pd.get_dummies(datahigh, columns=["Col189","Col190","Col191","Col192","Col193","Col194","Col195","Col196","Col197",])

datahigh=datahigh.drop(['Col192_?','Col193_?','Col194_?','Col195_?','Col196_?','Col197_?'],axis=1)

datahigh=datahigh.replace('?', np.nan)

missingdata=datahigh.select_dtypes(include=['object']).columns

missingdata=missingdata.drop('ID')

for i in missingdata:

    datahigh[i]=datahigh[i].astype('Float64')



    

datahigh=datahigh.drop(["Class"],axis=1).fillna(datahigh.mean())

datahigh['Class']=inpdf['Class'].values

datahigh
datalow=datahigh.loc[:,:'Col188']

datalow
datacat=datahigh.loc[:,'Col185':]

datacat["ID"]=datahigh['ID']

datacat.head()
data = datalow.drop(['ID'],axis=1)

# data = datahigh.drop(['ID',"Class"],axis=1)
# scaler=StandardScaler()

# datahighh=scaler.fit(data).transform(data)

# data=pd.DataFrame(datahighh)

# data.tail()
trans = RobustScaler().fit(data)

datahighh=trans.transform(data)

datahigh1 = pd.DataFrame(datahighh)

datahigh1['Class'] = inpdf['Class']

datahigh1['ID'] = inpdf['ID']



data=datahigh1
labeleddata=datahigh[datahigh["Class"] >-1]

labeleddata.tail()
corr = data.corr()

corr

graph ,axis= plt.subplots()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

square=True, ax=axis, annot = False)
print(labeleddata[labeleddata.columns[0:]].corr()['Class'][:].sort_values(ascending=False)[:15])

print(labeleddata[labeleddata.columns[0:]].corr()['Class'][:].sort_values(ascending=True)[:15])
pca50 = PCA(n_components=50)

pca50.fit(data.drop(['ID','Class'],axis=1))

datared1 = pca50.transform(data.drop(['ID','Class'],axis=1))
tsne=TSNE(n_iter=3000,n_components=2,perplexity=15,n_jobs=4)

datared1 = tsne.fit_transform(data.drop(['ID','Class'],axis=1))

plt.figure(figsize=(10,8))

sns.scatterplot(

    x=datared1[:1300,0], y=datared1[:1300,1],

    hue=labeleddata['Class'],

    palette=sns.color_palette("hls", 5),

    

    legend="full",

    alpha=0.3,

)
datared=pd.DataFrame(datared1)

datared

# labelarr=list()

# datah1=datared

# datah1['Class']=datahigh['Class']

# datah1['ID']=datahigh['ID']

# labelarr.append(datah1[datah1["Class"] ==1].drop(['Class'],axis=1))

# labelarr.append(datah1[datah1["Class"] ==2].drop(['Class'],axis=1))

# labelarr.append(datah1[datah1["Class"] ==3].drop(['Class'],axis=1))

# labelarr.append(datah1[datah1["Class"] ==4].drop(['Class'],axis=1))

# labelarr.append(datah1[datah1["Class"] ==5].drop(['Class'],axis=1))



# centroids=list()

# centroids.append(labelarr[0].mean(axis=0))

# centroids.append(labelarr[1].mean(axis=0))

# centroids.append(labelarr[2].mean(axis=0))

# centroids.append(labelarr[3].mean(axis=0))

# centroids.append(labelarr[4].mean(axis=0))

# centroids=np.asarray(centroids)

# centroids



# #initialisation with centroids of 1300 samples

# kmeans = KMeans(n_clusters=5,init=centroids,n_jobs=4).fit(datared)
# kmeans = KMeans(n_clusters=5,n_jobs=4,random_state=45).fit(datared)

# maxx=0

# mapping=list()

# for a in range(1,6):

#     for b in range(1,6):

#         for c in range(1,6):

#             for d in range(1,6):

#                 for e in range(1,6):

#                         count=[0,0,0,0,0]

#                         lbldata_trans=list()

#                         for ele in ll:

#                             if(ele==1):

#                                 lbldata_trans.append(a)

#                             elif(ele==2):

#                                 lbldata_trans.append(b)

#                             elif(ele==3):

#                                 lbldata_trans.append(c)

#                             elif(ele==4):

#                                 lbldata_trans.append(d)

#                             elif(ele==5):

#                                 lbldata_trans.append(e)

#                         score=0

#                         for i in range(len(lbldata_trans)):

#                             if(lbldata_trans[i]==givnclass[i]):

#                                 count[lbldata_trans[i]-1]+=1

#                                 score+=1

#                         if(score-maxx>-10):

#                             mp=[a,b,c,d,e]

#                             print("--",score,mp, "Count",count)

#                         if(score>maxx):

#                             maxx=score

#                             mapping=[a,b,c,d,e]

#                             print(maxx,mapping)

                            

    
# def eucdist(a,b):

#     dist=0

#     for i in range(len(a)):

#         dist+=(a[i]-b[i])**2

#     dist=math.sqrt(dist)

#     return dist



# def centroid_loss(c1,c2):

#     loss=0

#     for i in range(len(c1)):

#         loss+=eucdist(c1[i],c2[i])

#     return loss
# num_clust=5

# traindata = datared

# traindata=traindata.values.tolist()

# old_centroids = np.zeros(centroids.shape)

# loss = centroid_loss(old_centroids,centroids)



# clusters = [-1]*len(traindata)

# init_labels = inpdf[:1300]["Class"]

# for i in range(len(init_labels)):

#     clusters[i] = init_labels[i]-1

# while (loss!=0):

#     for point in range(1300,len(traindata)):

#         clus_dist = list()

#         for centr in range(num_clust):

#             clus_dist.append(eucdist( traindata[point],centroids[centr]))

#         clusters[point]=np.argmin(clus_dist)

        

#     for i in range(num_clust):

#         old_centroids[i] = copy.deepcopy(centroids[i])

    

#     for i in range(num_clust):

#         list_clust = list()

#         for j in range(len(traindata)):

#             if(clusters[j]==i):

#                 list_clust.append(traindata[j])

#         centroids[i] = np.mean(list_clust)

        

#     loss = centroid_loss(old_centroids,centroids)

#     print(loss)             
# for i in range(len(clusters)):

#     clusters[i]=clusters[i]+1

    

# output=clusters[1300:]
nclus=25

kmeans = KMeans(n_clusters=nclus,n_jobs=4,random_state=42).fit(datared)
predlabel=kmeans.labels_[:1300]

givnlabel=labeleddata['Class']
mapping=list()

for i in range(nclus):   

    mapping.append(int(stat.mode(givnlabel[predlabel==i])[0]))

mapping

output=list()

for i in kmeans.labels_[1300:]:

    output.append(mapping[i])
outp = pd.DataFrame({'ID': inpdf['ID'].iloc[1300:], 'Class': output})

len(outp)

outp.head()
val=0

comp=list()

for i in kmeans.labels_[:1300]:

    comp.append(mapping[i])

for i in range(1300):

    if(comp[i]==givnlabel[i]):

        val+=1

val
from sklearn.cluster import AgglomerativeClustering as AC
nclus=35

agglo = AC(n_clusters=nclus)

predi = agglo.fit_predict(datared)

predlabel=predi[:1300]

givnlabel=labeleddata['Class']
mapping=list()

for i in range(nclus):   

    mapping.append(int(stat.mode(givnlabel[predlabel==i])[0]))

mapping
output=list()

for i in predi[1300:]:

    output.append(mapping[i])
val=0

comp=list()

for i in predlabel:

    comp.append(mapping[i])

for i in range(1300):

    if(comp[i]==givnlabel[i]):

        val+=1

val
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "Agglo_tsne2.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(outp)
