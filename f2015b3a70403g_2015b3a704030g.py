import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

#import seaborn as sns



#import sklearn.preprocessing as sk

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm
df=pd.read_csv("../input/dataset.csv", sep=",")
data=df
data = data.drop(['Class'], 1) # Drop Total from domain knowledge
out = data.replace({'?': np.nan})
out.fillna(out.mode().iloc[0], inplace=True)

#out = out.replace({ np.nan:0})
#out.fillna(out.mean().iloc[0], inplace=True)

out = out.replace({ np.nan:0})
idcol=data['id']
out = out.drop(['id'], 1) # Drop Total from domain knowledge

out.columns = out.columns.str.replace(' ','')

out.columns = out.columns.str.replace('#','')

# out['Credit1']=out.Credit1.astype(int)

# out['InstallmentRate']=out.InstallmentRate.astype(int)

# out['MonthlyPeriod']=out.MonthlyPeriod.astype(int)

# out['TenancyPeriod']=out.TenancyPeriod.astype(int)

# out['Authorities']=out.Authorities.astype(int)

# out['Expatriate']=out.Expatriate.astype(float)

# out['InstallmentCredit']=out.Expatriate.astype(float)

# out['YearlyPeriod']=out.YearlyPeriod.astype(float)

out.info()
#out['Plotsize']=out.Plotsize.astype(str).str.upper()
data1=out
data1 = data1.drop(['Phone','InstallmentRate','MonthlyPeriod'], 1) # Drop Total from domain knowledge

data1.info()
data1 = pd.get_dummies(data1, columns=["Account1","History","Motive","Account2","EmploymentPeriod","Gender&Type","Plotsize","Plan","Housing","Post",'Expatriate',"Sponsors"])

data1.info()
from sklearn import preprocessing



#Performing Min_Max Normalization

standardscaler = preprocessing.StandardScaler()

np_scaled = standardscaler.fit_transform(data1)

dataN1 = pd.DataFrame(np_scaled)

dataN1.head()
from sklearn.decomposition import PCA

pca1 = PCA(n_components=2)

pca1.fit(dataN1)

T1 = pca1.transform(dataN1)
from sklearn.cluster import KMeans

plt.figure(figsize=(16, 8))

preds1 = []

for i in range(2, 11):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(dataN1)

    pred = kmean.predict(dataN1)

    preds1.append(pred)

    

    plt.subplot(2, 5, i - 1)

    plt.title(str(i)+" clusters")

    plt.scatter(T1[:, 0], T1[:, 1], c=pred)

    

    centroids = kmean.cluster_centers_

    centroids = pca1.transform(centroids)

    plt.plot(centroids[:, 0], centroids[:, 1], 'b+', markersize=30, color = 'brown', markeredgewidth = 3)
dataN2=dataN1
colors = ['red','green','blue','yellow','purple','pink','palegreen','violet','cyan']
plt.figure(figsize=(16, 8))



kmean = KMeans(n_clusters = 3, random_state = 42)

kmean.fit(dataN2)

pred = kmean.predict(dataN2)

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
pred
res=[]

for i in range(len(pred)):

    res.append(pred[i])
res
len(res)
df1=pd.read_csv("./dataset.csv", sep=",")
df1.info()
res1 =[]

for i in range(175):

    res1.append(df1['Class'][i])
res1
res1 = [int(x) for x in res1]
len(res1)
res2 =[]

for i in range(175):

    res2.append(res[i])
res2
count=0

for i in range(175):

    if((res1[i]==0 and res2[i]==0) or (res1[i]==1 and res2[i]==1) or (res1[i]==2 and res2[i]==2)):

        count+=1

count
count=0

for i in range(175):

    if((res1[i]==0 and res2[i]==1) or (res1[i]==1 and res2[i]==0) or (res1[i]==2 and res2[i]==2)):

        count+=1

count
count=0

for i in range(175):

    if((res1[i]==0 and res2[i]==2) or (res1[i]==1 and res2[i]==1) or (res1[i]==2 and res2[i]==0)):

        count+=1

count
count=0

for i in range(175):

    if((res1[i]==0 and res2[i]==0) or (res1[i]==1 and res2[i]==2) or (res1[i]==2 and res2[i]==1)):

        count+=1

count
count=0

for i in range(175):

    if((res1[i]==0 and res2[i]==1) or (res1[i]==1 and res2[i]==2) or (res1[i]==2 and res2[i]==0)):

        count+=1

count
count=0

for i in range(175):

    if((res1[i]==0 and res2[i]==2) or (res1[i]==1 and res2[i]==0) or (res1[i]==2 and res2[i]==1)):

        count+=1

count
for i in range(len(res)):

    if(res[i]==1):

        res[i]=0

    elif(res[i]==0):

        res[i]=1

    elif(res[i]==2):

        res[i]=2
res
res3=[]

for i in range(175):

    res3.append(res[i])

res3
ans=0

for i in range(175):

    if(res3[i]==res1[i]):

        ans+=1

ans/175
len(res)
len(idcol)
idcol1=idcol[175:]
res1=res[175:]
dict={'id':idcol1,'Class':res1}
dict
ans = pd.DataFrame(dict,columns=['id','Class'])
ans
ans.to_csv('submission.csv', sep=',',index=False)
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



create_download_link(ans)