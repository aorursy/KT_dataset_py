import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data_orig = pd.read_csv("../input/dmassign1/data.csv", sep=',',encoding = "ISO-8859-1")

d = data_orig
dum=[]

dum.append('Col189')

dum.append('Col190')

dum.append('Col191')

dum.append('Col193')

dum.append('Col194')

dum.append('Col195')

dum.append('Col196')
final_data=d.copy()
final_data=final_data.drop(['Class'],axis=1)

final_data=final_data.drop(['ID'],axis=1)

final_data=final_data.drop(['Col192'],axis=1)

final_data=final_data.drop(['Col197'],axis=1)

for i in dum:

    final_data=final_data.drop([i],axis=1)
final_data.shape
columns=final_data.columns

for i in columns:

    if final_data[i].dtype=='object':

        r=final_data[i]

        count=np.sum((r == '?')+0)

        if count>0:

            final_data=final_data.drop([i],axis=1)

 
corr_matrix = final_data.corr().abs()



# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# Find features with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]



# Drop features 

final_data.drop(to_drop, axis=1, inplace=True)
from sklearn.preprocessing import MinMaxScaler



scaler=MinMaxScaler()

scaled_data=scaler.fit(final_data).transform(final_data)

scaled_df=pd.DataFrame(scaled_data,columns=final_data.columns)

scaled_df.tail()
from sklearn.manifold import TSNE



model=TSNE(n_iter=20000,n_components=2,perplexity=100)

#scaled_df.drop(columns='age')

model_data=model.fit_transform(scaled_df)

model_data.shape
from sklearn.cluster import KMeans



wcss = []

for i in range(2, 19):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(model_data)

    wcss.append(kmean.inertia_)

    

plt.plot(range(2,19),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
from sklearn.datasets import make_blobs

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_samples, silhouette_score

from sklearn import metrics

kmean = KMeans(n_clusters = 16, random_state = 0)

kmean.fit(model_data)

pred = kmean.predict(model_data)

pred_pd = pd.DataFrame(pred)

arr = pred_pd[0].unique()
arr=d.iloc[:,198].values##converting a column into an array

arr
fre=[]

for j in range(16):

    row=[]

    for i in range(5):

        row.append(0)

    fre.append(row)

for i in range(1300):

    fre[int(pred[i])][int(arr[i])-1]=fre[int(pred[i])][int(arr[i])-1]+1

fre
res=[]

for i in range(0,13000):

    if pred[i]==0:

        res.append(5)

    elif pred[i]==1:

        res.append(5)

    elif pred[i]==2:

        res.append(1)

    elif pred[i]==3:

        res.append(1)

    elif pred[i]==4:

        res.append(5)

    elif pred[i]==5:

        res.append(5)

    elif pred[i]==6:

        res.append(2)  

    elif pred[i]==7:

        res.append(5)

    elif pred[i]==8:

        res.append(4)

    elif pred[i]==9:

        res.append(4)

    elif pred[i]==10:

        res.append(5)

    elif pred[i]==11:

        res.append(3)

    elif pred[i]==12:

        res.append(1)

        

    elif pred[i]==13:

        res.append(1)

    elif pred[i]==14:

        res.append(1)    

    else:

        res.append(3)
res1 = pd.DataFrame(res)

te=[]

te=d['ID']

res1.insert(0,"ID",te,True)

res1 = res1.rename(columns={0: "Class"})

res1.head()
res1.drop(res1.index[:1300], inplace=True)
res1.to_csv('submission1.csv', index = False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"): 

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html     =     '<a     download="{filename}"     href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(res1)