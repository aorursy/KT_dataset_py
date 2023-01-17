import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
df = pd.read_csv('/kaggle/input/dmassign1/data.csv')

df1= df
df
for i in range(1,178):

    if (df['Col'+str(i)]=='?').sum() > 0:

        df['Col'+str(i)] = df['Col'+str(i)].replace('?',np.nan)

        m = df['Col'+str(i)].astype('float32').mean()

        df['Col'+str(i)].fillna(int(m),inplace=True)

        df['Col'+str(i)] = df['Col'+str(i)].astype('int64')





for i in range(179,188):

    if (df['Col'+str(i)]=='?').sum() > 0:

        df['Col'+str(i)] = df['Col'+str(i)].replace('?',np.nan)

        m = df['Col'+str(i)].astype('float64').mean()

        df['Col'+str(i)].fillna(m,inplace=True)

        df['Col'+str(i)] = df['Col'+str(i)].astype('float64')





df['Col197'].replace('sm','SM',inplace=True)

df['Col197'].replace('me','ME',inplace=True)

df['Col197'].replace('M.E.','ME',inplace=True)

df['Col197'].replace('la','LA',inplace=True)



for i in range(189,198):

    if (df['Col'+str(i)]=='?').sum() > 0:

        df['Col'+str(i)] = df['Col'+str(i)].replace('?',np.nan)

        df['Col'+str(i)].fillna(df['Col'+str(i)].mode()[0], inplace=True)

    X_encoded = pd.get_dummies(df['Col'+str(i)],prefix = 'Col'+str(i))

    df = pd.concat([df,pd.DataFrame(X_encoded)],axis=1)

    df.drop(columns = ['Col'+str(i)],axis=1,inplace=True)
cols = []

for i in range(1,179):

    cols.append('Col'+str(i))

df = df[cols]



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_scaled = scaler.fit_transform(df)

df = pd.DataFrame(x_scaled)
from sklearn.manifold import TSNE

X_embedded = TSNE(n_components=2,n_iter=10000,verbose = 2).fit_transform(df)

df_x = pd.DataFrame(X_embedded)

df_x
ans = {}

from sklearn.cluster import KMeans

for k in range(1,10):

    print(k)

    clf =  KMeans(n_clusters=48)

    clf.fit(df_x)

    clf.cluster_centers_

    clf.labels_



    mydict1 = {i: np.where(clf.labels_ == i)[0] for i in range(clf.n_clusters)}



    total = [0]*5

    for i in range(0,len(mydict1)):

        arr = [0]*5

        for j in range(len(mydict1[i])):

            if(mydict1[i][j]<1300):

                arr[int(df1['Class'][mydict1[i][j]])-1] = arr[int(df1['Class'][mydict1[i][j]])-1] +1

        val = np.argmax(arr)+1

        if(arr[val-1]==0):

            val = 4

        for j in range(len(mydict1[i])):

            if(mydict1[i][j]>=1300):

                if(k==1):

                   ans[df1['ID'][mydict1[i][j]]] = [val]

                else:

                    ans[df1['ID'][mydict1[i][j]]].append(val)

                total[val-1] = total[val-1] + 1

     #   print(i)

     #   print(arr," ", sum (arr))

    print(total) 
a = []

for i in range(1300,13000):

    mode = max(set(ans[df1['ID'][i]]), key=ans[df1['ID'][i]].count)

    a.append((df1['ID'][i],mode))

a = pd.DataFrame(a)

a['ID'] = a[0]

a['Class'] = a[1]

a = a.drop(columns=[0,1],axis=1)

a.to_csv('Fianl_sub.csv',index = False)
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

create_download_link(a)