!pip install --user numpy

!pip install --user pandas

!pip install --user matplotlib

!pip install --user seaborn

!pip install --user sklearn



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans, Birch,AgglomerativeClustering
df=pd.read_csv(r'../input/dmassign1/data.csv',low_memory=False)
df.head()
X=df.drop(['Class','ID'],axis=1)

Y=df['Class']
X = X.applymap(lambda s:s.upper() if type(s) == str else s)
X.head()
pd.get_dummies(data=X, columns=['Col189', 'Col190','Col191','Col192','Col193','Col194','Col195','Col196','Col197'])
for i in X.select_dtypes(object).columns:

    X[i]=X[i].replace({'?':np.nan})
X.head()
X['Col197'].replace({'M.E.':'ME'},inplace=True)

X=pd.get_dummies(data=X, columns=['Col189', 'Col190','Col191','Col192','Col193','Col194','Col195','Col196','Col197'])
for i in X.select_dtypes(object).columns:

    X[i]=pd.to_numeric(X[i],errors='raise')
X.head()
list(X.isna().any())
X.info()
X.fillna(X.mean(),inplace=True)
X.isna().any().any()
X_new = pd.DataFrame(StandardScaler().fit_transform(X),columns = X.columns)
X_new
km=KMeans(n_clusters=5,n_jobs=-1)

birch=Birch(n_clusters=5)

ac=AgglomerativeClustering(n_clusters=5,affinity='cosine',linkage='complete')
km.fit(X_new)
birch.fit(X_new)
ac.fit(X_new)
df_km=pd.DataFrame([df['ID'],km.fit_predict(X_new)],['ID','KM']).T
df_birch=pd.DataFrame([df['ID'],birch.fit_predict(X_new)],['ID','Birch']).T
df_ac=pd.DataFrame([df['ID'],ac.fit_predict(X_new)],['ID','AC']).T
pca=PCA(n_components=5).fit_transform(X_new.iloc[:,:-37])
pca_df = pd.DataFrame(data = pca

             , columns = ['principal component 1', 'principal component 2','principal component 3', 'principal component 4','principal component 5'])
plt.figure(figsize=(8,6))

plt.scatter(pca_df['principal component 1'][:1300],pca_df['principal component 2'][:1300],c=df['Class'][:1300])
plt.figure(figsize=(8,6))

plt.scatter(pca_df['principal component 1'][:1300],pca_df['principal component 2'][:1300],c=df_km['KM'][:1300])
plt.figure(figsize=(8,6))

plt.scatter(pca_df['principal component 1'][:1300],pca_df['principal component 2'][:1300],c=df_birch['Birch'][:1300])
plt.figure(figsize=(8,6))

plt.scatter(pca_df['principal component 1'][:1300],pca_df['principal component 2'][:1300],c=df_ac['AC'][:1300])
df1=pd.DataFrame([df['ID'][:1300],df['Class'][:1300],df_km['KM'][:1300],df_birch['Birch'][:1300],df_ac['AC'][:1300]]).T
df1
for i in df1.columns[1:]:

    df1[i]=pd.to_numeric(df1[i])
df2=pd.get_dummies(data=df1,columns=['KM','Class'])
df2.corr()
df3=pd.get_dummies(data=df1,columns=['Birch','Class'])
df3.corr()
df4=pd.get_dummies(data=df1,columns=['AC','Class'])



df4.corr()
df_km['KM']=df_km['KM'].map({0:1,1:3,2:2,3:5,4:4})

#df_ac['AC']=df_ac['AC'].map({0:4,1:3,2:2,3:5,4:1})
df_km.columns=['ID','Class']

df_ac.columns=['ID','Class']
df_km[1300:].to_csv('output.csv',index=False)
#df_ac[1300:].to_csv('sub9.csv',index=False)
km = KMeans(n_clusters=15, random_state = 42, n_init = 25)

df_km=pd.DataFrame([df['ID'],km.fit_predict(X_new)],['ID','KM']).T

df1 = pd.DataFrame([df['ID'][:1300], df['Class'][:1300], df_km['KM'][:1300]]).T

mat=[[0 for _ in range(5)]for _ in range(15)]

for i in range(1300):

    c = df1['Class'].astype(int).iloc[i] - 1

    r = df1['KM'].iloc[i]

    mat[r][c] += 1
mat
l1 = df1.groupby('Class')['ID'].apply(set)

l2 = df1.groupby('KM')['ID'].apply(set)
d={0:5,1:4,2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:1,10:1,11:1,12:1,13:4,14:3}
df_km['KM']=df_km['KM'].astype(int).map(d)
df_km.columns=['ID','Class']
df_km['Class']=df_km['Class'].astype(int)
df_km.head()
df_km[1300:].to_csv('output.csv',index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(df_km[1300:])