import sys

!{sys.executable} -m pip install sklearn

import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn import svm
#import data from csv to dataframe df

df = pd.read_csv("../input/dmassign1/data.csv", sep=",") #dataframe object

df_new = pd.read_csv("../input/dmassign1/data.csv", sep=",") #dataframe object

df.info()
df=df.drop(['Class'], axis=1)

df = df.replace({"?": np.nan})

df=df.fillna(df.mode())
df.fillna(value=df.mode().loc[0],inplace=True)  

df.head(20)
one_hot=['Col189','Col190','Col191','Col192','Col193','Col194','Col195','Col196','Col197']

for i in one_hot:

    df = pd.get_dummies(df, columns=[i],prefix=[i])

df=df.drop('ID',axis=1)

for i in df.columns:

    df[i]=df[i].astype('float64')

df.head()
# StandardScaler



scaler=StandardScaler()

scaled_data=scaler.fit(df).transform(df)

scaled_df=pd.DataFrame(scaled_data,columns=df.columns)

scaled_df.tail()
from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=30, affinity="cosine", linkage="average")  

cluster.fit_predict(scaled_df)

labels=cluster.labels_
labels=labels+1

labels
rows, cols = (31, 6) 

arr = [[0 for i in range(cols)] for j in range(rows)] 
clabel=df_new['Class']

cl=clabel[0:1300]

l=labels[0:1300]

for i in range(1300):

    a=l[i].astype(int)

    b=cl[i].astype(int)

    arr[a][b]=arr[a][b]+1
cl
arr
rows, cols = (31,1) 

r = [[0 for i in range(cols)] for j in range(rows)] 

for j in range(1,31):

    max=0

    for i in range(1,6):

        if arr[j][i]>=max:

            max=arr[j][i]

            r[j]=i   
i=0

for i in range(13000):

    labels[i]=labels[i]*10; 

labels[0:100]
i=0

for i in range(13000):

    labels[i]=r[(labels[i]/10).astype(int)]

labels
labels
ID=df_new['ID']

sol1=pd.DataFrame(data=labels)

sol2=pd.DataFrame(data=ID)
sol=pd.concat([sol2,sol1],axis=1)

sol.rename(columns = {0:'Class'}, inplace = True) 
ans=sol[1300:13000]

ans
s=0

for i in range(1300):

    if labels[i]==cl[i]:

        s=s+1
k=s/1300.0

k
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

create_download_link(ans)