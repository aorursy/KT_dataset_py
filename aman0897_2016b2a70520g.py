import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/dmassign1/data.csv")
df.info()
df.head()
null_columns = df.columns[df.isnull().any()]

null_columns
y = df['Class']

df=df.drop('Class',axis=1)

df=df.drop('ID',axis=1)

df=df.drop('Col189',axis=1)

df=df.drop('Col190',axis=1)

df=df.drop('Col191',axis=1)

df=df.drop('Col192',axis=1)

df=df.drop('Col193',axis=1)

df=df.drop('Col194',axis=1)

df=df.drop('Col195',axis=1)

df=df.drop('Col196',axis=1)

df=df.drop('Col197',axis=1)
df.head()
df.info()
col = df.columns

for i in col:

    df[i]=df[i].replace('?','0')
df=df.astype(float)
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(df)

scaled_df=pd.DataFrame(np_scaled)
scaled_df.tail()
#model=TSNE(learning_rate=1000)

from sklearn.manifold import TSNE

model=TSNE(n_iter=20000,n_components=2,perplexity=100)

model_data=model.fit_transform(scaled_df)

model_data.shape
colors = ['red','green','blue','yellow','purple','pink','palegreen','violet','cyan','black','orange','brown','gray','crimson','indigo','peru','firebrick','khaki']
from sklearn.cluster import KMeans

kmean = KMeans(n_clusters = 48, random_state = 60)

kmean.fit(model_data)

pred = kmean.predict(model_data)
y=y.dropna()

y = y.astype(int)
print(y)
print(pred)
uniqueElements, countsElements = np.unique(pred, return_counts=True)
print(uniqueElements)
print(countsElements)
freq = []

for j in range(48):

  row=[]

  for i in range(5):

    row.append(0)

  freq.append(row)

for i in range(len(y)):

  freq[pred[i]][y[i]-1]=freq[pred[i]][y[i]-1]+1
print(freq)
mapping = []

for i in range(48):

    maxind=0

    maxval=-1

    for j in range(5):

        if(freq[i][j]>maxval):

            maxind=j

            maxval=freq[i][j]

    mapping.append(maxind)

        
print(mapping)
len(pred)
f_preds = []

for i in range(1300,len(pred)):

  s = 'id'+str(i)

  f_preds.append([s,mapping[pred[i]]+1])
print(f_preds)
final_csv = pd.DataFrame(f_preds,columns=['ID','Class'])
print(final_csv)
final_csv.to_csv('2016B2A70520.csv',index=False)
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

create_download_link(final_csv)