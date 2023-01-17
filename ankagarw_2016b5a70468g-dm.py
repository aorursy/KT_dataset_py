import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



X = pd.read_csv("../input/dmassign1/data.csv",na_values=['?'],sep=',',index_col="ID")  #read csv

X, y_true = X.drop(columns="Class"), X["Class"]

IDs = X.index
cols = X

numerical = cols.select_dtypes(include=["number"]).columns

categorical = cols.select_dtypes(exclude=["number"]).columns
from sklearn.preprocessing import RobustScaler

from sklearn.decomposition import PCA

for col in categorical:

  X[col]=X[col].str.lower()

X['Col197'].replace({'m.e.':'me'},inplace=True)

X = pd.get_dummies(X)

X = X.fillna(X.mean())

X.loc[:, "Col1":"Col188"] = RobustScaler().fit_transform(X.loc[:, "Col1":"Col188"])

X = PCA(n_components=20).fit_transform(X)
from sklearn.cluster import Birch

birch = Birch(n_clusters=200, threshold=0.1, branching_factor=100).fit(X)

y = pd.Series(birch.predict(X))

def mapping(y_pred,y_true):

  temp=np.array(y_true)

  ans={}

  for i in set(y_pred[0:1300]):

    temp3={}

    indx=pd.DataFrame(y_pred[0:1300]).index[pd.DataFrame(y_pred[0:1300])[0]==i]

    temp2=temp[indx]

    for j in temp2:

      if j in temp3.keys():

        temp3[j]+=1

      else:

        temp3[j]=1

    for j in temp3.keys():

      if temp3[j]==max(temp3.values()):

        ans[i]=np.round(j)

        break

  return ans
map1 = mapping(y[:1300], y_true[:1300])

for i in range(200):

  if i not in map1.keys():

    map1[i] = 2

np.mean(np.array(y[:1300].replace(map1).astype(int))==np.array((y_true[:1300]).astype(int)))
ans=pd.DataFrame(data={'ID':IDs[1300:],'Class':y[1300:].replace(map1).astype(int)})
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

 csv = ans.to_csv(index=False)

 b64 = base64.b64encode(csv.encode())

 payload = b64.decode()

 html = '<a download="sub.csv" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

 html = html.format(payload=payload,title=title,filename=filename)

 return HTML(html)

create_download_link("sub.csv")