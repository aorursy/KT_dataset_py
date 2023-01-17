# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
orig_data = pd.read_csv("/kaggle/input/dmassign1/data.csv", sep=',')

data = pd.read_csv("/kaggle/input/dmassign1/data.csv", sep=',',index_col=0,na_values='?')

data
data.describe()
pd.set_option('display.max_rows', 199)

data.isnull().sum(axis = 0)
null_columns = data.columns[data.isnull().any()]

null_columns = null_columns[:-1]

null_columns
class_df = data.iloc[:,-1]

df = data.iloc[0:,1:-1]

df.head()
for cols in null_columns:

  print(cols,df[cols].isnull().sum())
obj_cols = ['Col192','Col193','Col194','Col195','Col196','Col197']

null_columns2 = [x for x in null_columns if x not in obj_cols]

null_columns2
#How to handle null values

df[null_columns2] = df[null_columns2].fillna(df[null_columns2].median(), inplace=False)

# df[obj_cols] = df[obj_cols].fillna(df[obj_cols].mode(),inplace=False)

pd.set_option('display.max_rows', 198)

df.isnull().sum(axis = 0)
df["Col192"].value_counts()

df["Col192"].fillna("p2",inplace=True)

df["Col193"].mode()

df["Col193"].fillna("F0",inplace=True)

df["Col194"].mode()

df["Col194"].fillna("ad",inplace=True)

df["Col195"].mode()

df["Col195"].fillna("Jb3",inplace=True)

df["Col196"].mode()

df["Col196"].fillna("H3",inplace=True)

df["Col197"].mode()

df["Col197"].fillna("XL",inplace=True)

pd.set_option('display.max_rows', 198)

df.isnull().sum(axis = 0)
df2 = df.drop_duplicates()

df2
obj_cols1 = ['Col189','Col190','Col191','Col192','Col193','Col194','Col195','Col196','Col197']
df_onehot2 = pd.get_dummies(df2,columns=obj_cols1,prefix=obj_cols1) 

df_onehot = pd.get_dummies(df,columns=obj_cols1,prefix=obj_cols1)

df_onehot2

df2
from sklearn.preprocessing import StandardScaler



scaler=StandardScaler()

scaled_data2=scaler.fit_transform(df_onehot2)

scaled_df2=pd.DataFrame(scaled_data2,columns=df_onehot2.columns)

scaled_data = scaler.transform(df_onehot)

scaled_df = pd.DataFrame(scaled_data,columns=df_onehot.columns)

scaled_df2

scaled_df
#on z-score normalize  #1,2,3

from sklearn.decomposition import PCA

pca = PCA().fit(scaled_df2)

#Plotting the Cumulative Summation of the Explained Variance

plt.figure()

plt.xticks(np.arange(0, 200, 10))

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') #for each component

plt.title('Pulsar Dataset Explained Variance')

# plt.figure(figsize=(200,100))

plt.grid(color='r', linestyle='-', linewidth=2)

plt.show()
model=PCA(n_components=80)

model_data2=model.fit_transform(scaled_df2)

model_data = model.transform(scaled_df)

np.shape(model_data)
dataN2 = scaled_df2

T2 = model_data2

pca2 = model
# Accuracy method to the optimal number of clusters

from sklearn.cluster import AgglomerativeClustering as AC

from sklearn.metrics import confusion_matrix 

acc_list = []

for i in range(2,50):

  print(i)

  aggclus = AC(n_clusters = i,affinity='cosine',linkage='average',compute_full_tree='auto')

  y_aggclus = aggclus.fit_predict(model_data)

  # plt.scatter(model_data[:, 0], model_data[:, 1], c=y_aggclus)

  pred = y_aggclus

  predictions = pd.Series(pred+1,index=data.index,dtype = np.float64)

  classes = (confusion_matrix(class_df[:1300],predictions[:1300]).argmax(axis=0)+1).astype(np.float64)

  print(np.unique(classes))

  predictions.replace({cluster+1:classes[cluster] for cluster in range(0,len(classes))},inplace=True)

  predictions = predictions.astype(int)

  # print(predictions.value_counts())

  acc_cnt = 0

  for i in range(1300):

    if(predictions[i]==class_df[i]):

      acc_cnt = acc_cnt + 1

  print(acc_cnt)

  acc_list.append(acc_cnt/1300)
from sklearn.cluster import AgglomerativeClustering as AC

aggclus = AC(n_clusters = 25,affinity='cosine',linkage='average',compute_full_tree='auto')

y_aggclus = aggclus.fit_predict(model_data)

# plt.scatter(model_data[:, 0], model_data[:, 1], c=y_aggclus)

pred = y_aggclus
from sklearn.metrics import confusion_matrix 

predictions = pd.Series(pred+1,index=orig_data.index,dtype = np.float64)

classes = (confusion_matrix(class_df[:1300],predictions[:1300]).argmax(axis=0)+1).astype(np.float64)

predictions.replace({cluster+1:classes[cluster] for cluster in range(0,len(classes))},inplace=True)
unique,count = np.unique(predictions,return_counts = True)

unique,count
res1 = pd.DataFrame(predictions.astype(int))

final = pd.concat([orig_data["ID"], res1], axis=1).reindex()

final = final.rename(columns={0: "Class"})

final.head()

final[1300:]
final[1300:].to_csv("dataset.csv", index = False)
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

create_download_link(final[1300:])