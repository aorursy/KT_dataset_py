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
# from google.colab import files

# files.upload()
# data = pd.read_csv("data.csv", sep=",") #dataframe object

data = pd.read_csv("/kaggle/input/dmassign1/data.csv",sep=",")
data.info()
df = data.copy()
df.drop(['ID','Class'], axis = 1, inplace=True)
df.replace('?',np.nan,True)
null_columns = df.columns[df.isnull().any()]

null_columns
df = df.apply(pd.to_numeric, errors='ignore', downcast = 'float')
df.fillna(df.mean(), inplace=True)
null_columns = df.columns[df.isnull().any()]

print(null_columns)
for col in null_columns:

  df[col].fillna(df[col].mode()[0],inplace=True)
df.info()
df.select_dtypes(include=['object'])
df_onehot = df.copy()

df_onehot = pd.get_dummies(df_onehot, columns=["Col189","Col190","Col191","Col192","Col193","Col194","Col195","Col196","Col197"])

df_onehot.info()
corr = df_onehot.corr()

fig, ax = plt.subplots(figsize=(20,10))

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

square=True, ax=ax, annot = False)
corr_matrix = df_onehot.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]

df1 = df_onehot.drop(df_onehot[to_drop], axis=1)



df1.info()
corr = df1.corr()

fig, ax = plt.subplots(figsize=(20,10))

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

square=True, ax=ax, annot = False)
df1.describe()
df = df_onehot
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler



scaler=StandardScaler()

scaled_data=scaler.fit(df).transform(df)

scaled_df=pd.DataFrame(scaled_data,columns=df.columns)
import scipy.cluster.hierarchy as shc

plt.figure(figsize =(8, 8)) 

plt.title('Visualising the data') 

Dendrogram = shc.dendrogram((shc.linkage(scaled_df, method ='ward'))) 
from sklearn.cluster import AgglomerativeClustering as AC

acs = []

for i in range(2,13):

  aggclus = AC(n_clusters = i*5,affinity='euclidean',linkage='ward',compute_full_tree='auto')

  acs.append(aggclus)
from sklearn.cluster import AgglomerativeClustering as AC

aggclus = AC(n_clusters = 40,affinity='euclidean',linkage='ward',compute_full_tree='auto')

y_aggclus= aggclus.fit_predict(scaled_df)
pred = y_aggclus
pred
num_clstr=40

clusters = np.array(pred)

orig = data[['Class']].loc[0:1299]



cluster_frequency = np.zeros((num_clstr,6))



for i in range(0,1300):

  cluster_frequency[pred[i]][int(orig['Class'].loc[i])]+=1



print(cluster_frequency)



cluster_mapping = []

for i in range(0,num_clstr):

  cluster_mapping.append(np.argmax(cluster_frequency[i]))

print(cluster_mapping)



for i in range(0,13000):

  pred[i] = cluster_mapping[pred[i]]

# print(pred[12999])
res = pd.DataFrame(columns=['ID','Class'])

res['Class'] = pred[1300:]

ids = []

for i in range(1300,13000):

  ids.append("id"+str(i))

res['ID'] = ids

res
# from google.colab import files

res.to_csv('2017A7PS0939G_sub.csv',index=False)

# files.download('final.csv')
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

create_download_link(res)