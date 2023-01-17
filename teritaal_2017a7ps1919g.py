def write_to_csv(csv_name,labels,startindex=1300):
    with open(csv_name,"w") as f:
        f.write("id,Class\n")
        for ind, i in enumerate(labels):
            string = "id"+str(ind+startindex)+","+str(i)+"\n"
            f.write(string)
def label_clusters(pred_array):
    counter=[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]] # each element represents cluster.
    for i in range(0,1300):
        try:
            counter[pred_array[i]][Y_true[i]-1]+=1
        except IndexError:
            print("IE ",i)
            break
    print(counter)
    for ind,i in enumerate(counter):
        maxind = i.index(max(i))
        counter[ind]=maxind+1
    return counter
#label all predictions.
def label_data(cluster_labels, preds, start_index=1300,total_length=13000):
    if len(preds)!= total_length:
        print("Size mismatch. ",total_length," vs ",len(preds),". Returning None")
        return None
    labels=[]
    for i in range(start_index,total_length):
        try:
            labels.append(cluster_labels[preds[i]])
        except IndexError:
            print(preds[i])
            break
    labels = np.asarray(labels)
    return labels
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Any results you write to the current directory are saved as output.
from sklearn import preprocessing
from sklearn.cluster import Birch
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering as AC
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer
path = '../input/dmassign1/data.csv'
data_og = pd.read_csv(path)
data_cp = data_og
Y_true=data_cp['Class'].to_numpy()
data_cp = pd.get_dummies(data_cp,columns=['Col189','Col190','Col191','Col192','Col193','Col194','Col195','Col196','Col197'])

X = data_cp.to_numpy()
X = X[:,1:-2]
a=np.where(X=='?')
for i in range(len(a[0])):
    if X[a[0][i]][a[1][i]]=='?':
        X[a[0][i]][a[1][i]]=0
X = X.astype('float64')

Y_true = Y_true.astype(int)
Y_true = Y_true.reshape(Y_true.shape[0])
Y_true = Y_true[:1300]
scl = preprocessing.StandardScaler()
scl.fit(X)
X=scl.transform(X)
my_imputer = SimpleImputer()
X = my_imputer.fit_transform(X)
pca1 = PCA(n_components=100).fit_transform(X)
#aggclus = AC(n_clusters=10,affinity='cosine',linkage='complete')
brch = Birch(n_clusters=10,threshold=.05,branching_factor=25)
#km = KMeans(n_clusters=5)
#y_k=km.fit_predict(X)
#y_a=aggclus.fit_predict(pca1)
y_b=brch.fit_predict(pca1)
#cluster_labels_agg = label_clusters(y_a)
#cluster_labels_km = label_clusters(y_k)
cluster_labels_brch = label_clusters(y_b)
#labels_agg = label_data(cluster_labels_agg,y_a)
labels_brch = label_data(cluster_labels_brch, y_b)
#labels_km = label_data(cluster_labels_km, y_k)
# write_to_csv('km.csv',labels_km)
#write_to_csv('ag.csv',labels_agg)
write_to_csv('br.csv',labels_brch)
df = pd.read_csv('br.csv')
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
create_download_link(df)