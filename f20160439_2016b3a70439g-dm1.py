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
df = pd.read_csv("../input/dmassign1/data.csv", sep=',', na_values=["?"])
df=df.replace('?',np.NaN)
df.fillna(value=df.mean(),inplace=True)
null_columns = df.columns[df.isnull().any()]

null_columns
df.fillna(value=df.mode().loc[0],inplace=True) 
null_columns = df.columns[df.isnull().any()]

null_columns
df_onehot = df.copy()

df_onehot = pd.get_dummies(df_onehot, columns=['Col189','Col190','Col191','Col192','Col193','Col194','Col195','Col196','Col197'])  

df=df_onehot
df_train=df_onehot.head(1300)
df_train.describe(include='object')
df_train.shape

X=df.drop(['ID','Class'],axis=1)
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler(copy=True)

scaled_data=scaler.fit(X).transform(X)

scaled_df=pd.DataFrame(scaled_data,columns=X.columns)

scaled_df.head()
df.shape
y = df['Class']

y=y.astype('int')

X = scaled_df
pca = PCA().fit(X)

#Plotting the Cumulative Summation of the Explained Variance

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') #for each component

plt.title('Dataset Explained Variance')

plt.show()
X.shape
from sklearn.decomposition import PCA

pca=PCA(70)

pca.fit(X)

T1=pca.transform(X)
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree

from scipy.cluster.hierarchy import fcluster

linkage_matrix1 = linkage(T1, "ward",metric="euclidean")

ddata1 = dendrogram(linkage_matrix1,color_threshold=10)

X.shape
T1.shape
from sklearn.cluster import AgglomerativeClustering as AC

aggclus = AC(n_clusters = 30,affinity='cosine',linkage='average')

aggclus.fit(T1)

y_aggclus=aggclus.labels_



y_pred=y_aggclus

from collections import Counter 

  

def most_frequent(List):    #function for most frequent occuring in the list

    occurence_count = Counter(List) 

    return occurence_count.most_common(1)[0][0] 

 

def dist(x1, x2):           #calculates the euclidean distance

    tot = 0

    for (i, j) in zip(x1,x2):

        tot += (i-j)**2

    return tot

 

def getNearestPt(centroids, x):   #find the nearest point between all the centroids and the data point

    minDist = 0

    for i in range(len(centroids)):

        if dist(x,centroids[minDist]) > dist(x,centroids[i]):

            minDist = i

    return minDist

 

def assignLabels(centroids, df, outputColName):  #for assigning labels

    mapping = [[] for i in range(len(centroids))]

    dfList = df.drop(outputColName, axis=1).values.tolist()

    for i in range(len(dfList)):

        pt = getNearestPt(centroids, dfList[i])

        #print(pt)

        mapping[pt].append(df.iloc[i][outputColName])

    #print(mapping)

    centroidLabels = []

    for i in mapping:

        if len(i) > 0:

            centroidLabels.append(most_frequent(i))

        else:

            centroidLabels.append("None")

    return centroidLabels
from collections import Counter 

#mapping function for the y_pred found

#takes parameters as y_pred and n_clusters

def mapfun(n_clusters,y_pred):

    arr = {}

    for i in range(n_clusters):

        arr[i] = []



    for i, j in enumerate(y_pred[:1300]):

        arr[j].append(i)



    counts = {}

    for i in range(n_clusters):

        counts[i] = int(Counter(df['Class'].iloc[arr[i]].dropna()).most_common(1)[0][0])



    df_test = [counts.get(n, n) for n in y_pred[:1300]]

    y_pred1 = [counts.get(n, n) for n in y_pred[1300:]]

    return y_pred1
y_ans=mapfun(30,y_pred)

y_ans[0]
ID=df[['ID']]

ID=ID.tail(13000-1300)

ID.head()
ID['Class']=y_ans
ID.info()
df=ID
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