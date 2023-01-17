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
!pip install pandas-profiling
from sklearn import preprocessing

import pandas_profiling

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler,LabelEncoder

%matplotlib inline

from IPython.display import HTML

import base64

import warnings

import operator

from sklearn.cluster import DBSCAN,KMeans,Birch,SpectralClustering,AgglomerativeClustering,MeanShift

from collections import defaultdict,Counter

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis

from sklearn.metrics import roc_auc_score,accuracy_score,pairwise_distances
def create_download_link(df, title = "Download CSV file",count=[0]):

    count[0] = count[0]+1

    filename = "data"+str(count[0])+".csv"

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)
class NearestCentroid():

    def __init__(self, metric='euclidean'):

        self.metric = metric

    def fit(self, X, y):

        n_samples, n_features = X.shape

        le = LabelEncoder()

        y_ind = le.fit_transform(y)

        self.classes_ = classes = le.classes_

        n_classes = classes.size

        self.centroids_ = np.empty((n_classes, n_features), dtype=np.float64)

        for cur_class in range(n_classes):

            center_mask = y_ind == cur_class

            self.centroids_[cur_class] = np.median(X[center_mask], axis=0)

    def predict(self, X):

        return self.classes_[pairwise_distances(X, self.centroids_, metric=self.metric).argmin(axis=1)]
df = pd.read_csv("/kaggle/input/eval-lab-3-f464/train.csv")
df.drop(["custId"],axis=1,inplace=True)

df.drop_duplicates(inplace=True)

df.reset_index(drop=True,inplace = True)

def converter(a):

    try:

        return(float(a))

    except:

        return 0.0

df["TotalCharges"]=df["TotalCharges"].apply(lambda x:converter(x))
le = preprocessing.LabelEncoder()

df["gender"]=le.fit_transform(df["gender"])

le1 = preprocessing.LabelEncoder()

df["Married"]=le1.fit_transform(df["Married"])

le2 = preprocessing.LabelEncoder()

df["Children"]=le2.fit_transform(df["Children"])

le3 = preprocessing.LabelEncoder()

df["Internet"] = le3.fit_transform(df["Internet"])

le4 = preprocessing.LabelEncoder()

df["AddedServices"] = le4.fit_transform(df["AddedServices"])

le5 = preprocessing.LabelEncoder()

df["Subscription"] = le5.fit_transform(df["Subscription"])

le6 = preprocessing.LabelEncoder()

df["PaymentMethod"] = le6.fit_transform(df["PaymentMethod"])
d={"Cable":0,"DTH":1,"No":2}

df["TVConnection"]=df["TVConnection"].apply( lambda x:d[x])

g={"Yes":0,"No":1,"No tv connection":2}

channels = ["Channel1","Channel2","Channel3","Channel4","Channel5","Channel6"]

for channel in channels:

    df[channel] = df[channel].apply(lambda x: g[x])

h={"Yes":0,"No":1,"No internet":2}

df["HighSpeed"] = df["HighSpeed"].apply(lambda x:h[x])
df.profile_report(style={'full_width':True})
df_onlynet = df[(df['TVConnection'] == 2) & (df['Internet'] == 1)]

df_onlytv = df[df['HighSpeed'] == 2 ]

df_both = df[(df['Internet'] == 1) & (df['TVConnection'] != 2)]
df_onlynet.drop(['TVConnection','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','Internet'],axis=1,inplace=True)

df_onlytv.drop(['Internet','HighSpeed','TVConnection'],axis=1,inplace=True)

df_both.drop(['Internet'],axis=1,inplace=True)

df_onlytv.reset_index(drop=True,inplace = True)

df_onlynet.reset_index(drop=True,inplace = True)

df_both.reset_index(drop=True,inplace = True)
df_onlynet.profile_report(style={'full_width':True})
df_onlytv.profile_report(style={'full_width':True})
df_both.profile_report(style={'full_width':True})
x_b = df_both.drop(["Satisfied"],axis=1).to_numpy()

y_b=df_both["Satisfied"].to_numpy()

sc1 = StandardScaler()

#x_b = sc1.fit_transform(x_b)

lda_b = LinearDiscriminantAnalysis(n_components=1)

x_b = lda_b.fit(x_b,y_b).transform(x_b)

#x_train, x_test, y_train, y_test = train_test_split(x_b, y_b,test_size=0.2)

model_b = KMeans(n_clusters=2)

model_b.fit(x_b)

pred1_b = model_b.predict(x_b)

# model_b = NearestCentroid()

# model_b.fit(x_b,pred1_b)

#pred2_b = model_b.predict(x_test)

mapping_b=defaultdict(list)

for i in range(len(pred1_b)):mapping_b[pred1_b[i]].append(y_b[i])

for i in mapping_b:

    mapping_b[i]=Counter(mapping_b[i])

print(mapping_b)

for i in mapping_b:

    mapping_b[i] = 1 if mapping_b[i][1] > mapping_b[i][0] else 0
mapping_b[0]=1#cluster-labelling based on manual inspection of true class concentration  of the cluster

mapping_b[1]=0

mapping_b[2]=0

mapping_b[3]=1



preds = [mapping_b[i] for i in pred1_b]

print(roc_auc_score(preds,y_b))

# preds = [mapping_b[i] for i in pred2_b]

# print(roc_auc_score(preds,y_test))
x_n = df_onlynet.drop(["Satisfied"],axis=1).to_numpy()

y_n=df_onlynet["Satisfied"].to_numpy()

sc2 = StandardScaler()

#x_n = sc2.fit_transform(x_n)

lda_n = LinearDiscriminantAnalysis(n_components=1)

x_n = lda_n.fit(x_n,y_n).transform(x_n)

x_train, x_test, y_train, y_test = train_test_split(x_n, y_n,test_size=0.2)

model_n = KMeans(n_clusters=5)

model_n.fit(x_n)

pred1_n = model_n.predict(x_n)

# model_n = NearestCentroid()

# model_n.fit(x_n,pred1_n)

mapping_n=defaultdict(list)

for i in range(len(pred1_n)):mapping_n[pred1_n[i]].append(y_n[i])

for i in mapping_n:

    mapping_n[i]=Counter(mapping_n[i])

d=[]

print(mapping_n)

for i in mapping_n:

    mapping_n[i] = 1 if mapping_n[i][1] > mapping_n[i][0] else 0
for i in range(5):

    mapping_n[i] = 1 if i not in [1] else 0 #cluster-labelling based on manual inspection of true classes  of the cluster

preds = [mapping_n[i] for i in pred1_n]

print(roc_auc_score(preds,y_n))

x_t = df_onlytv.drop(["Satisfied"],axis=1).to_numpy()

y_t=df_onlytv["Satisfied"].to_numpy()

sc3 = StandardScaler()

#x_t = sc3.fit_transform(x_t)

lda_t = LinearDiscriminantAnalysis(n_components=3)

x_t = lda_t.fit(x_t,y_t).transform(x_t)

x_train, x_test, y_train, y_test = train_test_split(x_t, y_t,test_size=0.2)

model_t = KMeans(n_clusters=5)

model_t.fit(x_t)

pred1_t = model_t.predict(x_t)

# model_t = NearestCentroid()

# model_t.fit(x_t,pred1_t)

mapping_t=defaultdict(list)

for i in range(len(pred1_t)):mapping_t[pred1_t[i]].append(y_t[i])

for i in mapping_t:

    mapping_t[i]=Counter(mapping_t[i])

print(mapping_t)

for i in mapping_t:

    mapping_t[i] = 1 if mapping_t[i][1] > mapping_t[i][0] else 0
for i in range(3):

    mapping_t[i] = 1 if i not in [4] else 0 #cluster-labelling based on manual inspection of true classes  of the cluster

preds = [mapping_t[i] for i in pred1_t]

print(roc_auc_score(preds,y_t))
results={}
tdf = pd.read_csv("/kaggle/input/eval-lab-3-f464/test.csv")

ids = tdf["custId"]

tdf["TotalCharges"]=tdf["TotalCharges"].apply(lambda x:converter(x))

tdf["gender"]=le.transform(tdf["gender"])

tdf["Married"]=le1.transform(tdf["Married"])

tdf["Children"]=le2.transform(tdf["Children"])

tdf["Internet"] = le3.transform(tdf["Internet"])

tdf["AddedServices"] = le4.transform(tdf["AddedServices"])

tdf["Subscription"] = le5.transform(tdf["Subscription"])

tdf["PaymentMethod"] = le6.transform(tdf["PaymentMethod"])

d={"Cable":0,"DTH":1,"No":2}

tdf["TVConnection"]=tdf["TVConnection"].apply( lambda x:d[x])

g={"Yes":0,"No":1,"No tv connection":2}

channels = ["Channel1","Channel2","Channel3","Channel4","Channel5","Channel6"]

for channel in channels:

    tdf[channel] = tdf[channel].apply(lambda x: g[x])

h={"Yes":0,"No":1,"No internet":2}

tdf["HighSpeed"] = tdf["HighSpeed"].apply(lambda x:h[x])

tdf_onlynet = tdf[(tdf['TVConnection'] == 2) & (tdf['Internet'] == 1)]

tdf_onlytv = tdf[tdf['HighSpeed'] == 2 ]

tdf_both = tdf[(tdf['Internet'] == 1) & (tdf['TVConnection'] != 2)]
tdf_onlynet.drop(['TVConnection','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','Internet'],axis=1,inplace=True)

tdf_onlytv.drop(['Internet','HighSpeed','TVConnection'],axis=1,inplace=True)

tdf_both.drop(['Internet'],axis=1,inplace=True)

tdf_onlytv.reset_index(drop=True,inplace = True)

tdf_onlynet.reset_index(drop=True,inplace = True)

tdf_both.reset_index(drop=True,inplace = True)
id_b = tdf_both["custId"]

x_b = tdf_both.drop(["custId"],axis=1).to_numpy()

#x_b = sc1.transform(x_b)

x_b = lda_b.transform(x_b)

pred1_b = model_b.predict(x_b)

preds = [mapping_b[i] for i in pred1_b]

for i in range(len(preds)):

    results[id_b[i]] = preds[i]
id_n = tdf_onlynet["custId"]

x_n = tdf_onlynet.drop(["custId"],axis=1).to_numpy()

#x_n = sc2.transform(x_n)

x_n = lda_n.transform(x_n)

pred1_n = model_n.predict(x_n)

preds = [1 for i in pred1_n]

print(len(preds))

for i in range(len(preds)):

    results[id_n[i]] = preds[i]
id_t = tdf_onlytv["custId"]

x_t = tdf_onlytv.drop(["custId"],axis=1).to_numpy()

#x_t = sc3.transform(x_t)

x_t = lda_t.transform(x_t)

pred1_t = model_t.predict(x_t)

preds = [mapping_t[i] for i in pred1_t]

print(len(preds))

for i in range(len(preds)):

    results[id_t[i]] = preds[i]
len(results)
ssdf = pd.read_csv("/kaggle/input/eval-lab-3-f464/test.csv")

ssdf["Satisfied"] = ssdf["custId"].apply(lambda x: results[x])

sdf = ssdf[["custId","Satisfied"]]

create_download_link(sdf)