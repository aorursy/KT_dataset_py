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
from sklearn import preprocessing

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler,LabelEncoder,RobustScaler

%matplotlib inline

from IPython.display import HTML

import base64

import warnings

import operator

from sklearn.cluster import DBSCAN,KMeans,AgglomerativeClustering

from collections import defaultdict,Counter

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score,pairwise_distances
import pandas as pd

import numpy as np
def create_download_link(df, title = "Download CSV file",count=[0]):

    count[0] = count[0]+1

    filename = "data"+str(count[0])+".csv"

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)
df = pd.read_csv("/kaggle/input/dmassign1/data.csv")

df_test = df.iloc[1300:]

df_train = df.iloc[:1300]
df_train.head()
class_labels={}

class_id = {}

for i,j in df_train.iterrows():

    class_labels[j["ID"]] = j["Class"]

    class_id[i]=j["ID"]

labels = df_train["Class"]

df_train1 = df_train.drop(["ID","Class"],axis=1)

df_test1 = df_test.drop(["ID","Class"],axis=1)
df_full = df_train1.append(df_test1)

df_full.replace("?",np.nan,inplace=True)
df_full.info(verbose=True,null_counts=True)
a = df_full.nunique()

print(a)

a = dict(a)

for i,j in a.items():

    a[i] = [j,df_full[i].dtype]

for i,j in a.items():

    t,q = j

    if q == np.dtype('O'): 

        #print(i,t,q)

        if t <100:

            print(Counter(df_full[i]))
def myconverter(string):

    if string=="me":return "me"

    elif string=="ME":return "me"

    elif string=="M.E.":return "me"

    else:

        try:

            return string.lower()

        except:

            return string
le = preprocessing.LabelEncoder()

scaler = StandardScaler()

ab=[]

for i,j in a.items():

    t,q = j

    if q == np.dtype('O'): 

        #print(i,t,q)

        if t <100:

                df_full[i] = df_full[i].apply(lambda x:myconverter(x))

                df_full[i].fillna(df_full[i].mode()[0], inplace=True)

                ab.append(i)

        else:

            df_full[i]= df_full[i].astype("float64")

            df_full[i].fillna(df_full[i].mean(),inplace=True)

            df_full[i] = scaler.fit_transform(df_full[i].values.reshape(-1,1))

    else:

            df_full[i] = scaler.fit_transform(df_full[i].values.reshape(-1,1))

    

df_full = pd.get_dummies(df_full, prefix=ab)

        

            

        
df_full.shape
df_full
modified_PCA = PCA(n_components = 0.99,svd_solver="full").fit_transform(df_full.values)
modified_PCA.shape
PCA_df = pd.DataFrame(data=modified_PCA)
PCA_df
n_clusters=144

model = KMeans(n_clusters=n_clusters,random_state=42)

pred = model.fit_predict(PCA_df.values)

cntr = Counter(pred)

mapping={}

res={0:0,1:0,2:0,3:0,4:0,5:0}

score = [defaultdict(int) for i in range(n_clusters)]

for i,j in enumerate(pred[:1300]):

    score[j][labels[i]]+=1

for j,i in enumerate(score):

    t =[(j,k) for k,j in i.items()]

    t.sort(reverse=True)

    t = [(j,k) for k,j in t]

    try:

        mapping[j]=int(t[0][0])

        res[int(t[0][0])] +=cntr[j] 

            

    except:

        random_int = 0

        mapping[j]=random_int

        res[random_int]+=cntr[j]

print(res)    
PCA_df.shape
PCA_df["pred"] = pred
def mapper(x,mapping):

    y = mapping[x]

    if y==0:return np.random.randint(1,6)

    else:return y
PCA_df["pred"]=PCA_df["pred"].apply(lambda x:mapper(x,mapping))
Counter(PCA_df["pred"])
df_submission = PCA_df.iloc[1300:]
predictions = df_submission["pred"]
submission =  pd.DataFrame(data={"ID": df["ID"][1300:], "Class": predictions })
Counter(submission["Class"])
create_download_link(submission)
submission.to_csv("Final.csv",index=False)
submission.head()