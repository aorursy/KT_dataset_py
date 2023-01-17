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

import seaborn as sns
#import the file



location = r"/kaggle/input/dmassign1/data.csv"

df=pd.read_csv(location, sep = ",")
df.head()
# drop columns 'Class' and 'ID'

X = df.drop(['Class', 'ID'], axis = 1)

Y = df['Class']
X.head()
# convert string in columns to lower case

X = X.applymap(lambda s:s.lower() if type(s) == str else s)
X.head()
X.info()
# columns with non numeric values are column numbers 189 to 197 (observation)

columns=['Col189', 'Col190','Col191','Col192','Col193','Col194','Col195','Col196','Col197']

#pd.get_dummies(data=X, columns)
for c in columns:

    print(c, X[c].unique())
# '?' indicates missing values

# replace '?' with np.nan



for i in X.select_dtypes(object).columns:

    X[i]=X[i].replace({'?':np.nan})
X.head()
# col197 has entries m.e. and me. Assuming that they are same



X['Col197'].replace({'M.E.':'ME'},inplace=True)
#drop the column 192



X = X.drop(['Col192'], axis = 1)
# one hot encoding

#X=pd.get_dummies(data=X, columns=['Col189', 'Col190','Col191','Col192','Col193','Col194','Col195','Col196','Col197'])

X=pd.get_dummies(data=X, columns=['Col189', 'Col190','Col191','Col193','Col194','Col195','Col196','Col197'])
# convert all the object type columns to numeric type



for i in X.select_dtypes(object).columns:

    X[i]=pd.to_numeric(X[i],errors='raise')
X.info()
# Replace nan with median or mean of the column



#X.fillna(X.mean(),inplace=True)

X.fillna(X.median(),inplace=True)
X.isna().any().any()
# Normalization



from sklearn.preprocessing import StandardScaler, MinMaxScaler



X_new = pd.DataFrame(StandardScaler().fit_transform(X),columns = X.columns)

#X_new = pd.DataFrame(MinMaxScaler().fit_transform(X),columns = X.columns)
X_new.head()
#clustering algorithms used are KMeans, Birch and AgglomerativeClustering



from sklearn.cluster import KMeans

from sklearn.cluster import Birch

from sklearn.cluster import AgglomerativeClustering



km = KMeans(n_clusters=5 ,n_jobs=-1 ,random_state=42 ,n_init=25)

br = Birch(n_clusters = 5)

ac = AgglomerativeClustering(n_clusters = 5, affinity = "cosine", linkage = 'average')
no_of_components = 30
# Principle Component Analysis

from sklearn.decomposition import PCA

pca=PCA(n_components = no_of_components).fit_transform(X_new)

pca_df = pd.DataFrame(data = pca)
X_final=pd.concat(

    [X_new, pca_df], axis=1)



#Without PCA

#X_final = X_new
# X_final.iloc[:,-no_of_components:]
# Apply the clustering algoritms



df_km = pd.DataFrame([df['ID'],km.fit_predict(X_final.iloc[:,-no_of_components:])],['ID','KM']).T



df_br = pd.DataFrame([df['ID'],br.fit_predict(X_final.iloc[:,-no_of_components:])],['ID','BR']).T



df_ac = pd.DataFrame([df['ID'],ac.fit_predict(X_final.iloc[:,-no_of_components:])],['ID','AC']).T
#df1 stores the cluster numbers for all the rows for all 3 algorithms



df1=pd.DataFrame([df['ID'][:1300], df['Class'][:1300], df_km['KM'][:1300], df_br['BR'][:1300], df_ac['AC'][:1300]]).T
df1
#Data Post processing starts

#Group IDs by Class



l1=df1.groupby('Class')['ID'].apply(set)
l2_km = df1.groupby('KM')['ID'].apply(set)

l2_br = df1.groupby('BR')['ID'].apply(set)

l2_ac = df1.groupby('AC')['ID'].apply(set)
#Intersection matrices initialization

mat_km=[[0 for _ in range(5)]for _ in range(5)]

mat_br=[[0 for _ in range(5)]for _ in range(5)]

mat_ac=[[0 for _ in range(5)]for _ in range(5)]



for i in range(0,5):

    for j in range(0,5):

        mat_km[i][j] = len(l1.values[i].intersection(l2_km.values[j]))

        mat_br[i][j] = len(l1.values[i].intersection(l2_br.values[j]))

        mat_ac[i][j] = len(l1.values[i].intersection(l2_ac.values[j]))
#Mapping Algorithm



ver = []

hor = []

d_km = {}

for k in range(0,5):

    ma = 0

    ii = -1

    jj = -1

    for i in range(0,5):

        for j in range(0,5):

            if(i not in ver and j not in hor and mat_km[i][j] >= ma):

                ma = mat_km[i][j]

                ii = i

                jj = j

    ver.append(ii)

    hor.append(jj)

    d_km[jj] = ii+1

print("Mapping KMean", d_km)
ver = []

hor = []

d_br = {}

for k in range(0,5):

    ma = 0

    ii = -1

    jj = -1

    for i in range(0,5):

        for j in range(0,5):

            if(i not in ver and j not in hor and mat_br[i][j] >= ma):

                ma = mat_km[i][j]

                ii = i

                jj = j

    ver.append(ii)

    hor.append(jj)

    d_br[jj] = ii+1

print("Mapping Birch", d_br)
ver = []

hor = []

d_ac = {}

for k in range(0,5):

    ma = 0

    ii = -1

    jj = -1

    for i in range(0,5):

        for j in range(0,5):

            if(i not in ver and j not in hor and mat_ac[i][j] >= ma):

                ma = mat_km[i][j]

                ii = i

                jj = j

    ver.append(ii)

    hor.append(jj)

    d_ac[jj] = ii+1

print("Mapping AgglomerativeClustering", d_ac)
#Apply the mapping



df_km['KM'] = df_km['KM'].map(d_km)

df_br['BR'] = df_br['BR'].map(d_br)

df_ac['AC'] = df_ac['AC'].map(d_ac)
#Rename the columns



df_km.columns=['ID','Class']

df_br.columns=['ID','Class']

df_ac.columns=['ID','Class']
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df_km.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(df_km)