!pip install --user numpy

!pip install --user pandas

!pip install --user matplotlib

!pip install --user seaborn

!pip install --user sklearn





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
data = pd.read_csv("../input/dmassign1/data.csv", sep=",", encoding = 'utf-8')

dz = data

Y = data['Class']

data = data.drop(['ID'], axis = 1)
data.info()
data = data.replace({'?':np.nan})

null_columns = data.columns[data.isnull().any()]

null_columns
data = data.applymap(lambda s:s.upper() if type(s) == str else s)

data['Col197'].replace({'M.E.' : 'ME'}, inplace=True)
data.columns[data.isnull().any()]
data1 = pd.get_dummies(data, columns=['Col189', 'Col190', 'Col191','Col192', 'Col193', 'Col194', 'Col195', 'Col196', 'Col197'])
data1.head()

data1 = data1.fillna(data.median())
data1.columns[data1.isnull().any()]
data1 = data1.drop(['Class'], axis = 1)
from sklearn import preprocessing

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler, MinMaxScaler
for i in data1.select_dtypes(object).columns:

    data1[i]=pd.to_numeric(data1[i],errors='raise')

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = StandardScaler().fit_transform(data1)

dataN1 = pd.DataFrame(np_scaled, columns = data1.columns)
pcatest = PCA(n_components = 5)

principalcomps = pcatest.fit_transform(dataN1)
features = range(pcatest.n_components_)

plt.figure(figsize=(16, 8))

plt.bar(features, pcatest.explained_variance_ratio_, color='black')

plt.xlabel('PCA features')

plt.ylabel('variance %')

plt.xticks(features)

pcss = pd.DataFrame(principalcomps, columns = ['p1', 'p2', 'p3', 'p4','p5'])

ks = range(1, 20)

inertias = []

for k in ks:

    # Create a KMeans instance with k clusters: model

    model = KMeans(n_clusters=k)

    

    # Fit model to samples

    model.fit(pcss.iloc[:,:7])

    

    # Append the inertia to the list of inertias

    inertias.append(model.inertia_)

    

plt.figure(figsize=(16, 8))

plt.plot(ks, inertias, '-o', color='black')

plt.xlabel('number of clusters, k')

plt.ylabel('inertia')

plt.xticks(ks)

plt.show()
kmean = KMeans(n_clusters=15, random_state = 42, n_init = 20)
kmean.fit(pcss)

pred = kmean.predict(pcss)
dataset = pd.DataFrame([dz['ID'], pred],['ID','KM']).T

dfinal = pd.DataFrame([dz['ID'][:1300], dz['Class'][:1300], dataset['KM'][:1300]]).T

dfinal = dfinal.astype({"Class":int, "KM":int})
w, h = 15, 5;

Matrix = np.zeros(shape = (15, 5), dtype = int)

for i in range(1300):

    s = dfinal['Class'].iloc[i] - 1

    t = dfinal['KM'].iloc[i]

    Matrix[t][s] += 1

Matrix
mp = {0:1, 1:4, 2:1, 3:5, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1, 11:1, 12:1, 13:1, 14:1}
dataset = dataset.astype({"KM":int})

dataset['KM']=dataset['KM'].map(mp)

dataset.columns=['ID', 'Class']

dataset[1300:].to_csv('final.csv', index=False)
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

create_download_link(dataset)