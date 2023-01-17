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
df = pd.read_csv('/kaggle/input/dmassign1/data.csv')
df.info()
df.shape
df.columns[df.isnull().any()]
df = df.drop_duplicates()

df.shape
df.corr()
plt.figure(figsize=(20, 20))

corr = df.corr()

corr.style.background_gradient(cmap = 'RdYlGn')
for i in df.columns.values:

    df[i].replace('?', 9999999, inplace = True)
object_to_float = ['Col30',

 'Col31',

 'Col34',

 'Col36',

 'Col37',

 'Col38',

 'Col39',

 'Col40',

 'Col43',

 'Col44',

 'Col46',

 'Col47',

 'Col48',

 'Col49',

 'Col50',

 'Col51',

 'Col53',

 'Col56',

 'Col138',

 'Col139',

 'Col140',

 'Col141',

 'Col142',

 'Col143',

 'Col144',

 'Col145',

 'Col146',

 'Col147',

 'Col148',

 'Col149',

 'Col151',

 'Col152',

 'Col153',

 'Col154',

 'Col155',

 'Col156',

 'Col157',

 'Col158',

 'Col159',

 'Col160',

 'Col161',

 'Col162',

 'Col173',

 'Col174',

 'Col175','Col179',

 'Col180',

 'Col181',

 'Col182',

 'Col183',

 'Col184',

 'Col185',

 'Col186', 'Col187']

for i in object_to_float:

    df[i] = df[i].astype(float)
for i in df.columns.values:

    if df[i].dtype == 'float64' or df[i].dtype == 'int64':

        df[i].replace(9999999, df[i].median(), inplace = True)

    else:

        df[i].replace(9999999, df[i].mode()[0], inplace = True)
for i in df.columns.values:

    if df[i].dtype != 'int64' and df[i].dtype != 'float64':

        print(i)
df = pd.get_dummies(df, columns = ['Col189', 'Col190', 'Col191', 'Col192', 'Col193', 'Col194', 'Col195', 'Col196', 'Col197'])
df.head()
from sklearn.preprocessing import StandardScaler
df.drop(columns = ['ID'], inplace = True)
X = df.drop(columns = ['Class'])

y = df['Class']
ss = StandardScaler()

X = ss.fit_transform(X)
X
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans

from tqdm import tqdm

wcss = []

for i in tqdm(range(2, 100)):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(X)

    wcss.append(kmean.inertia_)
plt.plot(range(2,100),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
from sklearn import metrics

preds1 = []

for i in tqdm(range(2,100)):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(X)

    pred = kmean.predict(X)

    preds1.append(metrics.calinski_harabasz_score(X, kmean.labels_))



plt.plot(range(2,100),preds1)

plt.title('The Calinski-Harabasz Index')

plt.xlabel('Number of clusters')

plt.ylabel('Index')

plt.show()
clf = KMeans(n_clusters = 25, random_state = 42)

clf.fit(X)
y_pred = clf.predict(X)[:1300]
y_actual = y[:1300]
mapping = {}

for i in np.unique(y_pred):

    count = {}

    maximum = 0

    max_label = 0

    for j, ii in enumerate(y_actual):

        if y_pred[j] == i:

            if ii not in count:

                count[ii] = 1

            else:

                count[ii] += 1

            if count[ii] > maximum:

                max_label = ii

                maximum = count[ii]

    mapping[i] = max_label

    print(count)

    
mapping
count2 = {}

y_pred = clf.predict(X)

y_pred2 = []

for i in y_pred:

    y_pred2.append(mapping[i])

for i in y_pred2:

    if i in count2.keys():

        count2[i] += 1

    else:

        count2[i] = 1
count2
import seaborn as sns

sns.countplot(df['Class'])
df_submit = pd.read_csv('/kaggle/input/dmassign1/data.csv')

df_submit = df_submit[df_submit['Class'].isnull()]
df_submit
df_submit2 = pd.DataFrame()

df_submit2['ID'] = df_submit['ID']
df_submit2['Class'] = list(y_pred2[1300:])
df_submit2['Class'] = df_submit2['Class'].astype(int)
df_submit2
df_submit2.to_csv('submit.csv', index = False)
from sklearn.cluster import AgglomerativeClustering as AC

aggclus = AC(n_clusters = 30,affinity='euclidean',linkage='ward',compute_full_tree='auto')

y_pred= aggclus.fit_predict(X)
mapping = {}

for i in np.unique(y_pred):

    count = {}

    maximum = 0

    max_label = 0

    for j, ii in enumerate(y_actual):

        if y_pred[j] == i:

            if ii not in count:

                count[ii] = 1

            else:

                count[ii] += 1

            if count[ii] > maximum:

                max_label = ii

                maximum = count[ii]

    mapping[i] = max_label

    print(count)
mapping
count2 = {}

y_pred = clf.predict(X)

y_pred2 = []

for i in y_pred:

    y_pred2.append(mapping[i])

for i in y_pred2:

    if i in count2.keys():

        count2[i] += 1

    else:

        count2[i] = 1
count2
from sklearn.decomposition import PCA

pca = PCA(n_components = 2) 

X_pca = pca.fit_transform(X) 

X_pca = pd.DataFrame(X_pca) 
from sklearn.cluster import AgglomerativeClustering 

ac2 = AgglomerativeClustering(n_clusters = 10) 

ac3 = AgglomerativeClustering(n_clusters = 15) 

ac4 = AgglomerativeClustering(n_clusters = 20) 

ac5 = AgglomerativeClustering(n_clusters = 25) 

ac6 = AgglomerativeClustering(n_clusters = 30) 
k = [2, 3, 4, 5, 6] 

from sklearn.metrics import silhouette_score

# Appending the silhouette scores of the different models to the list 

silhouette_scores = [] 

silhouette_scores.append( 

        silhouette_score(X, ac2.fit_predict(X))) 

silhouette_scores.append( 

        silhouette_score(X, ac3.fit_predict(X))) 

silhouette_scores.append( 

        silhouette_score(X, ac4.fit_predict(X))) 

silhouette_scores.append( 

        silhouette_score(X, ac5.fit_predict(X))) 

silhouette_scores.append( 

        silhouette_score(X, ac6.fit_predict(X))) 

  

# Plotting a bar graph to compare the results 

plt.bar(k, silhouette_scores) 

plt.xlabel('Number of clusters', fontsize = 20) 

plt.ylabel('S(i)', fontsize = 20) 

plt.show()
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

create_download_link(df_submit2)