import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns

import os

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.decomposition import PCA

from scipy.spatial.distance import cdist

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

from sklearn import metrics

from collections import Counter

from sklearn.metrics import silhouette_samples, silhouette_score

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2,mutual_info_classif

import matplotlib.cm as cm

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv("/kaggle/input/dmassign1/data.csv")

pd.options.display.max_columns = None

pd.set_option('display.expand_frame_repr', False)

df = df.replace('?',np.nan)

df.head()
y = df['Class']

ids = df['ID']

df_train = df.drop(['ID','Class'],axis=1)

df_train.head()
# from pandas_profiling import ProfileReport

# profile = ProfileReport(df_train, title='Pandas Profiling Report', html={'style':{'full_width':True}})

# profile.to_notebook_iframe()
df_train.select_dtypes(include=['object'])

df_train.head()
x = df.isnull().sum(axis = 0).values

print(x)
float_columns = ['Col30', 'Col31', 'Col34', 'Col36', 'Col37', 'Col38', 'Col39', 'Col40', 'Col43', 'Col44', 'Col46', 'Col47', 'Col48', 'Col49', 'Col50', 'Col51', 'Col53', 'Col56', 'Col138', 'Col139', 'Col140', 'Col141', 'Col142', 'Col143', 'Col144', 'Col145', 'Col146', 'Col147', 'Col148', 'Col149', 'Col151', 'Col152', 'Col153', 'Col154', 'Col155', 'Col156', 'Col157', 'Col158', 'Col159', 'Col160', 'Col161', 'Col162', 'Col173', 'Col174', 'Col175', 'Col179', 'Col180', 'Col181', 'Col182', 'Col183', 'Col184', 'Col185', 'Col186', 'Col187']

categorical_columns = ['Col189','Col190','Col191','Col192','Col193','Col194','Col195','Col196','Col197']
for col in float_columns:

    df_train[col] = df_train[col].astype(float)
df_train.select_dtypes(include=['object']).head()
from sklearn.base import TransformerMixin



class DataFrameImputer(TransformerMixin):



    def __init__(self):

        """Impute missing values.



        Columns of dtype object are imputed with the most frequent value 

        in column.



        Columns of other types are imputed with median of column.



        """

    def fit(self, X, y=None):



        self.fill = pd.Series([X[c].value_counts().index[0]

            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],

            index=X.columns)



        return self



    def transform(self, X, y=None):

        return X.fillna(self.fill)







df_train = DataFrameImputer().fit_transform(df_train)
x = df_train.isnull().sum(axis = 0).values

print(x)
df_train.Col197.value_counts()
def fun_Col197(x):

    if x=='XL':

        x='xl'

    elif x == 'ME':

        x = 'me'

    elif x == 'SM':

        x = 'sm'

    elif x == 'M.E.':

        x = 'me'

    elif x == 'LA':

        x = 'la'

    return x
df_train['Col197']= df_train['Col197'].apply(fun_Col197)
print(df_train['Col197'].value_counts())
for i in range(len(categorical_columns)):

    print(" Col no: {}".format(categorical_columns[i]))

    print(df_train[categorical_columns[i]].value_counts())
df_train = pd.get_dummies(df_train, columns=categorical_columns)
df_train[:][:1300].head()
sel = SelectKBest(mutual_info_classif, k=50)

sel.fit(df_train.iloc[:1300],df["Class"][:1300])

df_new = sel.transform(df_train)

# df_new.head()
print(df_new.shape[1])
corr = df_train.corr()
corr
columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):

    for j in range(i+1, corr.shape[0]):

        if corr.iloc[i,j] <= 0.1:

            if columns[j]:

                columns[j] = False

selected_columns = df_train.columns[columns]
print(len(selected_columns))

print(selected_columns.values)

df_train2 = df_train.drop(columns= list(selected_columns.values),axis=1)

df_train2.head()
scaled_data = StandardScaler().fit_transform(df_train2)

df_scaled=pd.DataFrame(scaled_data,columns=df_train2.columns)

df_scaled.head()
pca = PCA(n_components=0.9,svd_solver="full")

pca.fit(df_scaled)

T1 = pca.transform(df_scaled)

pca.explained_variance_ratio_.sum()
from sklearn.cluster import KMeans

wcss = []

for i in range(2, 50):

    kmean = KMeans(n_clusters = i, random_state = 10)

    kmean.fit(df_scaled)

    wcss.append(kmean.inertia_)



plt.plot(range(2,50),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
mx, mr, mn = 0, 0, 0



# Random centroid initialization and n_clusters simulation

for r in range(35,50):

    for num in range(40, 60):

        kmeans = KMeans(n_clusters = num, random_state=r)

        pred = kmeans.fit_predict(T1)



        a = {}

        for item in range(num):

            a[item] = []

        

        for index, p in enumerate(pred[:1300]):

            a[p].append(index)



        subs = {}

        for item in range(num):

            if len(a[item]) == 0:

                continue

            subs[item] = int(Counter(df['Class'].iloc[a[item]].dropna()).most_common(1)[0][0])



        test = [subs.get(n, n) for n in pred[:1300]]

        pred1 = [subs.get(n, n) for n in pred[1300:]]



        correct, total = 0,0

        for i,j in zip(test, y[:1300]):

            if i==int(j):

                correct+=1

            total+=1



        if correct/total>mx:

            mx = correct/total

            mn = num

            mr = r

    print('Iteration :', r)

    

print('Found optimal hyperparameters ->')

print('Number of clusters: ', mn)

print('Random State: ', mr)
kmeans = KMeans(n_clusters = mn, random_state=mr)

pred = kmeans.fit_predict(T1)



color = cm.nipy_spectral(float(i) / mn)

plt.scatter(T1[:, 0], T1[:, 1], marker='o', c=pred, edgecolor='k')
a = {}

for item in range(mn):

    a[item] = []



for index, p in enumerate(pred[:1300]):

    a[p].append(index)

    

subs = {}

for item in range(mn):

    if len(a[item]) == 0:

                continue

    subs[item] = int(Counter(df['Class'].iloc[a[item]].dropna()).most_common(1)[0][0])



test = [subs.get(n, n) for n in pred[:1300]]

pred1 = [subs.get(n, n) for n in pred[1300:]]
Counter(pred1).keys()

Counter(pred1).values()
correct, total = 0,0

for i,j in zip(test, y[:1300]):

    if i==int(j):

        correct+=1

    total+=1



print(correct/total)
for i in range(len(pred1)):

    if pred1[i] not in [1,2,3,4,5]:

        pred1[i] = 5



res = pd.DataFrame({'ID': df['ID'].iloc[1300:], 'Class': pred1})



len(res)

res.head()
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

create_download_link(res)
# from sklearn.ensemble import RandomForestClassifier

# rf = RandomForestClassifier(max_depth=5)

# rf.fit(df_train.iloc[:1300], df['Class'][:1300])

# rf.feature_importances_
selected = (rf.feature_importances_ > 0)

selected
df_new2 = df_train[df_train.columns[selected]]

df_new2.head()
from sklearn.feature_selection import RFECV

from sklearn.model_selection import KFold

rfecv = RFECV(estimator=rf, step=1, cv=KFold(2), scoring='accuracy')

rfecv.fit(df_train.iloc[:1300], df['Class'][:1300])

selected = rfecv.support_
df_new3 = df_train[df_train.columns[selected]]

df_new3.head()
scaled_data = StandardScaler().fit_transform(df_new)

cols = [i for i in range(df_new.shape[1])]

# scaled_data = StandardScaler().fit_transform(df_train)

# df_scaled=pd.DataFrame(scaled_data,columns=df_train.columns)

df_scaled = pd.DataFrame(scaled_data)

df_scaled.head()
pca = PCA(n_components=0.90, svd_solver="full")

pca.fit(df_scaled)

pca.fit(scaled_data)

T1 = pca.transform(df_scaled)

pca.explained_variance_ratio_.sum()
mx, mr, mn = 0, 0, 0



for r in range(35,50):

    for num in range(40,70):

    #     centroids_scaled = naive_sharding(T1, num)

        kmeans = KMeans(n_clusters = num, random_state=r)

        pred = kmeans.fit_predict(T1)



        a = {}

        for item in range(num):

            a[item] = []



        for index, p in enumerate(pred[:1300]):

            a[p].append(index)



        temp = {}

        for item in range(num):

            if len(a[item]) == 0:

                continue

            temp[item] = int(Counter(df['Class'].iloc[a[item]].dropna()).most_common(1)[0][0])



        test = [temp.get(n, n) for n in pred[:1300]]

        pred1 = [temp.get(n, n) for n in pred[1300:]]



        correct, total = 0,0

        for i,j in zip(test, y[:1300]):

            if i==int(j):

                correct+=1

            total+=1

        acc = correct/total

        if correct/total>mx:

            mx = correct/total

            mn = num

            mr = r

    print(' Cluster : {}, RandomState : {}, Accuracy : {}'.format(mn,mr,mx))

    

print('Found optimal hyperparameters ->')

print('Number of clusters: ', mn)

print('Random State: ', mr)
mn , mr = 55, 41

kmeans = KMeans(n_clusters = mn, random_state=mr)

pred = kmeans.fit_predict(T1)



a = {}

for item in range(mn):

    a[item] = []



for index, p in enumerate(pred[:1300]):

    a[p].append(index)

    

subs = {}

for item in range(mn):

    if len(a[item]) == 0:

        continue

    subs[item] = int(Counter(df['Class'].iloc[a[item]].dropna()).most_common(1)[0][0])



test = [subs.get(n, n) for n in pred[:1300]]

pred1 = [subs.get(n, n) for n in pred[1300:]]



correct, total = 0,0

for i,j in zip(test, y[:1300]):

    if i==int(j):

        correct+=1

    total+=1



print(correct/total)
# df_train['cluster'] = pred

# pd.plotting.parallel_coordinates(df_train, 'cluster')

# df_train.drop(['cluster'],axis=1)

Counter(pred1)
for i in range(len(pred1)):

    if pred1[i] not in [1,2,3,4,5]:

        pred1[i] = np.random.randint(1,5,size=1)[0]

        

# print(np.random.randint(1,5,size=1)[0])

Counter(pred1)

res = pd.DataFrame({'ID': df['ID'].iloc[1300:], 'Class': pred1})



len(res)

res.head()
Counter(res['Class'])
res['Class'].value_counts()
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

create_download_link(res)
scaled_data = StandardScaler().fit_transform(df_train)

df_scaled=pd.DataFrame(scaled_data,columns=df_train.columns)

df_scaled.head()



pca = PCA(n_components=0.9, svd_solver="full")

pca.fit(df_scaled)

T1 = pca.transform(df_scaled)

pca.explained_variance_ratio_.sum()
ag = AgglomerativeClustering(n_clusters = 3,affinity='cosine',linkage='average',compute_full_tree='auto')

y_aggclus= ag.fit_predict(T1)
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree

from scipy.cluster.hierarchy import fcluster

plt.figure(figsize=(100,100))

linkage_matrix1 = linkage(T1, "average",metric="euclidean")

ddata1 = dendrogram(linkage_matrix1,color_threshold=10)
y_ac=cut_tree(linkage_matrix1, n_clusters = 50).T

y_ac[0]
plt.scatter(T1[:,0], T1[:,3], c=y_ac[0,:], s=100, label='')

plt.show()
scaled_data = StandardScaler().fit_transform(df_train)

df_scaled=pd.DataFrame(scaled_data,columns=df_train.columns)

df_scaled.head()



pca = PCA(n_components=60, svd_solver="full")

pca.fit(df_scaled)

T1 = pca.transform(df_scaled)

pca.explained_variance_ratio_.sum()
dbscan = DBSCAN(eps=5, min_samples=3)

pred = dbscan.fit_predict(T1)

plt.scatter(T1[:, 0], T1[:, 2], c=pred)
labels1 = dbscan.labels_

#labels1 = labels1[labels1 >= 0] #Remove Noise Points

labels1, counts1 = np.unique(labels1, return_counts=True)

print(len(labels1))

print(labels1)

print(len(counts1))

print(counts1)

from sklearn.manifold import TSNE



# Project the data: this step will take several seconds

tsne = TSNE(n_components=2, init='random', random_state=0)

T1 = tsne.fit_transform(df_train)
mx, mr, mn = 0, 0, 0



# Random centroid initialization and n_clusters simulation

for r in range(35,50):

    for num in range(45, 60):

        kmeans = KMeans(n_clusters = num, random_state=r)

        pred = kmeans.fit_predict(T1)



        a = {}

        for item in range(num):

            a[item] = []

        

        for index, p in enumerate(pred[:1300]):

            a[p].append(index)



        subs = {}

        for item in range(num):

            if len(a[item]) == 0:

                continue

            subs[item] = int(Counter(df['Class'].iloc[a[item]].dropna()).most_common(1)[0][0])



        test = [subs.get(n, n) for n in pred[:1300]]

        pred1 = [subs.get(n, n) for n in pred[1300:]]



        correct, total = 0,0

        for i,j in zip(test, y[:1300]):

            if i==int(j):

                correct+=1

            total+=1



        if correct/total>mx:

            mx = correct/total

            mn = num

            mr = r

    print('Iteration :', r)

    

print('Found optimal hyperparameters ->')

print('Number of clusters: ', mn)

print('Random State: ', mr)