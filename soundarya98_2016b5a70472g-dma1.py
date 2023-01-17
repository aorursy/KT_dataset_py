import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv('/kaggle/input/dmassign1/data.csv', low_memory=False)

df.head()
train_orig=df[df['Class'].isnull()==False]

train=train_orig



train=train.drop('ID', axis=1)



test_orig=df

test=test_orig



test=test.drop('ID', axis=1)

test=test.drop('Class', axis=1)

train.info()
train.replace('?',np.NaN, inplace=True)

test.replace('?',np.NaN, inplace=True)
float_columns=['Col179', 'Col180', 'Col181', 'Col182', 'Col183', 'Col184', 'Col185', 'Col186', 'Col187', 'Col188']

object_columns=['Col189', 'Col190', 'Col191', 'Col192','Col193', 'Col194', 'Col195', 'Col196', 'Col197']
for column in object_columns:

    train[column].fillna(train[column].mode()[0], inplace=True)

    test[column].fillna(test[column].mode()[0], inplace=True)
train_obj=pd.get_dummies(train[object_columns], columns=object_columns)

test_obj=pd.get_dummies(test[object_columns], columns=object_columns)



dummy_obj_columns=pd.get_dummies(train[object_columns], columns=object_columns).columns



train=pd.get_dummies(train, columns=object_columns)

test=pd.get_dummies(test, columns=object_columns)
train=train.astype(float)

test=test.astype(float)
train.fillna(train.mean(), inplace=True)

test.fillna(test.mean(), inplace=True)
train.info()
tmp = train.corr()

f, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(tmp, mask=np.zeros_like(tmp, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax, annot = False)
train['Class'].value_counts()
train_=train.drop(['Class'], axis=1)

from sklearn import preprocessing

#Performing Min_Max Normalization

std_scaler = preprocessing.StandardScaler()

np_train_scaled = std_scaler.fit_transform(train_)

train_n = pd.DataFrame(np_train_scaled)

train_n.head()



test_=test

from sklearn import preprocessing

#Performing Min_Max Normalization

std_scaler = preprocessing.StandardScaler()

np_test_scaled = std_scaler.fit_transform(test_)

test_n = pd.DataFrame(np_test_scaled)

test_n.head()
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(train_n)

principalDf = pd.DataFrame(data = principalComponents

             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, train[['Class']]], axis = 1)

fig = plt.figure(figsize = (8,8))

ax = fig.add_subplot(1,1,1) 

ax.set_xlabel('Principal Component 1', fontsize = 15)

ax.set_ylabel('Principal Component 2', fontsize = 15)

ax.set_title('2 component PCA', fontsize = 20)

targets = [1, 2, 3, 4, 5]

colors = ['r', 'g', 'b','c', 'm']

for target, color in zip(targets,colors):

    indicesToKeep = train['Class'] == target

    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']

               , finalDf.loc[indicesToKeep, 'principal component 2']

               , c = color

               , s = 50)

ax.legend(targets)

ax.grid()
from sklearn.decomposition import PCA

#Fitting the PCA algorithm with our Data

pca = PCA().fit(test_n)

#Plotting the Cumulative Summation of the Explained Variance

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') #for each component

plt.title('DM Assignment 1 Explained Variance')

plt.show()
from sklearn.decomposition import PCA

pca2 = PCA(n_components=55)

pca2.fit(test_n)

T2 = pca2.transform(test_n)
X=train.drop('Class', axis=1)

y=train['Class']

from sklearn.ensemble import ExtraTreesClassifier

clf = ExtraTreesClassifier(n_estimators=2000)

clf = clf.fit(X, y)

clf.feature_importances_
for i in range(len(clf.feature_importances_)):

    print(i+1, clf.feature_importances_[i])
feat_importances = pd.Series(clf.feature_importances_, index=train.drop('Class', axis=1).columns)

feat_importances.nlargest(20).plot(kind='barh')

plt.show()
imp_features=['Col184', 'Col186', 'Col179', 'Col182']
def generateConfMat(arr):

    ConfMatrix={}

    for i in range(1300):

        if arr[i] not in ConfMatrix:

            ConfMatrix[arr[i]]={}

        if train_orig['Class'][i] not in ConfMatrix[arr[i]]:

            ConfMatrix[arr[i]][train_orig['Class'][i]]=0

        ConfMatrix[arr[i]][train_orig['Class'][i]]+=1

    

    import operator

    for i in ConfMatrix:

        for j in ConfMatrix[i]:

            ConfMatrix[i] = dict(sorted(ConfMatrix[i].items(), key=operator.itemgetter(1),reverse=True))

    

    maps = {}

    for key in ConfMatrix:

        for key_ in ConfMatrix[key]:

            maps[key] = (int)(key_)

            break

    return maps
def remap(arr, maps):

    arr_mapped = [0] * len(arr)

    for i in range(len(arr)):

        if arr[i] not in maps:

            arr_mapped[i]=randint(1, 5)

        else:

            arr_mapped[i]=maps[arr[i]]

    return arr_mapped
def toCSV(arr, df):

    test_orig=df[df['Class'].isnull()==True]

    arr = pd.DataFrame(data=arr)

    test_orig=test_orig.reset_index(drop=True)

    answer=pd.concat([test_orig['ID'], arr], axis=1)

    answer.columns=['ID', 'Class']

    answer.to_csv('answer_eval_lab.csv', index=False)

    print(answer['Class'].value_counts())

    return answer
from sklearn.cluster import KMeans



wcss = []

for i in range(2, 50):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(test_n)

    wcss.append(kmean.inertia_)

    

plt.plot(range(2,50),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
#Mild elbow at n=15

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=15, random_state=42)

kmeans.fit(test_n)

y_kmeans_test = kmeans.predict(test_n)

maps=generateConfMat(y_kmeans_test)

y_kmeans_test_mapped=remap(y_kmeans_test, maps)

print(maps)
from sklearn.cluster import Birch

bclust=Birch(branching_factor=2000, compute_labels=True, copy=True, n_clusters=20,

   threshold=0.5).fit(test_obj)

print(bclust)

y_birch_test = bclust.predict(test_obj)
# import scipy.cluster.hierarchy as sch

# dendrogram = sch.dendrogram(sch.linkage(test_n, method='ward'))
import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

from sklearn.cluster import AgglomerativeClustering

import scipy.cluster.hierarchy as sch



model = AgglomerativeClustering(n_clusters=42, affinity='cosine', linkage='average')

model.fit(T2)

y_agglo_test = model.labels_
maps=generateConfMat(y_agglo_test)

from random import randint

y_agglo_test_mapped=remap(y_agglo_test, maps)

df_answer=toCSV(y_agglo_test_mapped[1300:], df)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(train_orig['Class'], y_agglo_test[:1300], labels=[1, 2, 3, 4, 5])

cm

cm = pd.DataFrame(cm, range(5), range(5))

sns.set(font_scale=1.4)

f, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(cm, annot=True, annot_kws={"size": 16}) # font size

plt.show()
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "results.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(df_answer)