import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

from time import time

import numpy as np

import matplotlib.pyplot as plt

from sklearn import metrics

from sklearn.cluster import KMeans

from sklearn.datasets import load_digits

from sklearn.decomposition import PCA

from sklearn.preprocessing import scale

import matplotlib.pyplot as plt

import pandas

from pandas.plotting import scatter_matrix

from sklearn.preprocessing import MinMaxScaler

from sklearn.datasets.samples_generator import (make_blobs,

                                                make_circles,

                                                make_moons)

from sklearn.cluster import KMeans, SpectralClustering

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import silhouette_samples, silhouette_score





from keras.layers import Input, Dense

from keras.models import Model

from keras import regularizers

from sklearn.model_selection import train_test_split





import seaborn as sns

from sklearn.metrics import accuracy_score

from sklearn.metrics.cluster import completeness_score

from sklearn.metrics import confusion_matrix
data = pd.read_csv("../input/diabetic_data.csv")

data.head()

data.isna().sum()
data.describe()
data.drop(['encounter_id', 'patient_nbr', 'weight','medical_specialty', 'payer_code','admission_source_id' ], axis=1, inplace= True)
data['gender'].value_counts()
data = data[data.gender != 'Unknown/Invalid']
data = data[(data.discharge_disposition_id != 11) & (data.discharge_disposition_id != 19) & (data.discharge_disposition_id != 20) & (data.discharge_disposition_id != 21) & (data.discharge_disposition_id != 7) ]
data = data[(data.diag_1 != '?') | (data.diag_2 != '?') | (data.diag_3 != '?')]

data.drop(['chlorpropamide','acetohexamide', 'tolbutamide', 'rosiglitazone', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton',

       'glyburide-metformin', 'glipizide-metformin','glimepiride-pioglitazone', 'metformin-rosiglitazone','metformin-pioglitazone'],axis=1, inplace= True)
data.drop(['diag_1','diag_2','diag_3'], axis=1, inplace= True)
data.race.value_counts()
data["Race"]= data["race"].map(lambda x:'NA' if x=='?' else x)

data.drop(['race'], axis=1, inplace= True)
def get_fn(row):

    if row['admission_type_id']==1 or row['admission_type_id']==2 or row['admission_type_id']==7 :

        return "Non ELective"

    elif row['admission_type_id']==3 or row['admission_type_id']==4:

        return "Elective"

    else :

        return "NA"

data['admission_type']= data.apply(get_fn,axis=1)

data.drop(['admission_type_id'], axis=1, inplace= True)
data.readmitted.value_counts()
def fn(x):

    if x =='NO' or x=='>30':

        return 0

    else :

        return 1

data['readmit']= data['readmitted'].map(fn)

data.drop(['readmitted'], axis=1, inplace= True)
data['A1Cresult'].value_counts()
def fun(z):

    if z =='None' or z=='Norm':

        return 1

    else :

        return 0

data['A1C']= data['A1Cresult'].map(fun)

data.drop(['A1Cresult'], axis=1, inplace= True)
def gt_ag(a):

    if a =='[0-10)' or a=='[10-20)' or a=='[20-30)':

        return 'young'

    elif a =='[30-40)' or a=='[40-50)' or a=='[50-60)':

        return 'mid'

    else:

        return'old'

data['Age']= data['age'].map(gt_ag)

data.drop(['age'], axis=1, inplace= True)
data['max_glu_serum'].value_counts()
data['max_glu_serum']=data['max_glu_serum'].replace('None',0)

data['max_glu_serum']=data['max_glu_serum'].replace('Norm',0)

data['max_glu_serum']=data['max_glu_serum'].replace('>200',1)

data['max_glu_serum']=data['max_glu_serum'].replace('>300',1)
def dp_id(a):

    if a ==6 or a==8 or a==9 or a==13 or a==1:

        return 'Discharged Home'

    elif a==18 or a ==25 or a==26 :

        return 'NA'

    else:

        return'Discharged/Transferred'

data['discharge']= data['discharge_disposition_id'].map(dp_id)

data.drop(['discharge_disposition_id'], axis=1, inplace= True)
data.head()
data = pd.get_dummies(data=data, columns=['discharge', 'Age','admission_type','Race','change','insulin','glipizide','gender'])
data.head()
data = pd.get_dummies(data=data, columns=['metformin','repaglinide','nateglinide','glimepiride','glyburide','pioglitazone','acarbose'])
data['diabetesMed'] = data['diabetesMed'].map({'Yes': 1, 'No': 0})
data.head()
x = data.drop(['readmit'], axis=1)

y = data['readmit']
scaler = MinMaxScaler(feature_range=[0,1])

x = scaler.fit_transform(x)
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.3)
x_test.shape, x_train.shape,y_test.shape, y_train.shape
from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42,max_iter = 1000)

    kmeans.fit(x_train)

    # inertia method returns wcss for that model

    wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10,5))



plt.figure(figsize=(10,5))

sns.lineplot(range(1, 11), wcss,marker='o',color='red')

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('SSE')

plt.show()
sse = {}

data = pd.DataFrame()

for k in range(1, 10):

    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(x_train)

    data["clusters"] = kmeans.labels_

    #print(data["clusters"])

    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

plt.figure()

plt.plot(list(sse.keys()), list(sse.values()))

plt.xlabel("Number of cluster")

plt.ylabel("SSE")

plt.show()
from scipy.spatial.distance import cdist

distortions = []

K = range(1,10)

for k in K:

    kmeanModel = KMeans(n_clusters = k, init = 'k-means++', random_state = 42,max_iter = 1000).fit(x_train)

    kmeanModel.fit(x_train)

    distortions.append(sum(np.min(cdist(x_train, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / x_train.shape[0])



# Plot the elbow

plt.plot(K, distortions, 'bx-')

plt.xlabel('k')

plt.ylabel('Distortion')

plt.title('The Elbow Method showing the optimal k')

plt.show()
for i, k in enumerate([2, 3, 4]):

    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.set_size_inches(18, 7)

    

    # Run the Kmeans algorithm

    km = KMeans(n_clusters=k, max_iter = 1000)

    labels = km.fit_predict(x_train)

    y_label_test = km.predict(x_test)

    centroids = km.cluster_centers_



    # Get silhouette samples

    silhouette_vals = silhouette_samples(x_train, labels)

    

    

    

    # Silhouette plot

    y_ticks = []

    y_lower, y_upper = 0, 0

    for i, cluster in enumerate(np.unique(labels)):

        cluster_silhouette_vals = silhouette_vals[labels == cluster]

        cluster_silhouette_vals.sort()

        y_upper += len(cluster_silhouette_vals)

        ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)

        ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))

        y_lower += len(cluster_silhouette_vals)



    # Get the average silhouette score and plot it

    avg_score = np.mean(silhouette_vals)

    ax1.axvline(avg_score, linestyle='--', linewidth=2, color='green')

    ax1.set_yticks([])

    ax1.set_xlim([-0.1, 1])

    ax1.set_xlabel('Silhouette coefficient values')

    ax1.set_ylabel('Cluster labels')

    ax1.set_title('Silhouette plot for the various clusters', y=1.02);

    

    # Scatter plot of data colored with labels

    ax2.scatter(x_train[:, 0], x_train[:, 1], c=labels)

    ax2.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='r', s=250)

    ax2.set_xlim([-2, 2])

    ax2.set_xlim([-2, 2])

    ax2.set_xlabel('Eruption time in mins')

    ax2.set_ylabel('Waiting time to next eruption')

    ax2.set_title('Visualization of clustered data', y=1.02)

    ax2.set_aspect('equal')

    plt.tight_layout()

    plt.suptitle(f'Silhouette analysis using k = {k}',

                 fontsize=16, fontweight='semibold', y=1.05);

   

    silhouette_avg = silhouette_score(x_train, labels)

    print("For n_clusters =", k,

          "The average silhouette_score is :", silhouette_avg)

    #y_label_train = kmeans.labels_

    accurcyscore = accuracy_score(y_test,y_label_test)

    completenessscore =completeness_score(y_test, y_label_test)

    print("Accuracy Score ",accurcyscore)

    print("Completeness Score ",completenessscore)

    #print(confusion_matrix(y_test, y_label_test))
# Applying AutoEncoder
input_data = Input(shape=(66,))

encoded = Dense(60, activation='relu')(input_data)

encoded = Dense(52, activation='relu')(encoded)

encoded = Dense(42, activation='relu')(encoded)

encoded = Dense(32, activation='relu')(encoded)

encoded = Dense(22, activation='relu')(encoded)

encoded = Dense(12, activation='relu')(encoded)



encoded = Dense(2, activation='relu')(encoded)



decoded = Dense(22, activation='relu')(encoded)

decoded = Dense(32, activation='relu')(decoded)

decoded = Dense(42, activation='relu')(decoded)

decoded = Dense(52, activation='relu')(decoded)

decoded = Dense(60, activation='relu')(decoded)

decoded = Dense(66, activation='sigmoid')(decoded)
autoencoder = Model(input_data, decoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')



autoencoder.fit(x_train, x_train,

                epochs=45,

                batch_size=128,

                shuffle=True,

                validation_data=(x_test, x_test))
encoder = Model(input_data, encoded)
reduced_x_train = encoder.predict(x_train)

reduced_x_test = encoder.predict(x_test)

reduced_x_train.shape, reduced_x_test.shape
from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(encoder).create(prog='dot', format='svg'))
dummy = pd.DataFrame()



sse = {}

for k in range(1, 10):

    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(reduced_x_train)

    dummy["auto_clusters"] = kmeans.labels_

    #print(data["clusters"])

    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

plt.figure()

plt.plot(list(sse.keys()), list(sse.values()))

plt.xlabel("Number of cluster")

plt.ylabel("SSE")

plt.show()
from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42,max_iter = 1000)

    kmeans.fit(reduced_x_train)

    # inertia method returns wcss for that model

    wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10,5))



plt.figure(figsize=(10,5))

sns.lineplot(range(1, 11), wcss,marker='o',color='red')

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('SSE')

plt.show()
for i, k in enumerate([2, 3, 4]):

    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.set_size_inches(18, 7)

    

    # Run the Kmeans algorithm

    km = KMeans(n_clusters=k, max_iter = 1000)

    labels = km.fit_predict(reduced_x_train)

    y_label_test = km.predict(reduced_x_test)

    centroids = km.cluster_centers_



    # Get silhouette samples

    silhouette_vals = silhouette_samples(reduced_x_train, labels)

    

    

    

    # Silhouette plot

    y_ticks = []

    y_lower, y_upper = 0, 0

    for i, cluster in enumerate(np.unique(labels)):

        cluster_silhouette_vals = silhouette_vals[labels == cluster]

        cluster_silhouette_vals.sort()

        y_upper += len(cluster_silhouette_vals)

        ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)

        ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))

        y_lower += len(cluster_silhouette_vals)



    # Get the average silhouette score and plot it

    avg_score = np.mean(silhouette_vals)

    ax1.axvline(avg_score, linestyle='--', linewidth=2, color='green')

    ax1.set_yticks([])

    ax1.set_xlim([-0.1, 1])

    ax1.set_xlabel('Silhouette coefficient values')

    ax1.set_ylabel('Cluster labels')

    ax1.set_title('Silhouette plot for the various clusters', y=1.02);

    

    # Scatter plot of data colored with labels

    ax2.scatter(reduced_x_train[:, 0], reduced_x_train[:, 1], c=labels)

    ax2.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='r', s=250)

    ax2.set_xlim([-2, 2])

    ax2.set_xlim([-2, 2])

    ax2.set_xlabel('Eruption time in mins')

    ax2.set_ylabel('Waiting time to next eruption')

    ax2.set_title('Visualization of clustered data', y=1.02)

    ax2.set_aspect('equal')

    plt.tight_layout()

    plt.suptitle(f'Silhouette analysis using k = {k}',

                 fontsize=16, fontweight='semibold', y=1.05);

   

    silhouette_avg = silhouette_score(reduced_x_train, labels)

    print("For n_clusters =", k,

          "The average silhouette_score is :", silhouette_avg)

    #y_label_train = kmeans.labels_

    accurcyscore = accuracy_score(y_test,y_label_test)

    completenessscore =completeness_score(y_test, y_label_test)

    print("Accuracy Score ",accurcyscore)

    print("Completeness Score ",completenessscore)

    #print(confusion_matrix(y_test, y_label_test))
#Using softmax to form clusters



input_data = Input(shape=(66,))

encoded = Dense(60, activation='relu')(input_data)

encoded = Dense(52, activation='relu')(encoded)

encoded = Dense(42, activation='relu')(encoded)

encoded = Dense(32, activation='relu')(encoded)

encoded = Dense(22, activation='relu')(encoded)

encoded = Dense(12, activation='relu')(encoded)



encoded = Dense(2, activation='softmax')(encoded)



decoded = Dense(22, activation='relu')(encoded)

decoded = Dense(32, activation='relu')(decoded)

decoded = Dense(42, activation='relu')(decoded)

decoded = Dense(52, activation='relu')(decoded)

decoded = Dense(60, activation='relu')(decoded)

decoded = Dense(66, activation='sigmoid')(decoded)
autoencoder = Model(input_data, decoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')



autoencoder.fit(x_train, x_train,

                epochs=45,

                batch_size=128,

                shuffle=True,

                validation_data=(x_test, x_test))
encoder = Model(input_data, encoded)

reduced_x_train = encoder.predict(x_train)

reduced_x_test = encoder.predict(x_test)
predict_clusters = np.argmax(reduced_x_test, axis=1)
predict_clusters
reduced_x_test.shape
from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(encoder).create(prog='dot', format='svg'))
dummy = pd.DataFrame()



sse = {}

for k in range(1, 10):

    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(reduced_x_train)

    dummy["auto_clusters"] = kmeans.labels_

    #print(data["clusters"])

    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

plt.figure()

plt.plot(list(sse.keys()), list(sse.values()))

plt.xlabel("Number of cluster")

plt.ylabel("SSE")

plt.show()
from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42,max_iter = 1000)

    kmeans.fit(reduced_x_train)

    # inertia method returns wcss for that model

    wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10,5))



plt.figure(figsize=(10,5))

sns.lineplot(range(1, 11), wcss,marker='o',color='red')

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('SSE')

plt.show()
for i, k in enumerate([2, 3, 4]):

    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.set_size_inches(18, 7)

    

    # Run the Kmeans algorithm

    km = KMeans(n_clusters=k, max_iter = 1000)

    labels = km.fit_predict(reduced_x_train)

    y_label_test = km.predict(reduced_x_test)

    centroids = km.cluster_centers_



    # Get silhouette samples

    silhouette_vals = silhouette_samples(reduced_x_train, labels)

    

    

    

    # Silhouette plot

    y_ticks = []

    y_lower, y_upper = 0, 0

    for i, cluster in enumerate(np.unique(labels)):

        cluster_silhouette_vals = silhouette_vals[labels == cluster]

        cluster_silhouette_vals.sort()

        y_upper += len(cluster_silhouette_vals)

        ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)

        ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))

        y_lower += len(cluster_silhouette_vals)



    # Get the average silhouette score and plot it

    avg_score = np.mean(silhouette_vals)

    ax1.axvline(avg_score, linestyle='--', linewidth=2, color='green')

    ax1.set_yticks([])

    ax1.set_xlim([-0.1, 1])

    ax1.set_xlabel('Silhouette coefficient values')

    ax1.set_ylabel('Cluster labels')

    ax1.set_title('Silhouette plot for the various clusters', y=1.02);

    

    # Scatter plot of data colored with labels

    ax2.scatter(reduced_x_train[:, 0], reduced_x_train[:, 1], c=labels)

    ax2.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='r', s=250)

    ax2.set_xlim([-2, 2])

    ax2.set_xlim([-2, 2])

    ax2.set_xlabel('Eruption time in mins')

    ax2.set_ylabel('Waiting time to next eruption')

    ax2.set_title('Visualization of clustered data', y=1.02)

    ax2.set_aspect('equal')

    plt.tight_layout()

    plt.suptitle(f'Silhouette analysis using k = {k}',

                 fontsize=16, fontweight='semibold', y=1.05);

   

    silhouette_avg = silhouette_score(reduced_x_train, labels)

    print("For n_clusters =", k,

          "The average silhouette_score is :", silhouette_avg)

    #y_label_train = kmeans.labels_

    accurcyscore = accuracy_score(y_test,y_label_test)

    completenessscore =completeness_score(y_test, y_label_test)

    print("Accuracy Score ",accurcyscore)

    print("Completeness Score ",completenessscore)

    #print(confusion_matrix(y_test, y_label_test))