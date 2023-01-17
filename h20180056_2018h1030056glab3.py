import time

import warnings



import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn import cluster, datasets, mixture

from sklearn.neighbors import kneighbors_graph

from sklearn.preprocessing import StandardScaler

from itertools import cycle, islice

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

from sklearn import metrics

import time

from sklearn.metrics import accuracy_score, classification_report, mean_squared_error,mean_absolute_error

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 

                              GradientBoostingClassifier, ExtraTreesClassifier)
import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

import sklearn

from scipy import stats

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set() # for plot styling
# df=pd.read_csv("train.csv")

# tt=pd.read_csv("test.csv")
df=pd.read_csv("/kaggle/input/eval-lab-3-f464/train.csv")

tt=pd.read_csv("/kaggle/input/eval-lab-3-f464/test.csv")
df.head()
tt.head()
df.isnull().any().any()
tt.isnull().any().any()
df.info()
tt.info()
df["TotalCharges"]=pd.to_numeric(df["TotalCharges"], errors='coerce')

df.info()
tt["TotalCharges"]=pd.to_numeric(tt["TotalCharges"], errors='coerce')

tt.info()
for feature in df.columns:

    if feature!="custId":

        print("For feature",feature,"value counts is")

        print(df[feature].value_counts())

        print("--------")
categorical_features=["gender","SeniorCitizen","Married","Children","TVConnection","Channel1","Channel2","Channel3","Channel4","Channel5","Channel6","Internet","HighSpeed","AddedServices","Subscription","PaymentMethod","tenure"]

train_custid= df["custId"]

numerical_features=["MonthlyCharges","TotalCharges"]
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()  #Instantiate the encoder

for feature in categorical_features:

    df[feature] = le.fit_transform(df[feature])  #Fit and transform the labels using labelencoder

    tt[feature] =le.transform(tt[feature])

df.head()
tt.head()
import seaborn as sns

f, ax = plt.subplots(figsize=(30, 15))

corr = df.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax)
tt["TotalCharges"].fillna(0.0,inplace=True)

tt.info()
df["TotalCharges"].fillna(0.0,inplace=True)

df.info()
print(df.isnull().any().any())

print(tt.isnull().any().any())
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=30)

sns.heatmap(df.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
categorical_features=["SeniorCitizen","Married","Children","TVConnection","Channel3","Channel4","Channel5","Channel6","AddedServices","Subscription","PaymentMethod","tenure"]

numerical_features=["MonthlyCharges","TotalCharges"]
X_train= df[categorical_features+numerical_features].copy()

y_train= df["Satisfied"].copy()

X_test= tt[categorical_features+numerical_features].copy()
# from sklearn.model_selection import train_test_split



# X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.33,random_state=42) 
from sklearn import preprocessing



scaler= preprocessing.RobustScaler()

rbscaler = preprocessing.RobustScaler()

stdscaler = preprocessing.StandardScaler()

mmscaler = preprocessing.MinMaxScaler()



X_train[numerical_features] = stdscaler.fit_transform(X_train[numerical_features])

X_test[numerical_features] = stdscaler.transform(X_test[numerical_features])

# test_data=scaler.transform(test_data)
# ward = cluster.AgglomerativeClustering(

#     n_clusters=2, linkage='ward')

# connectivity matrix for structured Ward

connectivity = kneighbors_graph(

    X_test, n_neighbors=3, include_self=False)

# make connectivity symmetric

connectivity = 0.5 * (connectivity + connectivity.T)

bandwidth = cluster.estimate_bandwidth(X_test, quantile=0.3)

# ============

# Create cluster objects

# ============

ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)

two_means = cluster.MiniBatchKMeans(n_clusters=2)

ward = cluster.AgglomerativeClustering(

    n_clusters=2, linkage='ward',

    connectivity=connectivity)

spectral = cluster.SpectralClustering(

    n_clusters=2, eigen_solver='arpack',

    affinity="nearest_neighbors")

dbscan = cluster.DBSCAN(eps=0.3)

#optics = cluster.OPTICS(min_samples=params['min_samples'],

#                        xi=params['xi'],

#                        min_cluster_size=params['min_cluster_size'])

# affinity_propagation = cluster.AffinityPropagation(

#     damping=params['damping'], preference=params['preference'])

complete = cluster.AgglomerativeClustering(

    n_clusters=2, linkage='complete',connectivity=connectivity)

average = cluster.AgglomerativeClustering(

    n_clusters=2, linkage='average',connectivity=connectivity)

single = cluster.AgglomerativeClustering(

    n_clusters=2, linkage='single',connectivity=connectivity)

birch = cluster.Birch(n_clusters=2)

gmm = mixture.GaussianMixture(

    n_components=2, covariance_type='full')

kmeans = cluster.KMeans(n_clusters = 2,init = "k-means++",random_state =0,max_iter=1000)
clustering_algorithms = (

    ('Agglomerative Single Linkage', single),

    ('Agglomerative Average Linkage', average),

    ('Agglomerative Complete Linkage', complete),

    ('Agglomerative Ward Linkage', ward),

    ('MiniBatchKMeans', two_means),

#     ('AgglomerativeClustering', average_linkage),

#     ('AffinityPropagation', affinity_propagation),

#     ('MeanShift', ms),

    ('SpectralClustering', spectral),

#     ('DBSCAN', dbscan),

    #('OPTICS', optics),

    ('Birch', birch),

    ('GaussianMixture', gmm),

    ('K-Means', kmeans)

)
# for name, algorithm in clustering_algorithms:

#     t0 = time.time()



#     # catch warnings related to kneighbors_graph

#     with warnings.catch_warnings():

#         warnings.filterwarnings(

#             "ignore",

#             message="the number of connected components of the " +

#             "connectivity matrix is [0-9]{1,2}" +

#             " > 1. Completing it to avoid stopping the tree early.",

#             category=UserWarning)

#         warnings.filterwarnings(

#             "ignore",

#             message="Graph is not fully connected, spectral embedding" +

#             " may not work as expected.",

#             category=UserWarning)

#         algorithm.fit(X_train)

        



#     t1 = time.time()

#     if hasattr(algorithm, 'labels_'):

#         y_pred = algorithm.labels_.astype(np.int)

#     else:

#         y_pred = algorithm.predict(X_train)

#     print("For algo= ",name)

#     print ("Accuracy : %.4g" % accuracy_score(y_pred, y_train))

#     print ("AUC Score : %.4g" % roc_auc_score(y_pred, y_train))


#     clustering_algorithms = (

#         ('MiniBatchKMeans', two_means),

#         ('AffinityPropagation', affinity_propagation),

#         ('MeanShift', ms),

#         ('SpectralClustering', spectral),

#         ('Ward', ward),

#         ('AgglomerativeClustering', average_linkage),

#         ('DBSCAN', dbscan),

#         #('OPTICS', optics),

#         ('Birch', birch),

#         ('GaussianMixture', gmm)

#     )



for name, algorithm in clustering_algorithms:

    t0 = time.time()



    # catch warnings related to kneighbors_graph

    with warnings.catch_warnings():

        warnings.filterwarnings(

            "ignore",

            message="the number of connected components of the " +

            "connectivity matrix is [0-9]{1,2}" +

            " > 1. Completing it to avoid stopping the tree early.",

            category=UserWarning)

        warnings.filterwarnings(

            "ignore",

            message="Graph is not fully connected, spectral embedding" +

            " may not work as expected.",

            category=UserWarning)

        algorithm.fit(X_test)

        



    t1 = time.time()

    if hasattr(algorithm, 'labels_'):

        y_pred = algorithm.labels_.astype(np.int)

    else:

        y_pred = algorithm.predict(X_test)

    print("For algo= ",name)

    submission={}

    submission['custId']= tt.custId

    submission['Satisfied']= y_pred

    submission=pd.DataFrame(submission)

    submission.to_csv(name+'.csv',index=False)

#     print ("Accuracy : %.4g" % accuracy_score(y_pred, y_train))

#     print ("AUC Score : %.4g" % roc_auc_score(y_pred, y_train))