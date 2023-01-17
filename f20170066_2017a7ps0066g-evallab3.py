import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



%matplotlib inline



from sklearn import cluster, datasets, mixture

from sklearn.neighbors import kneighbors_graph

from sklearn.preprocessing import StandardScaler
training_df=pd.read_csv('./eval-lab-3-f464/train.csv')

training_df1=pd.read_csv('./eval-lab-3-f464/train.csv')

test_df=pd.read_csv('./eval-lab-3-f464/test.csv')

test_df1=pd.read_csv('./eval-lab-3-f464/test.csv');
training_df.info()

test_df.info()
training_df.head(20)
#Checking for missing values

missing_count = training_df.isnull().sum()

missing_count[missing_count > 0]
#Finding type and unique values in each column

training_df_dtype_nunique = pd.concat([training_df.dtypes, training_df.nunique()],axis=1)

training_df_dtype_nunique.columns = ["dtype","unique"]

training_df_dtype_nunique
#Dropping irrelevant rows

training_df=training_df.drop('custId',axis=1)

test_df=test_df.drop('custId',axis=1)
numerical_features=['tenure', 'MonthlyCharges', 'TotalCharges']

categorical_features=['gender', 'SeniorCitizen', 'Married', 'Children', 'TVConnection', 'Channel1', 'Channel2','Channel3', 'Channel4', 'Channel5', 'Channel6', 'Internet', 'HighSpeed', 'AddedServices','Subscription', 'PaymentMethod']

features=numerical_features+categorical_features

features
#Replacing spaces by mean

training_df.replace(" ", np.nan, inplace = True)

training_df.head(10)

training_df['TotalCharges'].fillna(value=training_df['MonthlyCharges'].mean(), inplace=True)

test_df.replace(" ", np.nan, inplace = True)

test_df.head(10)

test_df['TotalCharges'].fillna(value=test_df['MonthlyCharges'].mean(), inplace=True)
#Setting each column to its correct datatype

training_df[numerical_features] = training_df[numerical_features].astype("float")

training_df['SeniorCitizen']=training_df['SeniorCitizen'].astype('object')

test_df[numerical_features] = test_df[numerical_features].astype("float")

test_df['SeniorCitizen']=test_df['SeniorCitizen'].astype('object')
training_df.isnull().any().any()
training_df[numerical_features].describe()
training_df.info()
training_df[numerical_features].hist(bins=15, figsize=(15, 6));
sns.countplot(training_df['Satisfied'])
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()
# apply le on categorical feature columns

training_df[categorical_features] = training_df[categorical_features].apply(lambda col: le.fit_transform(col))

training_df[categorical_features].head(10)

test_df[categorical_features] = test_df[categorical_features].apply(lambda col: le.fit_transform(col))

test_df[categorical_features].head(10)
#Choosing more correlated features

cor=training_df.corr()

cor_label=abs(cor['Satisfied'])

final_features=cor_label[cor_label>0.1]

f=list(final_features.index)

X=training_df[f]

f.remove('Satisfied')

X_test=test_df[f]
X.info()
#Addressing class imbalance

from imblearn.over_sampling import SMOTE



smote = SMOTE(ratio='minority',k_neighbors=3, n_jobs=-1)

X_sm, y_sm = smote.fit_resample(X, X['Satisfied'])

X_sm

X_temp = pd.DataFrame(X_sm, columns=X.columns)

X_temp['Satisfied'] = y_sm

y=X_temp['Satisfied']



X_temp['Satisfied'].value_counts().plot(kind='bar', title='Count (Satisfied)');

X=X_temp.drop("Satisfied", axis=1)
X.info()
plt.figure(figsize=(16,16))

sns.heatmap(training_df.corr(),annot=True)
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.ensemble import ExtraTreesClassifier
bestfeatures = SelectKBest(score_func=chi2, k='all')

fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)



featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.nlargest(50,'Score'))  #print best feature
model = ExtraTreesClassifier()

model.fit(X,y)

print(model.feature_importances_)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(50).plot(kind='bar')
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

X[numerical_features] = scaler.fit_transform(X[numerical_features])

X[numerical_features].head()
y=y.astype("int32")
from sklearn import metrics
import time

import warnings



from sklearn import cluster, datasets, mixture

from sklearn.neighbors import kneighbors_graph

from sklearn.preprocessing import StandardScaler

from itertools import cycle, islice



np.random.seed(0)



default_base = {'quantile': .3,

                'eps': .3,

                'damping': .9,

                'preference': -200,

                'n_neighbors': 10,

                'n_clusters': 2,

                'min_samples': 20,

                'xi': 0.05,

                'min_cluster_size': 0.1}

params = default_base.copy()



X = StandardScaler().fit_transform(X)



# estimate bandwidth for mean shift

bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])



# connectivity matrix for structured Ward

connectivity = kneighbors_graph(

    X, n_neighbors=params['n_neighbors'], include_self=False)

# make connectivity symmetric

connectivity = 0.5 * (connectivity + connectivity.T)



# ============

# Create cluster objects

# ============

ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)

two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])

ward = cluster.AgglomerativeClustering(

    n_clusters=params['n_clusters'], linkage='ward',

    connectivity=connectivity)

spectral = cluster.SpectralClustering(

    n_clusters=params['n_clusters'], eigen_solver='arpack',

    affinity="nearest_neighbors")

dbscan = cluster.DBSCAN(eps=params['eps'])

optics = cluster.OPTICS(min_samples=params['min_samples'],

                       xi=params['xi'],

                       min_cluster_size=params['min_cluster_size'])

affinity_propagation = cluster.AffinityPropagation(

    damping=params['damping'], preference=params['preference'])

average_linkage = cluster.AgglomerativeClustering(

    linkage="average", affinity="cityblock",

    n_clusters=params['n_clusters'], connectivity=connectivity)

birch = cluster.Birch(threshold=0.6, branching_factor=30, n_clusters=params['n_clusters'])

gmm = mixture.GaussianMixture(

    n_components=params['n_clusters'], covariance_type='full')



clustering_algorithms = (

    ('MiniBatchKMeans', two_means),

    ('AffinityPropagation', affinity_propagation),

    ('MeanShift', ms),

    ('SpectralClustering', spectral),

    ('Ward', ward),

    ('AgglomerativeClustering', average_linkage),

    ('DBSCAN', dbscan),

    ('OPTICS', optics),

    ('Birch', birch),

    ('GaussianMixture', gmm)

)



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

        algorithm.fit(X)



    t1 = time.time()

    if hasattr(algorithm, 'labels_'):

        y_pred = algorithm.labels_.astype(np.int)

    else:

        y_pred = algorithm.predict(X)



    auc=metrics.roc_auc_score(y, y_pred) #Calculating auc score

    print(algorithm,': ',auc)
gmm.fit(X_test)

y_test_pred = gmm.predict(X_test)

print(y_test_pred)



#Constructing final dataframe

df=pd.DataFrame(columns=['custId', 'Satisfied'])

df['custId']=test_df1['custId']

df['Satisfied']=y_test_pred
birch.fit(X_test)

y_test_pred = birch.predict(X_test)

print(y_test_pred)



#Constructing final dataframe

df1=pd.DataFrame(columns=['custId', 'Satisfied'])

df1['custId']=test_df1['custId']

df1['Satisfied']=y_test_pred
# df.to_csv('predictions.csv', index=False)