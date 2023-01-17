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
# Read Mushroom Dataset

file = "/kaggle/input/mushroom-classification/mushrooms.csv"

df = pd.read_csv(file)

df.head()
# Check for NA 

df.isna().sum()
# check for available classes for classification

df['class'].unique()
# Label encoder to convert the dataset

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

for col in df.columns:

    df[col] = le.fit_transform(df[col])

df.head()
# Visualizing outliers if any

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

plt.figure(figsize=(60, 6))

sns.boxplot(x="variable", y="value", data=pd.melt(df))
df.describe()
# Plotting correlation matrix which depicts that, there is good amount of correlation among the features.

plt.figure(figsize=(25, 15))

sns.heatmap(df.corr(),

            vmin=-1,

            cmap='coolwarm',

            annot=True);
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# As our dataset has nominal values, we will be performing the one hot encoding operation and making it more sparse.

file = "/kaggle/input/mushroom-classification/mushrooms.csv"

df = pd.read_csv(file)

df_one_hot = pd.concat([df.iloc[:,0], pd.get_dummies(df.iloc[:,1:23])], axis=1)

features = df_one_hot.iloc[:,1:118]

label = df_one_hot.iloc[:,0] 

label = le.fit_transform(label)
from sklearn.decomposition import PCA

pca = PCA(svd_solver='randomized', random_state=123) #instantiate.

pca.fit(features)

fig = plt.figure(figsize = (20,5))

ax = plt.subplot(121)

plt.plot(pca.explained_variance_ratio_)

plt.xlabel('principal components')

plt.ylabel('explained variance')



ax2 = plt.subplot(122)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance')



plt.show()
# what percentage of variance in data can be explained by first 2,3 and 4 principal components respectively?

pca.explained_variance_ratio_[0:50].sum().round(3)
df_pca = pca.transform(features)# our data transformed with new features as principal components.

df_pca = df_pca[:, 0:2] # Since we require first two principal components only.

df_s = scaler.fit_transform(df_pca) # s in df_s stands for scaled.

sns.pairplot(pd.DataFrame(df_s))
# Checks the possibility in the dataset for clustering and values depicts that the dataset is indeed highly clusterable



from sklearn.neighbors import NearestNeighbors

from random import sample

from numpy.random import uniform

import numpy as np

from math import isnan

 

def hopkins(X):

    d = X.shape[1]

    #d = len(vars) # columns

    n = len(X) # rows

    m = int(0.1 * n) 

    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)

 

    rand_X = sample(range(0, n, 1), m)

 

    ujd = []

    wjd = []

    for j in range(0, m):

        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)

        ujd.append(u_dist[0][1])

        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)

        wjd.append(w_dist[0][1])

 

    H = sum(ujd) / (sum(ujd) + sum(wjd))

    if isnan(H):

        print(ujd, wjd)

        H = 0

 

    return H



hopkins(pd.DataFrame(df_s))
from sklearn.cluster import KMeans # import.



# silhouette scores to choose number of clusters.

from sklearn.metrics import silhouette_score

def sil_score(df):

    sse_ = []

    for k in range(2, 15):

        kmeans = KMeans(n_clusters=k, random_state=123).fit(df_s) # fit.

        sse_.append([k, silhouette_score(df, kmeans.labels_)])

    plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1])



sil_score(df_s)
# Elbow method to find possible clusters.

def plot_ssd(df):

    ssd = []

    for num_clusters in list(range(1,19)):

        model_clus = KMeans(n_clusters = num_clusters, max_iter=50, random_state=123)

        model_clus.fit(df)

        ssd.append(model_clus.inertia_)

    plt.plot(ssd)



plot_ssd(df_s)
# K-means with K=3.

km2c = KMeans(n_clusters=3, max_iter=50, random_state=93)

km2c.fit(df_s)

df_dummy = pd.DataFrame.copy(df)

dfkm2c = pd.concat([df_dummy, pd.Series(km2c.labels_)], axis=1)

dfkm2c.rename(columns={0:'Cluster ID'}, inplace=True)

df_dummy = pd.DataFrame.copy(pd.DataFrame(df_s))

dfpcakm2c = pd.concat([df_dummy, pd.Series(km2c.labels_)], axis=1)

dfpcakm2c.columns = ['PC1', 'PC2', 'Cluster ID']

sns.pairplot(data=dfpcakm2c, vars=['PC1', 'PC2'], hue='Cluster ID')
# K-means with K=5.

km2c = KMeans(n_clusters=5, max_iter=50, random_state=93)

km2c.fit(df_s)

df_dummy = pd.DataFrame.copy(df)

dfkm2c = pd.concat([df_dummy, pd.Series(km2c.labels_)], axis=1)

dfkm2c.rename(columns={0:'Cluster ID'}, inplace=True)

df_dummy = pd.DataFrame.copy(pd.DataFrame(df_s))

dfpcakm2c = pd.concat([df_dummy, pd.Series(km2c.labels_)], axis=1)

dfpcakm2c.columns = ['PC1', 'PC2', 'Cluster ID']

sns.pairplot(data=dfpcakm2c, vars=['PC1', 'PC2'], hue='Cluster ID')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features,label,test_size=0.2,random_state=4)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_curve, auc, classification_report

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.gaussian_process import GaussianProcessClassifier

import xgboost

from sklearn import svm, tree

from sklearn import metrics



classifiers = []

nb_model = GaussianNB()

classifiers.append(("Gausian Naive Bayes Classifier",nb_model))

lr_model= LogisticRegression()

classifiers.append(("Logistic Regression Classifier",lr_model))

# sv_model = svm.SVC()

# classifiers.append(sv_model)

dt_model = tree.DecisionTreeClassifier()

classifiers.append(("Decision Tree Classifier",dt_model))

rf_model = RandomForestClassifier()

classifiers.append(("Random Forest Classifier",rf_model))

xgb_model = xgboost.XGBClassifier()

classifiers.append(("XG Boost Classifier",xgb_model))

lda_model = LinearDiscriminantAnalysis()

classifiers.append(("Linear Discriminant Analysis", lda_model))

gp_model =  GaussianProcessClassifier()

classifiers.append(("Gaussian Process Classifier", gp_model))

ab_model =  AdaBoostClassifier()

classifiers.append(("AdaBoost Classifier", ab_model))
cv_scores = []

names = []

for name, clf in classifiers:

    print(name)

    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  

    y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.

    print("Model Score : ",clf.score(X_test, y_pred))

    print("Number of mislabeled points from %d points : %d"% (X_test.shape[0],(y_test!= y_pred).sum()))

    scores = cross_val_score(clf, features, label, cv=10, scoring='accuracy')

    cv_scores.append(scores)

    names.append(name)

    print("Cross validation scores : ",scores.mean())

    confusion_matrix=metrics.confusion_matrix(y_test,y_pred)

    print("Confusion Matrix \n",confusion_matrix)

    classification_report = metrics.classification_report(y_test,y_pred)

    print("Classification Report \n",classification_report)
#Model Comparison

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

fig = plt.figure(figsize=(25, 10))

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(cv_scores)

ax.set_xticklabels(names)

plt.show()