import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import os
import scipy.stats as stats
from scipy.stats import zscore
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/soil-nutrients-data/Soil Nutrients Data.csv')
df.info()
df['S']=df['S'].astype(float)
df['P']=df['P'].astype(float)
df.describe()
for i in list(df.columns):
    sns.boxplot(df[i])
    plt.show()
df.corr()
df[df['N']> 500]
df[df['Lime']> 100]
df[df['P']> 400]
df.loc[(df['N']> 500) | (df['Lime']> 100) | (df['P']> 400),:]
df2 = df.drop(df[(df['N']> 500) | (df['Lime']> 100) | (df['P']> 400)].index, axis=0)
df2.head()
df2[(df2['S']> 170)]
df2[df2['Ca']> 45.10]
df3 = df2.drop(df2[(df2['S']> 170) | (df2['Ca']> 45.10)].index, axis=0)
for i in list(df3.columns):
    sns.boxplot(df3[i])
    plt.show()
#Removing outliers
df3_scaled= df3.apply(zscore)
df3_scaled.head()
from sklearn.cluster import KMeans
model = KMeans()
cluster_range = range( 1, 15 )
cluster_errors = []
for num_clusters in cluster_range:
  model = KMeans( num_clusters, n_init = 20 )
  model.fit(df3_scaled)
 # labels = clusters.labels_
 # centroids = clusters.cluster_centers_
  cluster_errors.append( model.inertia_ )
clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )
clusters_df
# Elbow plot

plt.figure(figsize=(18,10))
plt.plot( clusters_df['num_clusters'], clusters_df['cluster_errors'], marker = "o" )
from scipy.cluster.hierarchy import linkage, dendrogram
plt.figure(figsize=[10,10])
merg = linkage(df3, method='ward')
dendrogram(merg, leaf_rotation=90)
plt.title('Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distances')
plt.show() #Hierarchical Cluster
from __future__ import print_function
%matplotlib inline


from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

print(__doc__)

# Generating the sample data from make_blobs
# This particular setting has one distinct cluster and 3 clusters placed close
# together.
X, y = make_blobs(n_samples=500,
                  n_features=2,
                  centers=4,
                  cluster_std=1,
                  center_box=(-10.0, 10.0),
                  shuffle=True,
                  random_state=1)  # For reproducibility

range_n_clusters = [2, 3, 4, 5, 6,7,8,9,10]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(df3) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(df3)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(df3, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(df3, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.Spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    '''
    # 2nd Plot showing the actual clusters formed
    colors = cm.Spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(df3[:, 0], df3[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors)

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1],
                marker='o', c="white", alpha=1, s=200)

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
    '''
    plt.show()
kmeans_4 = KMeans(n_clusters=4, n_init= 10, random_state=3)
kmeans_4.fit(df3_scaled)
centroids_4 = kmeans_4.cluster_centers_
print(centroids_4)
centroid4_df= pd.DataFrame(centroids_4, columns = list(df3_scaled))
centroid4_df
df_class4 = pd.DataFrame(kmeans_4.labels_ , columns= ['class'])
df_class4['class']= df_class4['class'].astype('category') 
df_labelled = df.join(df_class4)
df_labelled.tail(9)
df_labelled.dropna(inplace=True)
df_labelled['class'].value_counts()
X = df_labelled.iloc[:,:-1]
y = df_labelled.iloc[:,-1]
#from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.3,random_state=0)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
Xs = ss.fit_transform(X)

X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
from sklearn.metrics import confusion_matrix , accuracy_score , roc_auc_score , roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
lr= LogisticRegression(fit_intercept= True, solver= 'liblinear')

lr.fit(X_train, y_train)
y_train_pred= lr.predict(X_train)

print('Confusion Matrix: ', '\n', confusion_matrix(y_train, y_train_pred))
print('Overall Accuracy -Train: ', accuracy_score(y_train, y_train_pred))
y_test_pred = lr.predict(X_test)
y_test_prob = lr.predict_proba(X_test)[:,1]
print('classification report - Train \n', classification_report(y_train,y_train_pred))
print('classification report - Test \n ', classification_report(y_test,y_test_pred))
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMClassifier

from sklearn import model_selection

import lightgbm as lgm

import xgboost as xgb

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier
models=[]
models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
models.append(('RandomForestClassifer', RandomForestClassifier()))
models.append(('Lightgbm',LGBMClassifier()))
models.append(('xgboost',XGBClassifier()))
results=[]
names=[]
for name,model in models:
    kfold= model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
    cv_results= model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=None)
    results.append(cv_results)
    names.append(name)
    print(name, cv_results.mean())
