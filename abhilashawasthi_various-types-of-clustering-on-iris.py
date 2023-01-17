%matplotlib inline
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
# Loading iris dataset
iris = pd.read_csv("../input/Iris.csv")
# A glimpse of the dataset
print(iris.head(2))
print(iris.tail(2))
# Shape of the dataframe
# There are 150 rows where each row corresponds to dimension of an iris flower 
# and 6 columns Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm & Species.  
print(iris.shape)
# Each type of iris flower has 50 observations
print(iris.Species.value_counts())
# Label encoding of Species column
le = LabelEncoder()
le.fit(iris['Species'])
print(list(le.classes_))
iris['Species'] = le.transform(iris['Species'])
# So after encoding 0 refers to Iris-setosa, 1 refers to Iris-versicolor, 2 refers to Iris-virginica
print(iris['Species'][0:5])
print(iris['Species'][50:55])
print(iris['Species'][100:105])
# Converting into numpy matrix
iris_matrix = pd.DataFrame.as_matrix(iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']])
# Building KMeans Clustering model with 3 clusters since we know the iris dataset has three classes of iris flower
# And so ideally thinking they should be in 3 different clusters (but this maynot be the case)
cluster_model = KMeans(n_clusters=3, random_state=10)
cluster_labels = cluster_model.fit_predict(iris_matrix)
adj_rand_score = adjusted_rand_score(iris['Species'],cluster_labels)
print ("For n_clusters = 3 ", 
          "The adjusted rand_score is:", adj_rand_score)
# In case we don't know prior the true labels in the dataset, and can't decide on number of clusters which should be given to KMeans.
# We can then build many KMeans clustering model and check the Silhouette Score of each of the clustering model.
# A higher Silhouette Coefficeint score relates to a model with better defined clusters.
# For more information on Silhouette Score look at http://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient
# Below we build 10 KMeans clustering model & check the Silhouette Score & Adjusted Rand score of each of the model. 
for n_clusters in range(2,11):
    cluster_model = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = cluster_model.fit_predict(iris_matrix)
    silhouette_avg = silhouette_score(iris_matrix,cluster_labels,metric='euclidean')
    adj_rand_score = adjusted_rand_score(iris['Species'],cluster_labels)
    print("For n_clusters =", n_clusters, 
          "The average silhouette_score is:", silhouette_avg)
    print ("For n_clusters =", n_clusters, 
          "The adjusted rand_score is:", adj_rand_score)
# From above results it can be seen that Silhouette Score is highest for 2 clusters 
# and Adjusted Rand Score is greatest for 3 clusters.
# Silhouette Score is highest for 2 clusters because Silhouette score is more for better defined & separated clusters.
# And as can be seen in below plot there are 2 distinct & better separated clusters.
# Adjusted Rand Score measures the similarity between true labels & predicted labels(obtained by clustering).
# Plot Copied from https://www.kaggle.com/benhamner/d/uciml/iris/python-data-visualizations/notebook
from pandas.tools.plotting import radviz
radviz(iris.drop("Id", axis=1), "Species")
# Now let us apply Hierarchical Clustering which is implemented in scikit-learn as AgglomerativeClustering
for n_clusters in range(2,11):
    cluster_model = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean',linkage='ward')
    cluster_labels = cluster_model.fit_predict(iris_matrix)
    silhouette_avg = silhouette_score(iris_matrix,cluster_labels,metric='euclidean')
    adj_rand_score = adjusted_rand_score(iris['Species'],cluster_labels)
    print("For n_clusters =", n_clusters, 
          "The average silhouette_score is:", silhouette_avg)
    print ("For n_clusters =", n_clusters, 
          "The adjusted rand_score is:", adj_rand_score)
# Will Try to improve this notebook more later.