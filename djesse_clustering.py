import pandas as pd



# Read training data

training_binary = pd.read_csv("../input/binary-cooking/training_binary.csv")

y_train = training_binary['cuisine']

x_train = training_binary.drop(['cuisine', 'id'],axis=1)
def create_ingredient_counter(x, y, binary_data):

    # Create dataframe to keep track of number of times an ingredient is used in a certain cuisine

    ingredient_counter = pd.DataFrame(set(y), columns=['cuisine'])



    # Iterate over all ingredients

    for ingredient, data in x.iteritems():

        column_data = binary_data[ingredient]



        # Init dictionary to keep track of ingredient occurences

        cuisine_dict = {key: 0 for key in list(set(y))}



        # Iterate over all scores in ingredient column

        for index, value in enumerate(column_data):



            # If an ingredient occurs in a specific cuisine, update its value

            if value == 1:

                cuisine = y.iloc[index]



                cuisine_dict[cuisine] += 1



        # Add the values as a column to the dataframe

        ingredient_counter[ingredient] = cuisine_dict.values()

        

    return ingredient_counter

    
# Create an ingredient counter for the training set

ingredient_counter_train = create_ingredient_counter(x_train, y_train, training_binary)
from scipy import sparse

from sklearn.feature_extraction.text import TfidfTransformer



# Normalize and standardize ingredient counter for training set

no_cuisine_data_train = ingredient_counter_train.loc[:, ingredient_counter_train.columns != 'cuisine'].values

countsMatrix_train = sparse.csr_matrix(no_cuisine_data_train)

transformer_train = TfidfTransformer()

tfidf_train = transformer_train.fit_transform(countsMatrix_train)
from sklearn.decomposition import PCA



# Perform pca on both the training set to get 2 principal components

pca_train = PCA(n_components=2)

pca_train.fit(tfidf_train.toarray())



pca_counter_data_train = pca_train.transform(tfidf_train.toarray())
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

import numpy as np



cluster_distances = []



# Cluster using K means for K = 1-10 and save the sum of squared distances

for i in range (1,10):

    kmeans = KMeans(n_clusters=i)

    kmeans.fit(pca_counter_data_train)

    cluster_distances.append(kmeans.inertia_)

    

# Plot the sum of squared distances to determine the value for K    

plt.plot(cluster_distances)

plt.title("Sum of squared errors plotted as a function of K")

plt.xlabel("K")

plt.ylabel("Sum of squared errors")

plt.savefig('k-means_k.png')

plt.show()
# Compute clusters using K-means with ideal number of K

kmeans = KMeans(n_clusters=2)

kmeans.fit(pca_counter_data_train)

kmeans_labels = kmeans.predict(pca_counter_data_train)



# Determine colors

color_spectrum = []



for i in range(5):

    color_spectrum.append(np.random.rand(3,))



colors = []



for label in kmeans_labels:

    colors.append(color_spectrum[label])

    

# Plot clusters

fig, ax = plt.subplots(figsize=(8,8))

    

ax.scatter(pca_counter_data_train[:,0], pca_counter_data_train[:,1], c=colors, s=300)

ax.set_title("K-means clustering using K=2")

ax.set_xlabel('PC1')

ax.set_ylabel('PC2')



# Plot names of cuisines

for i, cuisine in enumerate(set(y_train)):

    ax.annotate(cuisine, (pca_counter_data_train[:,0][i] + 0.03, pca_counter_data_train[:,1][i]))



fig.tight_layout()

plt.savefig('pc1_pc2_k-means.png')

plt.show()
from sklearn.cluster import DBSCAN

from sklearn.neighbors import NearestNeighbors



# Init NearestNeighbors with n_neighbors = 4 and fit on the training data

neigh = NearestNeighbors(n_neighbors=4)

nbrs = neigh.fit(pca_counter_data_train)



# Find the nearest neighbors of the data points and determine the distances

distances, indices = nbrs.kneighbors(pca_counter_data_train)



distances = np.sort(distances, axis=0)

distances = distances[:,1]



plt.plot(distances)

plt.title("Distance of points plotted to their 4-th nearest neighbor")

plt.xlabel("Number of points sorted according to distance to 4th nearest neighbor")

plt.ylabel("Distance to 4th nearest neighbor")



plt.savefig('pc1_pc2_dbscan.png')

plt.show()
# Compute clusters using DBSCAN, epsilon = 0.17 & minpts = 4

m = DBSCAN(eps=0.17, min_samples=4)

m.fit(pca_counter_data_train)



dbscan_labels = m.labels_



# Determine colors

for i in range(3):

    color_spectrum.append(np.random.rand(3,))



colors = []



for label in dbscan_labels:

    colors.append(color_spectrum[label])



# Plot clusters

fig, ax = plt.subplots(figsize=(8,8))

    

ax.scatter(pca_counter_data_train[:,0], pca_counter_data_train[:,1], c=colors, s=300)

ax.set_xlabel('PC1')

ax.set_ylabel('PC2')



# Plot names of cuisines

for i, cuisine in enumerate(set(y_train)):

    ax.annotate(cuisine, (pca_counter_data_train[:,0][i] + 0.03, pca_counter_data_train[:,1][i]))



plt.savefig('pc1_pc2_dbscan_result.png')

plt.show()
from sklearn.metrics import silhouette_score



# Calculate silhouette score for K-means clustering

print("Silhouette coefficient score K-means clustering: {}\n".format(silhouette_score(pca_counter_data_train, kmeans_labels)))



# Calculate silhouette score for DBSCAN clustering

print("Silhouette coefficient score DBSCAN: {}".format(silhouette_score(pca_counter_data_train, dbscan_labels)))
from scipy import spatial



# Calculate cosine similarity between 2 cuisines

def calculate_cosine_sim(row_1, row_2):

    return 1 - spatial.distance.cosine(row_1, row_2)



def calculate_sim_scores(ingredient_counter, y):

    sim_scores = {}

    cuisine_list = list(set(y))

    

    # Calculate cosine similarity of cuisine compared to other cuisines including itself

    for i in range(len(cuisine_list)):



        # Get values for cuisine 1

        row_1 = ingredient_counter.iloc[i].iloc[1:]

        row_sim_scores = {}



        # Calculate cosine similarity with other cuisines

        for j in range(20):



            # Get values for other row to compare with

            row_2 = ingredient_counter.iloc[j].iloc[1:]

            row_sim_scores[cuisine_list[j]] = calculate_cosine_sim(row_1, row_2)



        sim_scores[cuisine_list[i]] = row_sim_scores

    return sim_scores
sim_scores_train = calculate_sim_scores(ingredient_counter_train, y_train)

sim_matrix_train = pd.DataFrame.from_dict(sim_scores_train)



# Show similarity matrix for all cuisines

sim_matrix_train