import numpy as np

import pandas as pd



class KMeans:

    

    def __init__(self, X, K=2, num_iters=250):

        self.X = np.array(X)

        self.K = K

        self.num_iters = num_iters

        self.clusters = dict()

        if len(self.X.shape) == 1:

            self.num_features = 1

        else:

            self.num_features = self.X.shape[1]

        random_indexes = np.random.randint(0, X.shape[0], size=(self.K, 1))

        for i in range(K):

            self.clusters[i] = {"members": [], "centroid": self.X[random_indexes[i]]}

        self._cluster()

        

    def _euclidean(self, x, y):

        z = x - y

        z = np.reshape(z, (-1,))

        return np.sqrt(np.dot(z, z))

    

    def _reassign_clusters(self):

        for i in range(self.K):

            self.clusters[i]["members"] = []

        for i in range(X.shape[0]):

            xi = self.X[i]

            distances = []

            for k in range(self.K):

                centroid = self.clusters[k]["centroid"]

                distances.append((k, self._euclidean(xi, centroid)))

            assigned_cluster = min(distances, key=lambda e: e[1])[0]

            self.clusters[assigned_cluster]["members"].append(xi)

            

    def _update_centroids(self):

        self.converged = True

        for k in range(self.K):

            cluster = self.clusters[k]["members"]

            new_centroid = np.zeros(self.num_features)

            for x in cluster:

                    new_centroid += x

            new_centroid = new_centroid/len(cluster)

            self.converged = np.array_equal(self.clusters[k]["centroid"], new_centroid)

            self.clusters[k]["centroid"] = new_centroid

            

    def _cluster(self):

        for _ in range(self.num_iters):

            self._reassign_clusters()

            self._update_centroids()

            if self.converged:

                print("Converged early!")

                break

                



# if __name__ == "__main__":



#   data = pd.read_csv("iris.csv")

#   X = data.iloc[:, :2]



#   km = KMeans(X, K=3)

  

#   def split_list_of_arrays(list_of_arrays):

#     x = []

#     y = []

#     for array in list_of_arrays:

#         x.append(array[0])

#         y.append(array[1])

#     return x, y

    

#   cluster_0_x, cluster_0_y = split_list_of_arrays(km.clusters[0]["members"])

#   cluster_1_x, cluster_1_y = split_list_of_arrays(km.clusters[1]["members"])

#   cluster_2_x, cluster_2_y = split_list_of_arrays(km.clusters[2]["members"])

  

#   plt.scatter(cluster_0_x, cluster_0_y)

#   plt.scatter(cluster_1_x, cluster_1_y)

#   plt.scatter(cluster_2_x, cluster_2_y)

#   plt.show()