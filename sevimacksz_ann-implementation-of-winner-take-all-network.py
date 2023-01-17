from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

import numpy as np

from itertools import product, combinations
# Center points of the clusters

centers = [[1, 1, 1], [-1, -1, 1], [1, -1, -1]]

n_datapoints = 30

n_testpoints = 3

dim = 3

sigma = 0.3

input_data = []

test_data = []



# Creating 3 clusters with normal distribution about the center points, they have unit length

cluster1 = np.random.normal(centers[0], sigma, size=(n_datapoints+n_testpoints, dim))

cluster1 = [(vec / np.linalg.norm(vec)).tolist() for vec in cluster1]

input_data.append(cluster1[:n_datapoints])

test1 = np.array(cluster1[n_datapoints:])

test_data.append(test1)

cluster1 = np.array(cluster1)



cluster2 = np.random.normal(centers[1], sigma, size=(n_datapoints+n_testpoints, dim))

cluster2 = [(vec / np.linalg.norm(vec)).tolist() for vec in cluster2]

input_data.append(cluster2[:n_datapoints])

test2 = np.array(cluster2[n_datapoints:])

test_data.append(test2)

cluster2 = np.array(cluster2)



cluster3 = np.random.normal(centers[2], sigma, size=(n_datapoints+n_testpoints, dim))

cluster3 = [(vec / np.linalg.norm(vec)).tolist() for vec in cluster3]

input_data.append(cluster3[:n_datapoints])

test3 = np.array(cluster3[n_datapoints:])

test_data.append(test3)

cluster3 = np.array(cluster3)
# Creating shuffled input data

input_data = np.array(input_data)

input_data.shape = (n_datapoints * dim, dim)

np.random.shuffle(input_data)



# Creating shuffled test data

test_data = np.array(test_data)

test_data.shape = (n_testpoints * dim, dim)

np.random.shuffle(test_data)
fig = plt.figure(figsize=(10,10))

ax = Axes3D(fig)



# Draws sphere

u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

x = np.cos(u)*np.sin(v)

y = np.sin(u)*np.sin(v)

z = np.cos(v)

ax.plot_wireframe(x, y, z, color="y")



# Scatters clusters on the sphere

ax.scatter(cluster1[:,0], cluster1[:,1], cluster1[:,2], c='r', marker='o')

ax.scatter(cluster2[:,0], cluster2[:,1], cluster2[:,2], c='b', marker='o')

ax.scatter(cluster3[:,0], cluster3[:,1], cluster3[:,2], c='g', marker='o')



plt.show()
class WtaNet():

    def __init__(self, l_rate=0.2, n_iter=500):

        self.l_rate = l_rate

        self.n_iter = n_iter

        

    def normalize(self, weight):

        weight /= np.linalg.norm(weight)

        return weight

        

    def train(self, input_data):

        

        self.input_data = input_data

        

        # Initialize random weights that have unit length

        self.weights = np.random.normal(0, sigma, size=(dim, dim))

        self.weights = np.array([(vec / np.linalg.norm(vec)).tolist() for vec in self.weights])

        

        # Plots the initial random weights

        ax.scatter(self.weights[0,0], self.weights[0,1], self.weights[0,2], c='r', marker='x', s=25**2)

        ax.scatter(self.weights[1,0], self.weights[1,1], self.weights[1,2], c='b', marker='x', s=25**2)

        ax.scatter(self.weights[2,0], self.weights[2,1], self.weights[2,2], c='g', marker='x', s=25**2)

        

        

        for i in range(self.n_iter):

            

            for x in self.input_data:

                # This is the formula given in the class but it did not work for my implementation, 

                # instead I used the following

                    #winner_unit = np.argmax(weights.T * x) // 3

                

                # Picks the winner unit by looking the closest distance between weights and the input vector

                winner_unit = np.argmin([np.linalg.norm(i - x) for i in self.weights])

                

                # Updates winner weight and normalizes

                self.weights[winner_unit] += self.l_rate * (x - self.weights[winner_unit]) 

                self.weights[winner_unit] = self.normalize(self.weights[winner_unit])

                

                # Plots the path that weights follow during learning

                if i%100 == 0: 

                    ax.scatter(self.weights[0,0], self.weights[0,1], self.weights[0,2], c='r', marker='x', s=7**2)

                    ax.scatter(self.weights[1,0], self.weights[1,1], self.weights[1,2], c='b', marker='x', s=7**2)

                    ax.scatter(self.weights[2,0], self.weights[2,1], self.weights[2,2], c='g', marker='x', s=7**2)

       

        return self.weights

    

    def winner(self, test_input):

        winners = []

        for x in test_input:

            winner_unit = np.argmin([np.linalg.norm(i - x) for i in self.weights])

            winners.append(self.weights[winner_unit].tolist())

            

        return np.array(winners)

    

fig = plt.figure(figsize=(11,11))

ax = Axes3D(fig)



u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

x = np.cos(u)*np.sin(v)

y = np.sin(u)*np.sin(v)

z = np.cos(v)

ax.plot_wireframe(x, y, z, color="y")



ax.scatter(cluster1[:,0], cluster1[:,1], cluster1[:,2], c='r', marker='o')

ax.scatter(cluster2[:,0], cluster2[:,1], cluster2[:,2], c='b', marker='o')

ax.scatter(cluster3[:,0], cluster3[:,1], cluster3[:,2], c='g', marker='o')



wta = WtaNet()

winners = wta.train(input_data)



ax.scatter(winners[0,0], winners[0,1], winners[0,2], c='r', marker='X', s=60**2)

ax.scatter(winners[1,0], winners[1,1], winners[1,2], c='b', marker='X', s=60**2)

ax.scatter(winners[2,0], winners[2,1], winners[2,2], c='g', marker='X', s=60**2)

plt.show()
results = wta.winner(test_data)

print('Results: \n', results)



fig = plt.figure(figsize=(10,10))

ax = Axes3D(fig)



u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

x = np.cos(u)*np.sin(v)

y = np.sin(u)*np.sin(v)

z = np.cos(v)

ax.plot_wireframe(x, y, z, color="y")



ax.scatter(test1[:,0], test1[:,1], test1[:,2], c='r', marker='o')

ax.scatter(test2[:,0], test2[:,1], test2[:,2], c='b', marker='o')

ax.scatter(test3[:,0], test3[:,1], test3[:,2], c='g', marker='o')



ax.scatter(results[:,0], results[:,1], results[:,2], c='r', marker='x', s=80**2)

plt.show()