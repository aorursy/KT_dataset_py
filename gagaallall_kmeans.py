#https://kodekloud.com/p/docker-labs
import numpy as np

import matplotlib.pyplot as plt

%matplotlib notebook
means = [[1,5] , [5,2] , [10,9] , [14,15] , [20,-4] , [-5, -12]]

cov = [[8,0],[0,8]]

N = 300

K = 5



X = np.random.multivariate_normal(means[0] , cov , N)

for i in range(1, len(means)):

    X1 = np.random.multivariate_normal(means[i] , cov , N)

    X = np.concatenate((X, X1) , axis = 0)

X.shape
plt.figure(figsize = (8,8))

plt.subplot(111)

plt.scatter(X[:,0], X[:,1])

plt.title('Đầu vào')
import sys

import time

class KMeans:

    def __init__(self, n_clusters = 3 , random_state='kmean++' , plot_state = True , max_iteration = 300):

        '''

        X: {array[]}

            The data for cluster

        n_clusters: int

            Number of clusters.

        random_state : 'random' , 'kmean++'

            Initializing the centroids.

        plot_state : True/False , default: Flase

            True : Visualizing data , centroids.

        '''

        self.n_clusters = n_clusters

        self.random_state = random_state    

        self.plot_state = plot_state

        self.max_iteration = max_iteration

        

    def _distance(self,dis1,dis2):

        dist = np.sum((dis1-dis2)**2 , axis = 1)

        return dist

        

    def init_centers(self):

        if self.random_state == 'random':

            centroids = X[np.random.choice(self.X.shape[0] , self.n_clusters)]

            return centroids

        

        elif self.random_state == 'kmean++':

            centroids = []

            centroids.append( self.X[np.random.randint(self.X.shape[0]) , :] )

            

            for next_center in range(self.n_clusters-1):

                dist = []

                for point in range(self.X.shape[0]):

                    d = sys.maxsize

                    temp_dist = self._distance(np.array(centroids) , self.X[point])

                    d = min(d, min(temp_dist))

                    dist.append(d)

                                            

                centroids.append(self.X[np.argmax(dist),:])

            centroids = np.array(centroids)

            return centroids

        else:

            print('None type of {} init_center.'.format(self.init_centers))

    

    def __labeling__(self,y, centroids):

        for i in range (self.X.shape[0]):

            y[i] =np.argmin(self._distance(centroids , self.X[i]))

        return y

    

    def __updateCenters__(self,y, centroids):

        new_centroids = np.zeros(centroids.shape)

        for i in range(centroids.shape[0]):

            new_centroids[i] = np.mean(X[np.where(y==i)] , axis = 0)

        return new_centroids

    

    def fit(self,X):

        self.X = X

        centroids = self.init_centers()

        epoch = 0

        y = np.zeros((self.X.shape[0]))

        if self.plot_state == True:

            fig = plt.figure(figsize = (8,8))

            ax = fig.add_subplot(111)

        

            ax.scatter(self.X[:,0] , self.X[:,1])

            ax.scatter(centroids[:,0] , centroids[:,1] , c ='black', s=100)

        while True:

            y = self.__labeling__(y, centroids)

            new_centroids = self.__updateCenters__(y, centroids)

                     

            if set([tuple(a) for a in centroids]) == set([tuple(a) for a in new_centroids]) or epoch == self.max_iteration:

                break

            else:

                

                if self.plot_state == True:

                    ax.cla()

                    for i in range(centroids.shape[0]):

                        X_draw = self.X[np.where(y == i)]

                        

                        ax.scatter(X_draw[:,0] , X_draw[:,1])

                    ax.scatter(new_centroids[:,0] , new_centroids[:,1] , c = 'black' , s = 50)

                    ax.set_title('epoch: {}'.format(epoch))

                    fig.canvas.draw()

                epoch+=1

                centroids = new_centroids

        self.centroids = centroids

        self.y = y

        self.epoch = epoch

        return self.centroids , self.y , self.epoch

    def cluster_centers(self):

        return self.centroids
model = KMeans(n_clusters = 4, random_state = 'kmean++', plot_state = True)

centroids , y , epoch = model.fit(X)
def distance(XA,XB,metric = 'euclidean'):

    XA = np.asarray(XA)

    XB = np.asarray(XB)

    if len(XA.shape) != 2:

        raise ValueError('XA must be 2 dimentional array')

    if len(XB.shape) != 2:

        raise ValueError('XB must be 2 dimentional array')

    if XA.shape[-1] != XB.shape[-1]:

        raise ValueError('the second dimention of XA & XB must be the same')

    d = []

    for i in range (XA.shape[0]):

        d.append( np.sqrt ( np.sum((XB - XA[i])**2 , axis = 1) ))

    return np.asarray(d)      
class k_clusters_optimizing:

    def __init__(self,X,k,optimize_state = 'euclidean',random_state = 'kmean++', plot_state = False):

        self.X = np.asarray(X)

        self.plot_state = plot_state

        self.random_state = random_state

        if type(k) == int and k >= 3:

            self.k = [k for k in range(2,k + 1)]

        elif type(k) == list:

            self.k = k

        else:

            raise ValueError('Cluster k must be interger which is greater than 2 or list')

        self.optmize_state = optimize_state

        

    def WSS(self , k_index ):

        model = KMeans(n_clusters = k_index, random_state = self.random_state , plot_state = self.plot_state)

        centroids , y , epoch = model.fit(self.X)

        wss_k = 0

        for centroid_index in range(centroids.shape[0]):

            cluster_data = self.X[np.where(y == centroid_index)]

            dis = distance(cluster_data , [centroids[centroid_index]])

            sum_cluster = np.sum(dis)

            wss_k = wss_k + sum_cluster

        return wss_k, epoch

            

    def distortion(self): # label: y , data: X , centroids

        distortion_out = []

        for k_index in self.k:

            wss_k , epoch = self.WSS(k_index) 

            distortion_out.append([ wss_k /self.X.shape[0], epoch])

        self.distortion_out = np.asarray(distortion_out)

        return self.distortion_out

        

    def inertia(self):

        inertia_out = []

        for k_index in self.k:

            wss_k , epoch = self.WSS(k_index) 

            inertia_out.append([ wss_k, epoch])

        self.inertia_out = np.asarray(inertia_out)

        return self.inertia_out

    

    def a(self ,dist_all_data_new,y, j):

        C_i = self.X[ np.where(y == j)].shape[0]

        return 1/(C_i - 1) * dist_all_data_new[j]

    

    def b(self,dist_all_data_new,y, j):

        dist_all_data_new_temp = dist_all_data_new

        dist_all_data_new_temp[j] = sys.maxsize

        second_min = np.argmin(dist_all_data_new_temp)

        C_j = self.X[ np.where(y == second_min)].shape[0]

        return 1/C_j*dist_all_data_new[second_min]

    

    def s(self, a_i, b_i):

        return (b_i - a_i)/(max(a_i , b_i))

    

    def silhouette(self):

        silhouette_out = []

        for k_index in self.k:

            model = KMeans(n_clusters = k_index, random_state = self.random_state, plot_state = self.plot_state)

            centroids , y , epoch = model.fit(self.X)

            dist_all_data = distance(self.X , self.X)  

            dist_all_data_new = np.zeros((self.X.shape[0],centroids.shape[0] ))

            for i in range(self.X.shape[0]):

                for j in range(centroids.shape[0]):

                    dist_all_data_new[i,j] = np.sum(dist_all_data[i, np.where(y == j) ])

            s_k = []

            for i in range(self.X.shape[0]):

                j = int(y[i])

                a_i = self.a(dist_all_data_new[i],y,j)

                b_i = self.b(dist_all_data_new[i],y,j)

                s_i = self.s(a_i, b_i)

                s_k.append(s_i)

            silhouette_out.append([sum(s_k) / self.X.shape[0] , epoch])

        self.silhouette_out = np.asarray(silhouette_out)

        return self.silhouette_out

        

    def plot_(self, k_clusters):

        f = plt.figure(figsize = (8,8))

        ax1 = f.add_subplot(111)

        ax1.set_xlabel('k_cluster')

        ax1.set_ylabel('value')

        ax1.plot(self.k , k_clusters[:,0], color = 'blue')

        ax2 = ax1.twinx()

        ax2.set_ylabel('epoch')

        ax2.plot(self.k , k_clusters[:,1], color = 'green')

        plt.show()
opti = k_clusters_optimizing(X,[2,3,4,5,6], plot_state = False)



# # inertia = opti.inertia()

# distortion = opti.distortion()

silhouette = opti.silhouette()



opti.plot_(silhouette)