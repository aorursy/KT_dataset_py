from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
trainingdata = np.random.rand(200,2)

X = np.array(trainingdata)
trainingdata[:5]

kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(X)
kmeans.labels_
kmeans.inertia_


testdata = np.random.rand(20,2)
testdata[:5]
kmeans.predict(np.array(testdata))
centers = kmeans.cluster_centers_

plt.scatter(trainingdata[:,0],trainingdata[:,1] ,c ='g' , s = 8)
plt.scatter(testdata[:,0],testdata[:,1] ,c ='b' , s = 25)

for m in range(len(centers)): 
    plt.scatter(centers[m,0],centers[m,1] ,c ='r' , s = 100)

plt.show
print(kmeans.n_iter_ )