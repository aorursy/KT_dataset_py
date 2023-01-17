import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage,dendrogram
from sklearn.cluster import AgglomerativeClustering

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#first we will create our data (class) by using numpy (gaussian variable)

#class 1
x1=np.random.normal(25,5,1000)
y1=np.random.normal(25,5,1000)
#class 2
x2=np.random.normal(55,5,1000)
y2=np.random.normal(60,5,1000)
#class 3
x3=np.random.normal(35,5,1000)
y3=np.random.normal(15,5,1000)

x=np.concatenate((x1,x2,x3),axis=0)
y=np.concatenate((y1,y2,y3),axis=0)

dictionary={"x":x,"y":y}

#now we have our data 
data=pd.DataFrame(dictionary)
data.head()
#data visualization

plt.scatter(x1,y1)
plt.scatter(x2,y2)
plt.scatter(x3,y3)
plt.xlabel("")
plt.ylabel("")
plt.show()
#kmeans algorithm will see our data as it is shown below

plt.scatter(x1,y1,color="black")
plt.scatter(x2,y2,color="black")
plt.scatter(x3,y3,color="black")

plt.show()
#Kmeans
wcss=[]
for k in range(1,20):
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_) #for every k values, it will find kmeans
    
#now we will see our kmeans and wcss values by ploting graph
plt.plot(range(1,20),wcss)
plt.xlabel("number of k (cluster) values")
plt.ylabel("wcss values")
plt.show()

#in this plot we will find elbow point for kmeans's k value
#and we will see from the graph our k value should be 3
#kmeans model for k=3
kmeans=KMeans(n_clusters=3)
clusters=kmeans.fit_predict(data) #first it will fit our data and later apply prediction on our data
data["label"]=clusters

#we will see our data, kmeans divided 3 parts 
plt.scatter(data.x[data.label==0],data.y[data.label==0],color="red")
plt.scatter(data.x[data.label==1],data.y[data.label==1],color="green")
plt.scatter(data.x[data.label==2],data.y[data.label==2],color="blue")

#and we can see center of clusters, plt.scatter(x,y,color="")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color="orange")

plt.show()


#image reading and converting color

img=cv2.imread("../input/minion/mn.jpg")
plt.imshow(img)

#convert to gray scale
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#show the gray scale image
plt.imshow(img_gray,cmap="gray")

#visualize  images
images=[img,img_gray]
titles=["Original image","Gray image"]
for i in range(2):
        plt.subplot(2, 2, i+1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
plt.show()

#Kmeans with k=5 
kmeans=KMeans(n_clusters=5)
kmeans.fit(img_gray.reshape(img_gray.shape[0]*img_gray.shape[1],1))

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(kmeans.labels_.reshape(179,281))
plt.title('K-Means k=5')
plt.show()
#Kmeans clustering with differen k values
def main():
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #Input parameters
#1.samples : It should be of np.float32 data type, and each feature should be put in a single column.
#2.clusters(K) : Number of clusters required at end
#3.criteria : It is the iteration termination criteria. When this criteria is satisfied, 
#algorithm iteration stops. Actually, it should be a tuple of 3 parameters. They are ( type, max_iter, epsilon ):
    
    Z = img_gray.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    #output parameters
#1.compactness : It is the sum of squared distance from each point to their corresponding centers.
#2.labels : This is the label array (same as ‘code’ in previous article) where each element marked ‘0’, ‘1’.....
#3.centers : This is array of centers of clusters.
    K=2
    ret, label1, center1 = cv2.kmeans(Z, K, None,criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center1 = np.uint8(center1)
    res1 = center1[label1.flatten()]
    output1 = res1.reshape((img_gray.shape))
    
    K=4
    ret, label1, center1 = cv2.kmeans(Z, K, None,criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center1 = np.uint8(center1)
    res1 = center1[label1.flatten()]
    output2 = res1.reshape((img_gray.shape))
    
    K=12
    ret, label1, center1 = cv2.kmeans(Z, K, None,criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center1 = np.uint8(center1)
    res1 = center1[label1.flatten()]
    output3 = res1.reshape((img_gray.shape))

    output = [img, output1, output2, output3]
    titles = ['Original Image', 'K=2', 'K=4', 'K=12']
    
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(output[i])
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()
#run
if __name__ == "__main__":
    main()
#we will use imgg data which is created from minion image
#dendogram
merg=linkage(img_gray,method="ward")
dendrogram(merg,leaf_rotation=90)

#visualize
plt.xlabel("data points")
plt.ylabel("Euclidean distance")
plt.show()

#dendogram for data which is created before 
merg=linkage(data,method="ward")
dendrogram(merg,leaf_rotation=90)

#visualize
plt.xlabel("data points")
plt.ylabel("Euclidean distance")
plt.show()

#we can see from this figure, number of cluster should be 3
#Hierarcyhcal clustering with sklearn library

hierarcy_clustering=AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="ward")

cluster=hierarcy_clustering.fit_predict(data)

data["label"]=cluster

#visualization
plt.scatter(data.x[data.label==0],data.y[data.label==0],color="red")
plt.scatter(data.x[data.label==1],data.y[data.label==1],color="green")
plt.scatter(data.x[data.label==2],data.y[data.label==2],color="yellow")

plt.show()