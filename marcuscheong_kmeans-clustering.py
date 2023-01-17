import numpy as np

import pandas as pd 

import os

import cv2

import matplotlib.pyplot as plt

df = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")

df.head()
plt.scatter(df["sepal_length"], df["sepal_width"])

plt.xlabel("Sepal Length")

plt.ylabel("Sepal Width")
setosa = df[df["species"] == "Iris-setosa"]

versicolor = df[df["species"] == "Iris-versicolor"]

virginica = df[df["species"] == "Iris-virginica"]



plt.figure(figsize=(5,5))

plt.scatter(setosa["sepal_length"], setosa["sepal_width"], color = "black", label="Setosa")

plt.scatter(versicolor["sepal_length"], versicolor["sepal_width"], color = "pink", label="Versicolor")

plt.scatter(virginica["sepal_length"], virginica["sepal_width"], color = "blue", label="Virginica")

plt.xlabel("Sepal Length")

plt.ylabel("Sepal Width")

plt.title("Labeled flowers")

plt.legend()
from sklearn.cluster import KMeans

data = df[["sepal_length", "sepal_width"]].to_numpy()



kmeans =  KMeans(n_clusters=3, random_state=0).fit(data)

labels_ = kmeans.labels_



inertia = []

x = []

for i in range(1,10):

    kmeans =  KMeans(n_clusters=i, random_state=0).fit(data)

    x.append(i)

    inertia.append(kmeans.inertia_)

    plt.scatter(i, kmeans.inertia_)



plt.plot(x,inertia)

plt.title("Elbow method")
def kmean(data, k):

    centroids = []

    assigned = set()

    x_max = (data[:,0]).max()

    x_min = (data[:,0]).min()

    while(len(centroids) != k):

        r = np.random.randint(len(data))

        if(r not in assigned):

            centroids.append(data[r])





    iteration = 300

    while(iteration > 0):

        labels = []

        labels_dict = {}

        

        #Initialize dictionary

        for i in range(len(centroids)):

            labels_dict[i] = []

            

        for p in data:

            min_dist = np.inf

            centroid = 0

            for i in range(len(centroids)):

                c = centroids[i]

                dist = np.sqrt((p[0]-c[0])**2 + (p[1]-c[1])**2)

                if(dist < min_dist):

                    min_dist = dist

                    centroid = i

            labels.append(centroid)

            labels_dict[centroid].append(p)



        new_centroid = []

        for k,v in labels_dict.items():

            new_centroid.append(((np.sum(v, axis = 0))/len(v)))

        centroids = new_centroid

        iteration -= 1

    

    return labels,centroids, labels_dict



labels, centroids, labels_dict = kmean(data,3)
for k,v in labels_dict.items():

    if(k == 0):

        for p in v:

            plt.scatter(p[0],p[1],color="black")

    elif(k == 1):

        for p in v:

            plt.scatter(p[0], p[1], color = "pink")

    else:

        for p in v:

            plt.scatter(p[0], p[1], color = "blue")

    plt.title("KMeans")
for i in range(len(data)):

    if labels_[i] == 0:

        plt.scatter(df["sepal_length"][i],df["sepal_width"][i], color = "black")

    elif(labels_[i] == 1):

        plt.scatter(df["sepal_length"][i],df["sepal_width"][i], color = "pink")

    elif(labels_[i] == 2):

        plt.scatter(df["sepal_length"][i],df["sepal_width"][i], color = "blue")

    plt.title("KMeans from sklearn")
img_path = "../input/cat-and-dog/test_set/test_set/dogs/dog.4515.jpg"



# for img in os.listdir(DATADIR):

#     img_path = os.path.join(DATADIR,img)

#     print(img_path)

#     break



bgr_img = cv2.imread(img_path)



color_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

gray_img = cv2.imread(img_path,0)

lab_img = cv2.cvtColor(((bgr_img).astype("float32")/255), cv2.COLOR_BGR2LAB)



print(color_img.shape)





plt.figure(figsize=(10,10))

plt.subplot(131)

plt.imshow(color_img)

plt.subplot(132)

plt.imshow(gray_img, cmap='gray')

plt.subplot(133)

plt.imshow(lab_img)



color = ['b','g','r']



plt.figure(figsize = (10,5))



plt.subplot(121)

for i,col in enumerate(color):

    histr = cv2.calcHist([color_img],[i],None,[256],[0,256])

    plt.plot(histr,color = col)

plt.subplot(122)

plt.hist(gray_img.ravel(),256, [0,256])

plt.show()
# n: number of times GaussianBlur is applied on the image

def gaussian_smooth(img,n):

    for i in range(n):

        img = cv2.GaussianBlur(img, (5,5),0)

    return img





# data: flattened img array, k: number of centroids

def get_centroids(data,k):

    centroids = []

    assigned = set()

    while(len(centroids) != k):

            random = np.random.randint(len(data))

            if(random not in assigned):

                assigned.add(random)

                centroids.append(data[random])

    return centroids



def kmeans(data,k):

    centroids = get_centroids(data,k)

    iterations = 30

    while(iterations > 0):

        labels = []

        labels_dict = {}

        

        for c in range(len(centroids)):

            labels_dict[c] = []

        

        for p in data:

            min_dist = np.inf

            assigned_centroid = 0

            for c_i in range(len(centroids)):

                c = centroids[c_i]

                dist = 0

                for i in range(3):

                    dist += (p[i] - c[i])**2

                dist = np.sqrt(dist)

                if(dist < min_dist):

                    min_dist = dist

                    assigned_centroid = c_i

            labels.append(assigned_centroid)

            labels_dict[assigned_centroid].append(p)

        

        new_centroids = []

        for k,v in labels_dict.items():

            new_centroids.append(((np.sum(v, axis = 0))/len(v)))

        

        equal = 0

        for i,j in zip(centroids,new_centroids):

            if((i == j).all()):

                equal+=1

                

        if(equal==k):

            break

        else:

            centroids = new_centroids

            iterations -= 1

    return labels, centroids, labels_dict

        

    

    

def recover_img(labels,centroids,original_img):

    img = np.zeros((len(labels), 3))

    color = []

#     while(len(color) < 5):

#         r,g,b = np.random.randint(255),np.random.randint(255),np.random.randint(255)

#         color.append([r,g,b])

    for i in range(len(labels)):

        img[i] = (centroids[labels[i]])

    img = (img.reshape(original_img.shape)).astype(np.uint8)

    plt.imshow(img.astype(np.uint8))

    return img

        

    

    
lab_img = gaussian_smooth(lab_img,3)

color_img = gaussian_smooth(color_img,1)

data = (color_img.reshape((-1,3))).astype(float)
labels,centroids,labels_dict=kmeans(data,5)
seg_img = recover_img(labels,centroids,color_img)
#https://docs.opencv.org/master/d1/d5c/tutorial_py_kmeans_opencv.html

import cv2 as cv



img = color_img

Z = img.reshape((-1,3))

# convert to np.float32

Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

K = 5

ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image

center = np.uint8(center)

res = center[label.flatten()]

res2 = res.reshape((img.shape))

plt.imshow(res2)


