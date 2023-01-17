# importing libraries used in this notebook

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.cluster import KMeans
import os, shutil, glob, os.path
from PIL import Image
image.LOAD_TRUNCATED_IMAGES = True
from tqdm.notebook import tqdm 
import glob
import math
import pickle
%pylab inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
# storing the path of all the jpj file in filelist from our dataset
filelist = glob.glob(os.path.join('/kaggle/input/change-cl/dataset','*.jpg'))
#loading the featurelist that consist of bottleneck features of all the images
with open('/kaggle/input/change-cl/featurelist', 'rb') as f: 
    featurelist = pickle.load(f) 
# loading the weights of vgg16 model that is train on imagenet data that consists of 1000 classes
model = VGG16(weights='imagenet', include_top=False)
# converting the features list to array format for k means
featurearray = np.asarray(featurelist)
# printing shape of features array
featurearray.shape
# hyper parameter tuning for finding the best number of clusters

from sklearn.cluster import KMeans

clusters = [2,5,10,20,30,40,50]
square_distance = []

for i in tqdm(clusters):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(featurearray)
    square_distance.extend([kmeans.inertia_])
# plot to find the value of k for which the loss is minimum

plt.plot(clusters, square_distance)

plt.scatter(clusters, square_distance, label='clustering points')

plt.legend()
plt.xlabel("N: hyperparameter")
plt.ylabel("Loss")
plt.title("Loss PLOT")
plt.grid()
plt.show()
# so here loss may have been reduced further but due to computatinal complexity n = 50 looks good
k_best = 50
# using the best_N
kmeans = KMeans(n_clusters=k_best, random_state=0).fit(featurearray)
labels = kmeans.labels_
# length of labels list
len(labels)
# printing the labels array
labels
# counting number of images per label
from collections import Counter
co = Counter(labels)
co
# function that return the weight of a vector

def square_rooted(x):
    return math.sqrt(sum([a*a for a in x]))


# function to return the cosine similarity between two vectors

def cosine_similarity(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return numerator/float(denominator)
# function to find cosine similarity of a given image with all the images in the dataset.
#will only find cosine similarity of those images who belongs from the same cluster from where the input image belongs.

def find_similar(input_image_path,input_features,filelist,input_label,labels):
    for i in tqdm(range(len(filelist))):
        if filelist[i]== input_image_path:
            continue
        if input_label!= labels[i]:
            continue
        dic_store[filelist[i]] = cosine_similarity(input_features,featurelist[i])
# function to return the bottleneck feature of a given input image

def input_features(input_image_path):
    img = image.load_img(input_image_path)
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = np.array(model.predict(img_data))  
    return features.flatten()
# function to represent the images as described in the problem statement

def print_images(N,input_image_path,sorted_x):
    columns = 3
    rows = ceil(N/3) +1
    fig=plt.figure(figsize=(8, 8))
    input_image = mpimg.imread(input_image_path)
    fig.add_subplot(rows, columns, 1)
    plt.imshow(input_image)

    i=3
    for key in sorted_x:
        i+=1
        if(i>N+3):
            break
        fig.add_subplot(rows, columns, i)
        img=mpimg.imread(key[0])
        imgplot = plt.imshow(img)
    plt.show()
# finding the cosine similarity between our image 3 and image 5

cosine_similarity(featurelist[2],featurelist[5]) # for demonstration
# dictionary to store the cosine similarity of input image with other images
dic_store = {}
# image for which you have to find similar images 

input_image_path = '/kaggle/input/change-cl/dataset/1000.jpg' # write image path
# showing the input image
im = Image.open(input_image_path)
im
# will return input features of the input image
features = input_features(input_image_path)
#converting the features of input image in an array to fed it to the kmeans for prediction
inputarray = np.asarray(features)
# predicting the label of the input image
inputlabel = kmeans.predict(inputarray.reshape(1,-1))
# printing the label of input image
inputlabel
# will find the cosine similarity of input image with other images

find_similar(input_image_path,features,filelist,inputlabel[0],labels)
# sorting dictionary in descending order on values
sorted_x = sorted(dic_store.items(), key=lambda kv: kv[1],reverse = True)
# printing top ten cosine similaity values and the images path
sorted_x[0:10]
# define the number of simailar images you want
N = 10 #you can take any number
# will return the N images similar to input image
print_images(N,input_image_path,sorted_x)