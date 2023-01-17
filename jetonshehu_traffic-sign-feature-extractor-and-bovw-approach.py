# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print(os.listdir("../input/germantrafficsigndatasetwithbackdoorimages/GermanTrafficSignDatasetWithBackdoorImages/CleanDataset/train"))

from PIL import Image

import random

import matplotlib.pyplot as plot

import cv2

import scipy

from sklearn import svm





# Any results you write to the current directory are saved as output.
def GenerateDF(path, dummy_classes=['Pedestrian', 'Parking', 'SpeedLimit', 'DoNotEnter', 'GiveWay', 'Stop', 'TurnRight']):

    classes = dummy_classes

    

    class_list = []

    path_list = []

    for c in classes:

        for file in os.listdir(os.path.join(path, c)):

            total_path = os.path.join(os.path.join(path, c), file)

            path_list.append(total_path)

            class_list.append(c)

            

    class_list = pd.Series(class_list)

    class_list = pd.get_dummies(class_list, columns=dummy_classes)

    path_list = pd.Series(path_list).rename("path")



    return pd.concat([path_list, class_list], 1)
train_data = GenerateDF("../input/germantrafficsigndatasetwithbackdoorimages/GermanTrafficSignDatasetWithBackdoorImages/CleanDataset/train").sample(frac=1).reset_index().drop("index", 1)

print(train_data.shape)

train_data.head()
test_data = GenerateDF("../input/germantrafficsigndatasetwithbackdoorimages/GermanTrafficSignDatasetWithBackdoorImages/CleanDataset/test").sample(frac=1).reset_index().drop("index", 1)

print(test_data.shape)

test_data.head()
def LoadImage(path):

    

    img = cv2.imread(path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img,(35,35),interpolation=cv2.INTER_AREA)



    return img



img = LoadImage(train_data.sample(1).values[0][0])

print(img.shape)

plot.imshow(img)
def extract_features(img, vector_size=35):

    # Using Kaze

    alg = cv2.KAZE_create()

    

    desc_size = []

    kps_size = []

    

    #Getting image keypoints

    kps = alg.detect(img)

    kps = sorted(kps, key=lambda x: -x.response)[:vector_size]

    

    kps, dsc = alg.compute(img, kps)

    #Check if image has descriptors

    if dsc is None:

        return None

    else:

        dsc = dsc.flatten()

        needed_size = (vector_size * 64)

        if dsc.size < needed_size:



            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])

    

    

        return dsc
def map_descriptors(path):

    img = LoadImage(path)

    return extract_features(img)
display("Train Data")

train_data = pd.concat([train_data, train_data.path.map(map_descriptors).rename("kaze")], 1)

train_data.dropna(inplace=True)

display(train_data.head())

display("Test Data")

test_data = pd.concat([test_data, test_data.path.map(map_descriptors).rename("kaze")], 1)

test_data.dropna(inplace=True)

test_data.head()
def PrepareForTraining(df):

    arrs = []

    sols = []

    for i, row in df.iterrows():

        arrs.append(row["kaze"])

        sols.append(np.argmax(row.drop(["path", "kaze"]).values))

    return np.array(arrs), np.array(sols)
X_train, Y_train = PrepareForTraining(train_data)

print(X_train.shape)

print(Y_train.shape)



X_test, Y_test = PrepareForTraining(test_data)

print(X_test.shape)

print(Y_test.shape)
from sklearn.metrics import confusion_matrix, accuracy_score

import seaborn as sns



def Evaluate(model, name):

    model.fit(X_train, Y_train)

    

    p = model.predict(X_test)

    

    acc = accuracy_score(Y_test, p)

    M = confusion_matrix(Y_test, p)

    

    print("Accuracy: "+str(acc))

    

    

    plot.figure(figsize=(8,6))

    plot.title(name)

    sns.heatmap(M, annot=True, fmt='g')

    plot.xticks(np.arange(M.shape[0]), ['Parking', 'SpeedLimit', 'Pedestrian', 'TurnRight', 'Stop', 'GiveWay', 'DoNotEnter', 'CycleTrack'], rotation=90)

    plot.yticks(np.arange(M.shape[0]), ['Parking', 'SpeedLimit', 'Pedestrian', 'TurnRight', 'Stop', 'GiveWay', 'DoNotEnter', 'CycleTrack'], rotation=0)

    plot.xlabel("Predicted")

    plot.ylabel("True")

    plot.savefig("confusion_"+str(name)+".png")

    plot.show()
from sklearn.linear_model import LogisticRegression



lreg = LogisticRegression()

Evaluate(lreg, "LogisticRegression")
from sklearn.ensemble import RandomForestClassifier



RF = RandomForestClassifier()

Evaluate(RF, "RandomForest")
from sklearn.neighbors import KNeighborsClassifier as KNN



knn = KNN(5)

Evaluate(knn, "KNN")
from sklearn.neural_network import MLPClassifier



mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation="relu")

Evaluate(mlp, "MLP")
#this function will get Kaze descriptors from the images

from sklearn.cluster import MiniBatchKMeans

from scipy.cluster.vq import kmeans,vq



def read_and_clusterize(file_images, num_cluster):



    kaze_keypoints = []



    for path in file_images.path:

        #read image

        image = cv2.imread(path,1)

        # Convert them to grayscale

        image =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image,(35,35),interpolation=cv2.INTER_AREA)

        # SIFT extraction

        kaze = cv2.KAZE_create()

        kp, descriptors = kaze.detectAndCompute(image,None)

        #append the descriptors to a list of descriptors

        if descriptors is not None:

            kaze_keypoints.append(descriptors)



    kaze_keypoints=np.asarray(kaze_keypoints)

    kaze_keypoints=np.concatenate(kaze_keypoints, axis=0)

    #with the descriptors detected,we create clusters

    kmeans = MiniBatchKMeans(n_clusters=num_cluster, random_state=0).fit(kaze_keypoints)

    #return the learned model

    return kmeans
def build_histogram(descriptor_list, cluster_alg):

    histogram = np.zeros(len(cluster_alg.cluster_centers_))

    cluster_result =  cluster_alg.predict(descriptor_list)

    for i in cluster_result:

        histogram[i] += 1.0

    return histogram
#with the k-means model found, this code generates the feature vectors 

#by building an histogram of classified keypoints in the kmeans classifier 

def calculate_centroids_histogram(file_images, model, num_clusters):



    feature_vectors=[]

    class_vectors=[]





    for i, row in file_images.iterrows():

        #read image

        image = cv2.imread(row.path,1)

        #Convert them to grayscale

        image =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image,(35,35),interpolation=cv2.INTER_AREA)

        #Kaze extraction

        kaze = cv2.KAZE_create()

        kp, descriptors = kaze.detectAndCompute(image, None)

        if descriptors is not None:

            #classification of all descriptors in the model

            predict_kmeans=model.predict(descriptors)

            #calculates the histogram

            hist = build_histogram(descriptors, model)

            #histogram is the feature vector

            feature_vectors.append(hist)

            #define the class of the image

            class_sample = np.argmax(row.drop(["path", "kaze"]).values)

            class_vectors.append(class_sample)



    feature_vectors=np.asarray(feature_vectors)

    class_vectors=np.asarray(class_vectors)

    #return vectors and classes we want to classify

    return class_vectors, feature_vectors

#Detecting features and making clusters from the descriptors

def create_bovw(num_clusters):

    model= read_and_clusterize(train_data, num_clusters)

    [train_class,train_featvec] = calculate_centroids_histogram(train_data,model, num_clusters)

    [test_class,test_featvec] = calculate_centroids_histogram(test_data,model, num_clusters)

    

    return train_featvec,train_class, test_featvec, test_class, 
x_train_bovw, y_train_bovw, x_test_bovw, y_test_bovw = create_bovw(800)
from sklearn.metrics import confusion_matrix, accuracy_score

import seaborn as sns



def Evaluate_BOVW(model, name):

    model.fit(x_train_bovw, y_train_bovw)

    

    p = model.predict(x_test_bovw)

    

    acc = accuracy_score(y_test_bovw, p)

    M = confusion_matrix(y_test_bovw, p)

    

    print("Accuracy: "+str(acc))

    

    

    plot.figure(figsize=(8,6))

    plot.title(name)

    sns.heatmap(M, annot=True, fmt='g')

    plot.xticks(np.arange(M.shape[0]), ['Parking', 'SpeedLimit', 'Pedestrian', 'TurnRight', 'Stop', 'GiveWay', 'DoNotEnter', 'CycleTrack'], rotation=90)

    plot.yticks(np.arange(M.shape[0]), ['Parking', 'SpeedLimit', 'Pedestrian', 'TurnRight', 'Stop', 'GiveWay', 'DoNotEnter', 'CycleTrack'], rotation=0)

    plot.xlabel("Predicted")

    plot.ylabel("True")

    plot.savefig("confusion_"+str(name)+".png")

    plot.show()
from sklearn.linear_model import LogisticRegression



lreg = LogisticRegression()

Evaluate_BOVW(lreg, "LogisticRegression")
from sklearn.ensemble import RandomForestClassifier



RF = RandomForestClassifier()

Evaluate(RF, "RandomForest")
from sklearn.neighbors import KNeighborsClassifier as KNN



knn = KNN(5)

Evaluate(knn, "KNN")
from sklearn.neural_network import MLPClassifier



mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation="relu")

Evaluate(mlp, "MLP")