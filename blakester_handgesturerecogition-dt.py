import numpy as np 

import pandas as pd 

import os

print(os.listdir("../input/leapgestrecog/leapGestRecog"))

from PIL import Image

import matplotlib.image as mpimg 

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import IPython.display

path='../input/leapgestrecog/leapGestRecog'

folders=os.listdir(path)

folders=set(folders)



#import codecs

#import json





different_classes=os.listdir(path+'/'+'00')

different_classes=set(different_classes)









print("The different classes that exist in this dataset are:")

print(different_classes)
x=[]

z=[]

y=[]#converting the image to black and white

threshold=200

import cv2





for i in folders:

    print('***',i,'***')

    subject=path+'/'+i

    subdir=os.listdir(subject)

    subdir=set(subdir)

    for j in subdir:

        #print(j)

        images=os.listdir(subject+'/'+j)

        for k in images:

            results=dict()

            results['y']=j.split('_')[0]

            img = cv2.imread(subject+'/'+j+'/'+k,0)

            img=cv2.resize(img,(int(160),int(60)))

            

            ret, imgf = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            imgD=np.asarray(img,dtype=np.float64)

            z.append(imgD)

            imgf=np.asarray(imgf,dtype=np.float64)

            x.append(imgf)

            y.append(int(j.split('_')[0]))

            results['x']=imgf



      

l = []

list_names = []

for i in range(10):

    l.append(0)

for i in range(len(x)):

    if(l[y[i] - 1] == 0):

        l[y[i] - 1] = i

        if(len(np.unique(l)) == 10):

            break

for i in range(len(l)):

    %matplotlib inline

    print("Class Label: " + str(i + 1))

    plt.imshow(np.asarray(z[l[i]]), cmap  =cm.gray)

    plt.show()

    plt.imshow(np.asarray(x[l[i]]), cmap = cm.gray)     

    plt.show()
x=np.array(x)

y=np.array(y)

y = y.reshape(len(x), 1)

print(x.shape)

print(y.shape)

print(max(y),min(y))
x_data = x.reshape((len(x), 60, 160, 1))



x_data/=255

x_data=list(x_data)

for i in range(len(x_data)):

    x_data[i]=x_data[i].flatten()
len(x_data)



from sklearn.decomposition import PCA

pca = PCA(n_components=20)

x_data=np.array(x_data)

print("Before PCA",x_data.shape)
x_data=pca.fit_transform(x_data)

print(pca.explained_variance_ratio_)  

print(pca.singular_values_)  



print('___________________')

print("After PCA",x_data.shape)
from sklearn.model_selection import train_test_split

x_train,x_further,y_train,y_further = train_test_split(x_data,y,test_size = 0.2)

x_train,x_valid,y_train, y_valid = train_test_split(x_train,y_train,test_size = 0.5)

from sklearn.preprocessing import StandardScaler  

scaler = StandardScaler()  

#The idea behind StandardScaler is that it will transform your data

#such that its distribution will have a mean value 0 and standard deviation of 1.

scaler.fit(x_train)



X_train = scaler.transform(x_train)  

X_valid = scaler.transform(x_valid)

X_test = scaler.transform(x_further)  
from sklearn import tree



# clf = tree.DecisionTreeClassifier(max_depth=15)

# clf = clf.fit(X_train, y_train)



depthValues =  [5,10,15,20,25]

dt_error = []

for depthValue in depthValues:

    model = tree.DecisionTreeClassifier(max_depth=depthValue)

    model = model.fit(X_train, y_train)

    error = 1. - model.score(X_valid, y_valid)

    dt_error.append(error)

plt.plot(depthValues, dt_error)

plt.title('Tree Depth vs. Model Error')

plt.xlabel('tree depth')

plt.ylabel('error')

plt.xticks(depthValues)

plt.show()



minError = dt_error.index(min(dt_error))

bestDepth = depthValues[minError]



print("Optimal Tree Depth: {}".format(bestDepth))
leafValues =  [100,200,300,400,500]

leaf_error = []

for leafValue in leafValues:

    model = tree.DecisionTreeClassifier(max_leaf_nodes=leafValue)

    model = model.fit(X_train, y_train)

    error = 1. - model.score(X_valid, y_valid)

    leaf_error.append(error)

plt.plot(leafValues, leaf_error)

plt.title('Max Leafs vs. Model Error')

plt.xlabel('leaf depth')

plt.ylabel('error')

plt.xticks(leafValues)

plt.show()



minError = leaf_error.index(min(leaf_error))

bestLeaf = leafValues[minError]





print("Optimal Max Number of Leaf Nodes: {}".format(bestLeaf))
splitterTypes = ['best', 'random']

splitterError = []

for splitterValue in splitterTypes:

    model = tree.DecisionTreeClassifier(splitter = splitterValue)

    model = model.fit(X=X_train, y=y_train)

    error = 1. - model.score(X_valid, y_valid)

    splitterError.append(error)

    

plt.plot(splitterTypes, splitterError)

plt.title('Splitter Type vs Error')

plt.xlabel('splitter')

plt.ylabel('error')

plt.xticks(splitterTypes)

plt.show()



print(splitterError)

minError = splitterError.index(min(splitterError))

bestSplitter = splitterTypes[minError]



print("Optimal Splitter Type: {}".format(bestSplitter))
criterionTypes = ['gini', 'entropy']

citerionError = []

for criterionValue in criterionTypes:

    model = tree.DecisionTreeClassifier(criterion = criterionValue)

    model = model.fit(X=X_train, y=y_train)

    error = 1. - model.score(X_valid, y_valid)

    citerionError.append(error)

    

plt.plot(criterionTypes, citerionError)

plt.title('Criterion Type vs Error')

plt.xlabel('criterion')

plt.ylabel('error')

plt.xticks(criterionTypes)

plt.show()



print(citerionError)

minError = citerionError.index(min(citerionError))

bestCriterion = criterionTypes[minError]



print("Optimal Criterion: {}".format(bestCriterion))


clf = tree.DecisionTreeClassifier(max_depth=15, criterion = bestCriterion, max_leaf_nodes=bestLeaf)

clf = clf.fit(X_train, y_train)
y_pred_dt=clf.predict(X_test)

y_train_score_dt=clf.predict(X_train)
from sklearn.metrics import accuracy_score

print("accuracy of the model is:\nTest ", accuracy_score(y_further, y_pred_dt, normalize=True, sample_weight=None))

print('Train',accuracy_score(y_train, y_train_score_dt, normalize=True, sample_weight=None))
from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

matrix = confusion_matrix(y_further, y_pred_dt)

precision = precision_score(y_further, y_pred_dt, average = None)

accuracy = accuracy_score(y_further, y_pred_dt, normalize=True, sample_weight=None)

recall = recall_score(y_further, y_pred_dt, average = None)



print("Confusion Matrix:\n", matrix, "\n")

print("Accuracy:", accuracy, "\n")

print("Recall:", recall, "\n")

print("Precision:", precision)