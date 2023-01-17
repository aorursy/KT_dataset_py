import os

from os import listdir

from os.path import isfile, join



import pandas as pd

import numpy as np



import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import PCA



import itertools

from sklearn.metrics import confusion_matrix





from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,

                              AdaBoostClassifier)



#help function to convert png into a gray image

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

#shapes_ is a 2D array containing the image files

circles = []

squares = []

triangles = []

shapes  =["circles","squares","triangles"]

shapes_ = [circles,squares,triangles]



#1. iterating over the folders to get the images

#2. adding noisy images

images = [[],[],[]]

noisy =[]

for i,shape in enumerate(shapes):

    

    path = "../input/shapes/shapes/"+shape

    files = [f for f in os.listdir(path) if isfile(join(path, f))]



    for file in files:

        

        fpath =path+"/"+file

        img=rgb2gray(mpimg.imread(fpath))

        

        # here i do a PCA to convert the Image file into a vector

        images[i].append(img)

        pca = PCA(n_components=1)

        img = pca.fit_transform(img)

        newimg = []

        for cell in img:

            newimg.append(cell[0])        

        shapes_[i].append(newimg)

        

   
print("circle")

plt.imshow(images[0][0])

plt.show()



print("square")

plt.imshow(images[1][0])



plt.show()

print("triangle")

plt.imshow(images[2][0])



plt.show()
mean_circle = sum(images[0])

plt.matshow(mean_circle)



mean_square = sum(images[1])

plt.matshow(mean_square)



mean_triangle = sum(images[2])

plt.matshow(mean_triangle)
from skimage.transform import rotate

from PIL import Image



def random_rotate(image):   

     angle = np.random.uniform(low=-30.0, high=30.0)

     image = rotate(image, angle, 'bicubic') 

     img = Image.fromarray(image)

     left, top, right, bottom = 0,0,28,28

     cropped = np.array(img.crop( ( left, top, right, bottom ) ))     

     return cropped



#insert rotated images

import random

def rotated_images(images,shape,n=int(0.1*len(images))):

    print(n)

    rimages=[]

    

    for image in images:

        rimg = random_rotate(image)

        pca = PCA(n_components=1)

        rimg = pca.fit_transform(rimg).flatten()

          

        rimages.append(rimg)

        

    df_rotate = pd.DataFrame(rimages).sample(n,replace=True)

    df_rotate["shape"]= [shape for c in range(len(df_rotate))]

    return df_rotate

#creating a Dataframe



dfc = pd.DataFrame(shapes_[0])   

dfc["shape"]= [1 for c in range(len(dfc))]



dfs = pd.DataFrame(shapes_[1])   

dfs["shape"]= [2 for c in range(len(dfs))]



dft = pd.DataFrame(shapes_[2])   

dft["shape"]= [3 for c in range(len(dft))]

#creating rotated

df = pd.concat([dfc,dfs,dft],ignore_index=True)



X = df.drop(["shape"], axis=1)

y = df["shape"]



X_rtrain, X_rtest, y_rtrain, y_rtest = train_test_split(X, y, test_size=0.25, random_state=12,stratify=y)



n=1000



dfc_rotate= rotated_images(images[0],1,n)

dfs_rotate= rotated_images(images[1],2,n)

dft_rotate= rotated_images(images[2],3,n*2)



def bootstrap(dfs,n):

    new_dfs=[]

    for df in dfs:

        for i in range(n):

            new_dfs.append(df)

    return pd.concat(new_dfs,ignore_index=True)





dfb= bootstrap([dfc,dfs,dft],10)

df = pd.concat([dfb,dfc_rotate,dfs_rotate,dft_rotate],ignore_index=True)





#prepare for classification

X = df.drop(["shape"], axis=1)

y = df["shape"]

#look at PCA in 2D

pca = PCA(n_components=2)

pca_2d = pca.fit_transform(X)

pca_df = pd.DataFrame(data = pca_2d

             , columns = ['principal component 1', 'principal component 2'])

pca_df["shape"]=y

#plot circles

fig = plt.figure(figsize=(10,10))

ax1 = fig.add_subplot(111)

plt.title("PCA of 300 Images")

plt.xlabel("principal component 1")

plt.ylabel("principal component 2")

circles = pca_df[pca_df["shape"]==1]

ax1.scatter(circles["principal component 1"],circles["principal component 2"] ,c="green",marker = "o",label = "circles")



squares = pca_df[pca_df["shape"]==2]

ax1.scatter(squares["principal component 1"],squares["principal component 2"] ,c="blue",marker = "s",label = "squares")



triangles = pca_df[pca_df["shape"]==3]



ax1.scatter(triangles["principal component 1"],triangles["principal component 2"] ,c="red",marker = "v",label = "triangles")

ax1.plot()

plt.legend()

plt.show()
#little help function to measure accuracy

def predict_acc(X_test,y_test,clf):

    predictions=[]                      

    for i in range(len(X_test)): 

        predictions.append(float(clf.predict([X_test.iloc[i]])[0]))   

    acc = accuracy_score(y_test, predictions)

    return [acc,predictions]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12,stratify=y)

# using Random Forest CLassification

#classifier = RandomForestClassifier()



classifier = ExtraTreesClassifier()

clf = classifier

clf = clf.fit(X_train, y_train)

acc = predict_acc(X_rtest,y_rtest,clf)[0]

print("Accuracy of Model is "+str(acc))



def plot_confusion_matrix(cm, classes,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt ="d"

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



class_names = shapes

y_pred = predict_acc(X_rtest,y_rtest,clf)[1]

# Compute confusion matrix

cnf_matrix = confusion_matrix(y_rtest, y_pred)

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure(figsize=(10,10))

plot_confusion_matrix(cnf_matrix, classes=class_names,

                      title='Confusion matrix')



plt.show()
import warnings

warnings.filterwarnings("ignore")

n = 100

labels = ["RandomForest","ExtraTree","AdaBoost"]

classifiers =  [RandomForestClassifier, ExtraTreesClassifier,

                              AdaBoostClassifier]

accs_classifier = []

for j,c in enumerate(classifiers):

    

    accs = []

    for i in range(n):

        X_train, X_test, y_train, y_test = train_test_split(

            X, y, test_size=0.25, random_state=12,stratify=y

        )

        clf = ExtraTreesClassifier()

        clf = clf.fit(X_train, y_train)

        acc = predict_acc(X_rtest,y_rtest,clf)[0]

        accs.append(acc)

    accs_classifier.append(accs)

    

    

    print("Mean accuracy of "+labels[j]+" "+str(np.mean(accs)))



fig = plt.figure(figsize=(10,10))

ax1 = fig.add_subplot(111)

plt.title("Distribution of Accuracy of "+str(n)+" sample splits")

ax1.boxplot(accs_classifier,labels=labels)
n = 100

labels = ["RandomForest","ExtraTree","AdaBoost"]

classifiers =  [RandomForestClassifier, ExtraTreesClassifier,

                              AdaBoostClassifier]

accs_classifier = []

for j,c in enumerate(classifiers):

    

    accs = []

    for i in range(n):

        X_rtrain, X_rtest, y_train, y_test = train_test_split(

            X, y, test_size=0.25, random_state=12,stratify=y

        )

        clf = ExtraTreesClassifier()

        clf = clf.fit(X_train, y_train)

        acc = predict_acc(X_test,y_test,clf)[0]

        accs.append(acc)

    accs_classifier.append(accs)

    

    

    print("Mean accuracy of "+labels[j]+" "+str(np.mean(accs)))



fig = plt.figure(figsize=(10,10))

ax1 = fig.add_subplot(111)

plt.title("Distribution of Accuracy of "+str(n)+" sample splits")

ax1.boxplot(accs_classifier,labels=labels)