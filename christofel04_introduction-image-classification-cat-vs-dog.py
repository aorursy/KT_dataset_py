# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import scipy

import csv as csv

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn as sk

from numpy import array

import cv2

import os

import skimage as ski

import random

import PIL

from PIL import ImageOps

from PIL import Image 

from PIL import ImageFilter

from sklearn.svm import LinearSVC, SVC

from sklearn.calibration import CalibratedClassifierCV

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import log_loss

import re

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from sklearn import preprocessing



from subprocess import check_output
# Unzip the dataset

!unzip ../input/dogs-vs-cats-redux-kernels-edition/train.zip -d train

!unzip ../input/dogs-vs-cats-redux-kernels-edition/test.zip -d test
#get full dataset

TRAIN_DIR = '../working/train/train/'

train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for training images





TEST_DIR = '../working/test/test/'

test_images = [TEST_DIR+i for i in os.listdir(TEST_DIR)] # use this for test images
#get labels

predictedlabels=[]

labels=[]

train_dogs=[]

images=[]





#def baseline():         #flips a coin for each image

 #   predictedlabels=[]

  #  for i in os.listdir(TRAIN_DIR):

   #     test = random.randint(1,10)

    #    if test>5:

     #       predictedlabels.append(1)

      #  else:

       #     predictedlabels.append(0)

   

    

def train():            #gets correct class for each image



    for i in os.listdir(TRAIN_DIR):

        if 'dog' in i:

            train_dogs.append(i)

            labels.append(1)

        else:

            labels.append(0)

            

    return labels

    

    

    

def getResults(predictedlabels, labels):    #outputs accuracy



    total=0

    newpredict=[]

    for r in range(0,len(labels)):

   

        #if predictedlabels[r] == labels[r]:

        

        if float(predictedlabels[r])>0.5:

            newpredict.append(1)

        else:

            newpredict.append(0)



        if newpredict[r] == labels[r]:

            total+=1

         

    print("Accuracy:",total,"/",len(labels),"* 100 =","{0:.3f}".format(total/len(labels)*100),"%")

  





def nn(correctlabels):             

    

    #convert to one hot labels

    onehotlabels=correctlabels[:]

    for i in range(0,len(correctlabels)):

        if correctlabels[i] == 0:

            onehotlabels[i]=[1,0]

        else:

            onehotlabels[i]=[0,1]   

    

    testlabels= onehotlabels[24000:]  #last 1k images

    trainlabels= onehotlabels[:24000] #first 24k images

    

    predictedlabels=[]

    testimagepixels=[]

    actuallabels=[]

    

    #tensorflow variables

    

    sess=tf.compat.v1.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 16384]) #no_samples,how many pixels image has

    W = tf.Variable(tf.random_normal([16384,2],stddev=0.00001))

    b = tf.Variable(tf.zeros([2]))      #two classes

    sess.run(tf.global_variables_initializer())

    y = tf.matmul(x,W) + b

    

    #training

    y_ = tf.placeholder(tf.float32, shape=[None,2]) #no_samples,how many different classes

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

    

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    

    init = tf.global_variables_initializer()



    sess=tf.compat.v1.InteractiveSession()

    

    sess.run(init) 

    

    #train classifier

    for i in range(0,24000):    

        

        trainimg = Image.open(train_images[i]).convert('L')    #preprocess images

        size=128,128

        trainimg =trainimg.resize(size, Image.ANTIALIAS)             

        trainimg = trainimg.filter(ImageFilter.BLUR)

        trainimg = trainimg.filter(ImageFilter.FIND_EDGES)

        

        

        trainimg = list(trainimg.getdata())

        label = trainlabels[i]    #get correct label and format

        trainimg = preprocessing.normalize([trainimg])

           

        train_step.run(feed_dict={x: trainimg, y_: [label]}) #run with image, correct label

        

    #test classifier    

    for i in range(0,1000):

        

        testimg = Image.open(train_images[24000+i]).convert('L')    #preprocess images

        size=128,128

        testimg =testimg.resize(size, Image.ANTIALIAS)

        

        

        testimg = testimg.filter(ImageFilter.BLUR)

        testimg = testimg.filter(ImageFilter.FIND_EDGES)

       

        testimg = list(testimg.getdata())

        

        testimg = preprocessing.normalize([testimg])        

     

        testimg=testimg[0]

        testimagepixels.append(testimg)

        

        label = testlabels[i]     #get correct label and format

        actuallabels.append(label)

      

        classification = sess.run(tf.argmax(y, 1), feed_dict={x:[testimg]})    

        predictedlabels.append(classification)

    

    prediction=y

    results=prediction.eval(feed_dict={x: testimagepixels})

    results=preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True).fit_transform(results)

    

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    flattened = [val for sublist in predictedlabels for val in sublist]  

    

    #double check the accuracy

    print(float(sess.run(accuracy, feed_dict={x:testimagepixels, y_: actuallabels})))

   # print("Log Loss",sk.metrics.log_loss(correctlabels[24000:],predictedlabels))

    getResults(flattened,correctlabels[24000:25000])  

   

    

    

    

def kaggletest(correctlabels):

    

    

    

    myfile = open('results.csv', 'w')

    wr = csv.writer(myfile, quoting=csv.QUOTE_NONE,quotechar='',escapechar='\\')

    wr.writerow(["id","label"])

    

     #convert to one hot labels

    onehotlabels=correctlabels[:]

    for i in range(0,len(correctlabels)):

        if correctlabels[i] == 0:

            onehotlabels[i]=[1,0]

        else:

            onehotlabels[i]=[0,1]

    

    

    

    

    trainlabels= onehotlabels[:] #first 24k images

    

    predictedlabels=[]

    testimagepixels=[]

    actuallabels=[]

    

    #tensorflow variables

    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 16384]) #no_samples,how many pixels image has

    W = tf.Variable(tf.random_normal([16384,2],stddev=0.00001))

   #W = tf.Variable(tf.truncated_normal([16384,2],mean=5.0,stddev=10)) #no_pixels,two outputs

    b = tf.Variable(tf.zeros([2]))      #two classes

    sess.run(tf.global_variables_initializer())

    y = tf.matmul(x,W) + b

    

    #training

    y_ = tf.placeholder(tf.float32, shape=[None,2]) #no_samples,how many different classes

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

    #tweak this and measure results 

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    

    init = tf.global_variables_initializer()

    sess = tf.InteractiveSession()

    sess.run(init) 

    guesses=[] 

    #train

    for i in range(0,25000):    

        

        trainimg = Image.open(train_images[i]).convert('L')    #preprocess images

        

        size=128,128

        trainimg =trainimg.resize(size, Image.ANTIALIAS)

        

        trainimg = trainimg.filter(ImageFilter.BLUR)

        trainimg = trainimg.filter(ImageFilter.FIND_EDGES)

        

        trainimg = list(trainimg.getdata())

        

        #trainimg = [float(i)/sum(trainimg) for i in trainimg]

        label = trainlabels[i]    #get correct label and format

        trainimg = preprocessing.normalize([trainimg])

      # trainimg = preprocessing.binarize(trainimg)

       

        train_step.run(feed_dict={x: trainimg, y_: [label]}) #run with image, correct label

    

    

    #test

    for i in range(0,len(test_images)):

        

        testimg = Image.open(test_images[i]).convert('L')    #preprocess images

        size=128,128

        testimg =testimg.resize(size, Image.ANTIALIAS)

        

        testimg = testimg.filter(ImageFilter.BLUR)

        testimg = testimg.filter(ImageFilter.FIND_EDGES)

        

        testimg = list(testimg.getdata())

           

        testimg = preprocessing.normalize([testimg])

        

        testimg=testimg[0]

        testimagepixels.append(testimg)

        '''

        #this is tensorflows guess at which class the image belongs to

        classification = sess.run(tf.argmax(y, 1), feed_dict={x:[testimg]})    

        predictedlabels.append(classification)

        '''

        prediction=y

       

        results=prediction.eval(feed_dict={x: [testimg]})

       

        results = [results[0][1]]

        #print(results)

        

        guesses.append(results)



    results=preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True).fit_transform(guesses)

    

    test=[]

    for i in range(0,len(test_images)):    

        test.append([i+1,results[i][0]])

        wr.writerow([i+1,results[i][0]])

    print(len(test))
y=train()   #get correct labels



clf=nn(y)   #get trained svm 