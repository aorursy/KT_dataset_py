import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense 
from keras.preprocessing.image import load_img

import warnings
warnings.filterwarnings('ignore')
cats = os.listdir("../input/cat-and-dog/training_set/training_set/cats")
print("cats Data =",cats[:10])
dogs=os.listdir("../input/cat-and-dog/training_set/training_set/dogs")
print("Uninfected Data = ",dogs[:10])
for i in range(5):
    img=load_img("../input/cat-and-dog/training_set/training_set/cats/"+cats[i])
    plt.imshow(img)
    plt.title("Cat")
    plt.show()
for i in range(5):
    img=load_img("../input/cat-and-dog/training_set/training_set/dogs/"+dogs[i])
    plt.imshow(img)
    plt.title("Dog")
    plt.show()
img=Image.open("../input/cat-and-dog/training_set/training_set/cats/cat.1007.jpg").convert('L')
plt.imshow(img)
new_image = img.resize((64, 64))
plt.imshow(new_image)
cats_train = os.listdir("../input/cat-and-dog/training_set/training_set/cats")
dogs_train = os.listdir("../input/cat-and-dog/training_set/training_set/dogs")

cats_test = os.listdir("../input/cat-and-dog/test_set/test_set/cats")
dogs_test = os.listdir("../input/cat-and-dog/test_set/test_set/dogs")
cat_train = []
dog_train = []
label = []

for i in cats_train: 
    if i != "_DS_Store":
        cat = Image.open("../input/cat-and-dog/training_set/training_set/cats/"+i).convert("L") # converting grey scale 
        cat = cat.resize((64,64), Image.ANTIALIAS) # resizing to 40,40
        #  Image.ANTIALIAS -> (a high-quality downsampling filter)
        cat = np.asarray(cat)/255 # bit format (RGB) (asarry -> Convert the input to an array.)
        cat_train.append(cat)
        label.append(1)
    else:
        continue
    
for i in dogs_train:
    if i != "_DS_Store":
        dog = Image.open("../input/cat-and-dog/training_set/training_set/dogs/"+i).convert("L") # converting grey scale 
        dog = dog.resize((64,64), Image.ANTIALIAS)
        dog = np.asarray(dog)/255 # bit format
        dog_train.append(dog)
        label.append(0)
    else:
        continue
        
for i in cats_test:
    if i != "_DS_Store":
        cat = Image.open("../input/cat-and-dog/test_set/test_set/cats/"+i).convert("L") # converting grey scale 
        cat = cat.resize((64,64), Image.ANTIALIAS) # resizing to 40,40
        #  Image.ANTIALIAS -> (a high-quality downsampling filter)
        cat = np.asarray(cat)/255 # bit format (RGB) (asarry -> Convert the input to an array.)
        cat_train.append(cat)
        label.append(1)
    else:
        continue
    
for i in dogs_test:
    if i != "_DS_Store":
        dog = Image.open("../input/cat-and-dog/test_set/test_set/dogs/"+i).convert("L") # converting grey scale 
        dog = dog.resize((64,64), Image.ANTIALIAS)
        dog = np.asarray(dog)/255 # bit format
        dog_train.append(dog)
        label.append(0)
    else:
        None
catt = np.array(cat_train)
print(type(catt))
print(catt.shape)

dogg = np.array(dog_train)
print(type(dogg))
print(dogg.shape)
x_train = np.concatenate((cat_train, dog_train),axis=0)
x_train_label = np.asarray(label)
x_train_label = x_train_label.reshape(x_train_label.shape[0], 1)

print("cat:",np.shape(cat_train) , "dog:",np.shape(dog_train))
print("train_dataset:",np.shape(x_train), "train_values:",np.shape(x_train_label))
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x_train, x_train_label, test_size=0.2, random_state=42)
number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]
number_of_train
X_train.shape
X_test.shape
X_train_flatten = X_train.reshape(number_of_train,X_train.shape[1]*X_train.shape[2])
X_test_flatten = X_test .reshape(number_of_test,X_test.shape[1]*X_test.shape[2])
print("X train flatten",X_train_flatten.shape)
print("X test flatten",X_test_flatten.shape)
x_train = X_train_flatten.T
x_test = X_test_flatten.T
y_train = Y_train.T
y_test = Y_test.T
print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)
# reshaping
x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T
from sklearn.linear_model import LogisticRegression
LogisticRegression(solver='lbfgs')

logistic=LogisticRegression(random_state=42,max_iter=20)
logistic.fit(x_train,y_train)

print("test accuracy: {} ".format(logistic.fit(x_train, y_train).score(x_test, y_test)))
print("train accuracy: {} ".format(logistic.fit(x_train, y_train).score(x_train, y_train)))
LogisticRegression(solver='lbfgs')

logistic=LogisticRegression(random_state=42,max_iter=100)
logistic.fit(x_train,y_train)

print("test accuracy: {} ".format(logistic.fit(x_train, y_train).score(x_test, y_test)))
print("train accuracy: {} ".format(logistic.fit(x_train, y_train).score(x_train, y_train)))
def build_classifier():
    classifier = Sequential() # initialize neural network
    classifier.add(Dense(units = 32 , kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))
    classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu')) 
    classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu')) 
    classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu')) 
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')) 
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train)
mean = accuracies.mean()
variance = accuracies.std()
print("Accuracy mean: "+ str(mean))
print("Accuracy variance: "+ str(variance))
def build_classifier():
    classifier = Sequential() # initialize neural network
    classifier.add(Dense(units = 32 , kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))
    classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu')) 
    classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu')) 
    classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu')) 
    classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu')) 
    classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu')) 
    classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')) 
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 3)
mean = accuracies.mean()
variance = accuracies.std()
print("Accuracy mean: "+ str(mean))
print("Accuracy variance: "+ str(variance))
