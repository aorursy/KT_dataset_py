import numpy as np # linear algebra
import matplotlib.pyplot as plt
import cv2 # opencv for read images
import random # mix data
import os
from tqdm import tqdm
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
def read_data(directory,data_list):
    for category in CATEGORIES:
        path = os.path.join(directory,category)
        class_num = CATEGORIES.index(category)
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                data_list.append([new_array,class_num])
            except Exception as e:
                pass
def edit_data(data_list):
    X = []
    Y = []
    for feature,label in data_list:
        X.append(feature)
        Y.append(label)
    X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    Y = np.array(Y)
    return X,Y

DATADIR_TRAIN = "/kaggle/input/chest-xray-pneumonia/chest_xray/train"
DATADIR_TEST = "/kaggle/input/chest-xray-pneumonia/chest_xray/test"
DATADIR_VAL = "/kaggle/input/chest-xray-pneumonia/chest_xray/val"

CATEGORIES = ["NORMAL","PNEUMONIA"]
IMG_SIZE = 50
training_data = []
testing_data = []
val_data = []

read_data(DATADIR_TRAIN,training_data)
read_data(DATADIR_TEST,testing_data)
read_data(DATADIR_VAL,val_data)

random.shuffle(training_data)
random.shuffle(testing_data)
random.shuffle(val_data)

X_train, Y_train = edit_data(training_data)
X_test, Y_test = edit_data(testing_data)
X_val, Y_val = edit_data(val_data)
number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]
number_of_val = X_val.shape[0]

x_train_flatten = X_train.reshape(number_of_train, X_train.shape[1]*X_train.shape[2])
x_test_flatten = X_test.reshape(number_of_test,X_test.shape[1]*X_test.shape[2])
x_val_flatten = X_val.reshape(number_of_val,X_val.shape[1]*X_val.shape[2])
classifier = Sequential()
classifier.add(Dense(units=5000,kernel_initializer="uniform",activation="relu",input_dim=x_train_flatten.shape[1]))
classifier.add(Dense(units=2000,kernel_initializer="uniform",activation="relu"))
classifier.add(Dense(units=1000,kernel_initializer="uniform",activation="relu"))
classifier.add(Dense(units=250,kernel_initializer="uniform",activation="relu"))
classifier.add(Dense(units=50,kernel_initializer="uniform",activation="relu"))
classifier.add(Dense(units=10,kernel_initializer="uniform",activation="relu"))
classifier.add(Dense(units=1,kernel_initializer="uniform",activation="sigmoid"))
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

history = classifier.fit(x_train_flatten,Y_train,batch_size=8,epochs=20)
train_accuracy = history.history["accuracy"]
train_accuracy = np.array(train_accuracy)
train_accuracy = train_accuracy.mean()
print("Train Accuracy: ",train_accuracy)

score_test = classifier.evaluate(x_test_flatten,Y_test)
score_val = classifier.evaluate(x_val_flatten, Y_val)
print("Test Accuracy: ",score_test[1])
print("Validation Accuracy: ",score_val[1])