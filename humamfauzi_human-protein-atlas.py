import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Image resizing
from scipy.misc import imread, imresize
%matplotlib inline

# For more readable dictionary print
import pprint

# Ignore Warning
import warnings 
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))
%%time
protein_part = ["Nucleoplasm", "Nuclear membrane", "Nucleoli", "Nucleoli fibrillar center",
                "Nuclear speckles", "Nuclear bodies", "Endoplasmic reticulum", "Golgi apparatus",
                "Peroxisomes", "Endosomes", "Lysosomes", "Intermediate filaments", "Actin filaments",
                "Focal adhesion sites", "Microtubules", "Microtubule ends", "Cytokinetic bridge", 
                "Mitotic spindle", "Microtubule organizing center", "Centrosome", "Lipid droplets",
                "Plasma membrane", "Cell junctions", "Mitochondria", "Aggresome", "Cytosol", "Cytoplasmic bodies",
                "Rods & rings"]

train = pd.read_csv("../input/train.csv")
train["list"] = train["Target"].apply(lambda x: x.split(" "))
print("Train Shape: ", train.shape)

for i in protein_part:
    train[i] = 0

for i in train.index:
    for k in train.loc[i, "list"]:
        train.loc[i, protein_part[int(k)]] = 1

sample = pd.read_csv("../input/sample_submission.csv")
print(sample.head())

print("Total Train Image: ",len(os.listdir("../input/train")))
print("Total Test Image: ",len(os.listdir("../input/test")))

print("List of Protein Type:")
for num, i in enumerate(protein_part):
    print(num, i)
arr = []
for i in protein_part:
    arr.append({"Protein": i, "Occurences": train[i].sum(), "Proportion": train[i].sum()/len(train) })
arr = pd.DataFrame(arr)
arr.set_index("Protein", inplace=True)
arr["Reciproc"] = 1 / arr["Proportion"]

class_weight = {i: arr.iloc[i,2] for i in range(len(protein_part))}
del arr
from sklearn.preprocessing import MinMaxScaler
IMAGE_DIMENSION = 128
def transform2arrayTest(image_id, dimension=IMAGE_DIMENSION):
    mms = MinMaxScaler()
    final = np.zeros([dimension,dimension,4])
    image_list = ["_red.png", "_blue.png", "_green.png", "_yellow.png"]
    for num, i in enumerate(image_list):
        img = imread("../input/test/" + image_id + i)
        img = imresize(img, (dimension, dimension))
        mms.fit(img)
        img = mms.transform(img)
        final[:,:,num] = img
    return final

def transform2array(image_id, dimension=IMAGE_DIMENSION):
    mms = MinMaxScaler()
    final = np.zeros([dimension,dimension,4])
    image_list = ["_red.png", "_blue.png", "_green.png", "_yellow.png"]
    for num, i in enumerate(image_list):
        img = imread("../input/train/" + image_id + i)
        img = imresize(img, (dimension, dimension))
        mms.fit(img)
        img = mms.transform(img)
        final[:,:,num] = img
    return final
%%time

from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.metrics import categorical_accuracy
from keras import backend as K

# From https://www.kaggle.com/guglielmocamporese/macro-f1-score-keras#
import tensorflow as tf

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)
def baseline_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 4)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(46))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(28))
    model.add(Activation('sigmoid'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=[categorical_accuracy, f1])
    return model
from sklearn.model_selection import train_test_split
import time

def prepare_data(start, end):
    X = np.zeros([end-start, 128, 128, 4])
    for num, i in enumerate(train.Id.loc[start:end-1]):
        X[num, :, :, :] = transform2array(i)
    Y = train[list(train.columns[3:])].loc[start:end-1]
    return train_test_split(X, Y, test_size=0.2)
    

def massive_print(num, num0):
    print("TRAINING PART {0:d} OF {1:d} || ITERATION NUMBER {2:d} OF {3:d}".format(num[0],num[1], num0[0], num0[1]))
%%time

iteration = 18
division = 10
partial_limit = np.array(np.percentile(train.index, np.linspace(0,100,division+1)), dtype="int64")

model = baseline_model()

accuracy_arr, error_arr, f1_arr = [], [], []

for j in range(iteration):
    for i in range(division):
        massive_print((i, division),(j, iteration-1))
        start_time = time.time()
        X_tr, X_te, Y_tr, Y_te =  prepare_data(partial_limit[i], partial_limit[i+1])
        print("Composing data require {0:2.2f} seconds".format(time.time() - start_time))

        start_time = time.time()
        model.fit(X_tr, Y_tr, epochs=20, batch_size=200, class_weight=class_weight,verbose=0)
        scores = model.evaluate(X_te, Y_te, verbose=0)
        accuracy_arr.append(scores[1])
        error_arr.append(scores[0])
        f1_arr.append(scores[2])
        print("Baseline Score: %.2f%%" % (scores[1]*100))
        print("Elapsed training time: ", time.time() - start_time)
        print(" ")
del X_tr
del X_te
del Y_tr
del Y_te

plt.figure(figsize=(13,4))
plt.plot(accuracy_arr)
plt.ylabel("Accuracy")
plt.xlabel("Iteration")
plt.grid()
plt.title("Accuracy Plot over Iteration")

plt.figure(figsize=(13,4))
plt.plot(error_arr)
plt.ylabel("Loss")
plt.xlabel("Iteration")
plt.grid()
plt.title("Loss Plot over Iteration")

plt.figure(figsize=(13,4))
plt.plot(f1_arr)
plt.ylabel("F1 Macro")
plt.xlabel("Iteration")
plt.grid()
plt.title("F1 Plot over Iteration")
from sklearn.metrics import roc_curve

len(os.listdir("../input/test/"))
test = pd.read_csv("../input/sample_submission.csv")

def batch_preparation(start, end):
    final = np.zeros([end-start, 128, 128, 4])
    for num, i in enumerate(test.Id.loc[start:end-1]):
        final[num, :, :, :] = transform2arrayTest(i)
    return final

def rigid_prediction(model, test_array, prediction_threshold):
    prediction = model.predict(test_array)
    final = np.zeros(prediction.shape)
    for i in range(prediction.shape[0]):
        for j in range(prediction.shape[1]):
            if prediction[i, j] >= prediction_threshold[j]:
                final[i, j] = 1
            else:
                final[i, j] = 0
    return final

def prediction2string(array):
    super_container = []
    
    for i in range(array.shape[0]):
        prediction = np.array(array[i, :], dtype="int64")
        container = []
        
        for num, i in enumerate(prediction):
            if i == 1:
                container.append(num)
                
        string_container = str(container)[1:-1].replace(",", "")
        super_container.append(string_container)
    return super_container

def train_predict(train, model, division=10):
    
    partial_limit = np.array(np.percentile(train.index, np.linspace(0,100,division + 1)), dtype="int64")
    total_prediction = np.zeros(train[train.columns[3:]].shape)
    
    for k in range(division):
        start, end = partial_limit[k], partial_limit[k+1]
        X = np.zeros([end-start, 128, 128, 4])
        
        for num, i in enumerate(train.Id.loc[start:end-1]):
            X[num, :, :, :] = transform2array(i)
            
        total_prediction[start:end, :] = model.predict(X)
    return total_prediction

def find_threshold(train, train_prediction):
    cols = train.columns[3:]
    prediction_threshold = np.zeros(len(cols))
    report = {}
    
    for num, i in enumerate(cols):    
        y_true = train[i]
        y_pred = train_prediction[:,num]
        staging_value = 0
        
        for k in np.linspace(0,1,300):
            rigid = np.array([1 if p > k else 0 for p in y_pred])
            tpr, fpr, _ = roc_curve(y_true, rigid)
            
            if tpr[1]/fpr[1] > staging_value:
                staging_value = tpr[1]/fpr[1]
                prediction_threshold[num] = k
                
        report[cols[num]] = staging_value
    pprint.pprint(report)
    return prediction_threshold

def toRigidPrediction(y_pred, threshold):
    return np.array([1 if k > threshold else 0 for k in y_pred])

# Testing Unit for find_threshold function and random prediction baseline
a = (train.shape[0], train.shape[1]-3)
dummy_train_prediction = np.random.random(a)
find_threshold(train, dummy_train_prediction)
%%time
train_prediction = train_predict(train, model)
threshold_prediction = find_threshold(train, train_prediction)
print(threshold_prediction)
%%time
limit = np.array(np.percentile(test.index, range(0, 100, division)), dtype="int64")
limit = np.concatenate([limit, [len(test.index)]])
test_counter = np.zeros((len(test), len(protein_part)))

for i in range(division):
    print("Start:", limit[i],"End:",  limit[i+1])
    test_batch = batch_preparation(limit[i], limit[i+1])
    
    rigid = rigid_prediction(model, test_batch, threshold_prediction)
    test_counter[limit[i]:limit[i+1],:] = rigid
    
    container = prediction2string(rigid)
    test.Predicted.loc[limit[i]:limit[i+1]-1] = container

arr = []
for i in range(len(protein_part)):
    arr.append({"Protein": protein_part[i], "Occurences": test_counter[:,i].sum(), "Proportion": test_counter[:,i].sum()/len(test) })
arr = pd.DataFrame(arr)
arr.set_index("Protein", inplace=True)
arr["Reciproc"] = 1 / arr["Proportion"]
print(arr)

print(test.head())
test.to_csv("submission.csv", index=False)