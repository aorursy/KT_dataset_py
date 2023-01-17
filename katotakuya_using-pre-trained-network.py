import numpy as np
import cv2
import os
import glob
import h5py
from tqdm import tqdm
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

def VGG_16(weights_path="../data/vgg16_weights.h5"):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1000, activation='softmax'))
    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    return model

def get_im_cv2(path, img_rows, img_cols, color_type=1):
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)
    # Reduce size
    resized = cv2.resize(img, (img_cols, img_rows))
    return resized

def load_train(img_rows, img_cols, j, color_type=1):
    X_train = []
    y_train = []
    print('Read c%d train images'%j)
    path = os.path.join('..', 'data', 'train', 'c' + str(j), '*.jpg')
    files = glob.glob(path)
    for fl in tqdm(files):
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl, img_rows, img_cols, color_type)
        X_train.append(img)
        y_train.append(j)
    return np.array(X_train, dtype=np.float32), np.array(y_train)


def load_test(img_rows, img_cols, read_range=[0, 1000], color_type=1):
    print('Read test images')
    path = os.path.join('..', 'data', 'test', '*.jpg')
    files = glob.glob(path)
    # Sanity check
    assert(read_range[0] < len(files))
    assert(read_range[0] < read_range[1])
    if read_range[1] > len(files):
        read_range[1] = len(files)
    files = files[read_range[0]:read_range[1]]
    X_test = []
    X_test_id = []
    total = 0
    for fl in tqdm(files):
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl, img_rows, img_cols, color_type)
        X_test.append(img)
        X_test_id.append(flbase)
    return np.array(X_test, dtype=np.float32), np.array(X_test_id)

if __name__=="__main__":
    # Load model
    model = VGG_16("../data/vgg16_weights.h5")
    model.compile(optimizer="sgd", loss="categorical_crossentropy")
    
    # Calculate train features and save it
    features_train = []
    labels_train = []
    for j in range(10):
        # Load train data
        X_train, y_train = load_train(224, 224, j, color_type=3)
        # Modify images
        print("Preprocessing images")
        X_train[:, :, :, 0] -= 103.939
        X_train[:, :, :, 1] -= 116.779
        X_train[:, :, :, 2] -= 123.68
        X_train = X_train.transpose((0, 3, 1, 2))
        # Calculate features
        print("Calculate features")
        out = model.predict(X_train, verbose=1)
        features_train.append(out)
        labels_train.append(y_train)
        
    # Save features
    features_train = np.concatenate(features_train, axis=0)
    labels_train = np.concatenate(labels_train)
    np.save("../data/features_train2", features_train)
    np.save("../data/labels_train2", labels_train)
    
    # Calculate test features and save it
    features_test = []
    ids = []
    for i in range(10):
        print("Converting %d th test set"%i)
        # Load train data
        X_test, X_test_id = load_test(224, 224, read_range=[8000*i, 8000*(i+1)], color_type=3)
        # Modify images
        print("Preprocessing images")
        X_test[:, :, :, 0] -= 103.939
        X_test[:, :, :, 1] -= 116.779
        X_test[:, :, :, 2] -= 123.68
        X_test = X_test.transpose((0, 3, 1, 2))
        # Calculate features
        print("Calculate features")
        out = model.predict(X_test, verbose=1)
        features_test.append(out)
        ids.append(X_test_id)
        
    # Save features
    features_test = np.concatenate(features_test, axis=0)
    ids = np.concatenate(ids)
    np.save("../data/features_test2", features_test)
    np.save("../data/ids_test2", ids)
from __future__ import division
import time
import numpy as np
import pandas as pd
from random import shuffle
from sklearn.metrics import log_loss
from sklearn.svm import SVC

# Record time
start = time.time()

# Load dataset
print("Load dataset")
train = np.load("../data/features_train2.npy")
labels = np.load("../data/labels_train2.npy")
test = np.load("../data/features_test2.npy")
ids = np.load("../data/ids_test2.npy")
load_end = time.time()
print("Loading took %f seconds"%(load_end - start))

# Train model
print("Training model")
perm = np.random.permutation(len(train))
train = train[perm]
labels = labels[perm]
model = SVC(probability=True)
model.fit(train, labels)
train_end = time.time()
print("Training took %f seconds"%(train_end - load_end))

# Predict
print("Predict labels")
probs = model.predict_proba(test)
pred_end = time.time()
print("Prediction took %f seconds"%(pred_end - train_end))

# Save submission
print("Save submission")
submission = pd.DataFrame({"img":ids})
names = ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"]
for i in range(len(names)):
    submission[names[i]] = probs[:, i]

submission.to_csv("../submission/submission17.csv", index=False)