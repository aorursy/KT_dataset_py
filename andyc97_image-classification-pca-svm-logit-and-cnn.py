# Load Libraries
import os
import numpy as np
import pandas as pd 
import cv2
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.applications import vgg16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import Sequential
%matplotlib inline
# Load images
def load_im():
    input_im, input_label = [], []
    resize = (224, 224)
    # Loop in folders
    for dirname, _, filenames in os.walk('/kaggle/input/pothole-detection-dataset'):
        for filename in filenames:
            photo_path = os.path.join(dirname, filename)
            photo_class = dirname.split('/')[-1]
            try:
                read_im = cv2.imread(photo_path)
                input_im.append(cv2.resize(read_im, resize))
                # potholes == 1
                if photo_class == 'potholes':
                    input_label.append(1)
                # normal == 0
                elif photo_class == 'normal':
                    input_label.append(0)
            except:
                print(photo_path)
    # return list of images and another list of correponding labels
    return input_im, input_label

input_im, input_label = load_im()
# Checking code: Print photo and class
index_set = np.random.choice(len(input_label), size = 5, replace = False)
for index in index_set:
    # show images
    plt.imshow(input_im[index])
    plt.show()
    # show label
    print(input_label[index])
# Train/Test split
def train_test_split(test_prop, input_im, input_label):
    # Random sampling of index
    test_size = int(np.floor(test_prop * len(input_label)))
    test_index = np.random.choice(len(input_label), size = test_size, replace = False)
    # Split
    train_x, test_x, train_y, test_y = np.delete(input_im, test_index, axis = 0), np.take(input_im, test_index, axis = 0), np.delete(input_label, test_index, axis = 0), np.take(input_label, test_index, axis = 0)
    # Return train and test sets for both images and labels
    return train_x, test_x, train_y, test_y, test_index

# 80/20 split for small data set
test_prop = 0.2
train_x, test_x, train_y, test_y, test_index = train_test_split(test_prop, input_im, input_label)
def append_im(input_im, input_label, im_iterator):
    input_label_n = input_label.copy()
    input_im_n = input_im.copy()
    for i in range(len(im_iterator)):
        im = im_iterator[i]
        im = im.astype('uint8')
        im_lbl = [input_label[i]]
        input_im_n = np.append(input_im_n, im, axis = 0)
        input_label_n = np.append(input_label_n, im_lbl, axis = 0)
    return input_im_n, input_label_n
# Flipping
flip_data_generator = ImageDataGenerator(horizontal_flip = True)
im_iterator = flip_data_generator.flow(train_x, batch_size = 1, shuffle = False)
input_im_n, input_label_n = append_im(train_x, train_y, im_iterator)

# Rotation - 30 deg 
#rotate_data_generartor = ImageDataGenerator(rotation_range = 30)
#im_iterator = rotate_data_generartor.flow(train_x, batch_size = 1, shuffle = False)
#input_im_n, input_label_n = append_im(input_im_n, input_label_n, im_iterator)

# Rotation - -30 deg 
#rotate_data_generartor = ImageDataGenerator(rotation_range = 330)
#im_iterator = rotate_data_generartor.flow(train_x, batch_size = 1, shuffle = False)
#input_im_n, input_label_n = append_im(input_im_n, input_label_n, im_iterator)
# Reshape
nx, ny, nz = train_x.shape[1], train_x.shape[2], train_x.shape[3]
train_x_nn, test_x_nn = input_im_n, test_x
train_x = input_im_n.reshape((input_im_n.shape[0], nx * ny * nz)) / 255
test_x = test_x.reshape((test_x.shape[0], nx * ny * nz)) / 255
train_y = input_label_n.reshape((input_label_n.shape[0], 1)) 
test_y = test_y.reshape((test_y.shape[0], 1)) 
# Dimensionality reduction - Full PCA
im_pca = PCA()
im_pca.fit(train_x)
variance_explained_list = im_pca.explained_variance_ratio_.cumsum()
print(variance_explained_list)
test_x_pca = im_pca.transform(test_x)
train_x_pca = im_pca.transform(train_x)
# Support vector machine with PCA
def svm_grid_search(C, kernel, train_x, train_y):
    accuracy_score_list = []
    
    for c in C:
        # Model training
        svmClassifier = svm.SVC(C = c, kernel = kernel)
        svmClassifier.fit(train_x, train_y.ravel())
        # Prediction on test set
        pred_y = svmClassifier.predict(train_x)
        # Accuracy
        accuracy = accuracy_score(train_y, pred_y)
        accuracy_score_list.append(accuracy)
        print('Regularization parameters: ', c, 'Accuracy', accuracy)
    
    max_accurarcy_id = accuracy_score_list.index(max(accuracy_score_list))
    return C[max_accurarcy_id] 

C, kernel = [0.1 * i for i in range(1, 30)], 'rbf'
opt_C = svm_grid_search(C, kernel, train_x_pca, train_y)
# Test set
svmClassifier = svm.SVC(C = opt_C, kernel = kernel)
svmClassifier.fit(train_x_pca, train_y.ravel())
pred_y = svmClassifier.predict(test_x_pca)
accuracy = accuracy_score(test_y, pred_y)
print(accuracy)
# Logistic Regression
def Logistic():
    logistic_model = Sequential()
    logistic_model.add(Dense(1, activation = 'sigmoid'))
    return logistic_model

# Compile Model
logistic_model = Logistic()
logistic_model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# Training Model
logistic_model.fit(train_x, train_y, batch_size = 32, epochs = 50, verbose = 1)
# Test set
print(logistic_model.metrics_names)
print(logistic_model.evaluate(test_x, test_y, verbose = 0))
# Convolutional Neural Network - Modified from AlexNet
def CNN():
    CNN_model = Sequential()
    
    CNN_model.add(Conv2D(filters = 96, input_shape = (224, 224, 3), kernel_size = (11, 11), strides = (4, 4), padding = 'valid'))
    CNN_model.add(Activation('relu'))
    CNN_model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
    
    CNN_model.add(Conv2D(filters = 256,  kernel_size = (5, 5), strides = (1, 1), padding = 'valid'))
    CNN_model.add(Activation('relu'))
    CNN_model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
    
    CNN_model.add(Flatten())
    CNN_model.add(Dense(512))
    CNN_model.add(Activation('relu'))
    
    CNN_model.add(Dense(256))
    CNN_model.add(Activation('relu'))
    
    CNN_model.add(Dense(1, activation = 'sigmoid'))
    
    return CNN_model

# Compile Model
cnn_model = CNN()
cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

# Training Model
cnn_model.fit(train_x_nn/255, train_y, batch_size = 256, epochs = 50, verbose = 1)
print(cnn_model.metrics_names)
print(cnn_model.evaluate(test_x_nn/255, test_y, verbose = 0))