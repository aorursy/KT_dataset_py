# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
#
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

size_img = 28
threshold_color = 100 / 255

# Any results you write to the current directory are saved as output.
file = open("../input/digit-recognizer/train.csv")
data_train = pd.read_csv(file)

y_train = np.array(data_train.iloc[:, 0])
x_train = np.array(data_train.iloc[:, 1:])

file = open("../input/digit-recognizer/test.csv")
data_test = pd.read_csv(file)
x_test = np.array(data_test)

n_features_train = x_train.shape[1]
n_samples_train = x_train.shape[0]
n_features_test = x_test.shape[1]
n_samples_test = x_test.shape[0]
print(n_features_train, n_samples_train, n_features_test, n_samples_test)
print(x_train.shape, y_train.shape, x_test.shape)
def show_img(x):
    plt.figure(figsize=(8,7))
    if x.shape[0] > 100:
        print(x.shape[0])
        n_imgs = 16
        n_samples = x.shape[0]
        x = x.reshape(n_samples, size_img, size_img)
        for i in range(16):
            plt.subplot(4, 4, i+1) #devide figure into 4x4 and choose i+1 to draw
            plt.imshow(x[i])
        plt.show()
    else:
        plt.imshow(x)
        plt.show()
show_img(x_train)
show_img(x_test)
def int2float_grey(x):
    x = x / 255
    return x
# x_train[x_train<100] = 0
# x_train[x_train>=100] = 1
# # print(x_train[0])
# show_img(x_train)
# find the left egde
# Note: the problem is that I don't do the parrallel part
def find_left_edge(x):
    edge_left = []
    n_samples = x.shape[0]
    for k in range(n_samples):
        for j in range(size_img):
            for i in range(size_img):
                if (x[k, size_img*i+j] >= threshold_color):
                    edge_left.append(j)
                    break
            if (len(edge_left) > k):
                break
    return edge_left
# find the right egde
# Note: the problem is that I don't do the parrallel part
def find_right_edge(x):
    edge_right = []
    n_samples = x.shape[0]
    for k in range(n_samples):
        for j in range(size_img):
            for i in range(size_img):
                if (x[k, size_img*i+(size_img-1-j)] >= threshold_color):
                    edge_right.append(size_img-1-j)
                    break
            if (len(edge_right) > k):
                break
    return edge_right
# find the top egde
# Note: the problem is that I don't do the parrallel part
def find_top_edge(x):
    edge_top = []
    n_samples = x.shape[0]
    for k in range(n_samples):
        for i in range(size_img):
            for j in range(size_img):
                if (x[k, size_img*i+j] >= threshold_color):
                    edge_top.append(i)
                    break
            if (len(edge_top) > k):
                break
    return edge_top
# find the bottom egde
# Note: the problem is that I don't do the parrallel part
def find_bottom_edge(x):
    edge_bottom = []
    n_samples = x.shape[0]
    for k in range(n_samples):
        for i in range(size_img):
            for j in range(size_img):
                if (x[k, size_img*(size_img-1-i)+j] >= threshold_color):
                    edge_bottom.append(size_img-1-i)
                    break
            if (len(edge_bottom) > k):
                break
    return edge_bottom
#Note：when we do the stretch part by ourselves,there may be some blank cells
# when the scale factor is more than 2

from skimage import transform
def stretch_image(x):
    #get edges
    edge_left = find_left_edge(x)
    edge_right = find_right_edge(x)
    edge_top = find_top_edge(x)
    edge_bottom = find_bottom_edge(x)
    
    #cropping and resize
    n_samples = x.shape[0]
    x = x.reshape(n_samples, size_img, size_img)
    for i in range(n_samples):      
        x[i] = transform.resize(x[i][edge_top[i]:edge_bottom[i]+1, edge_left[i]:edge_right[i]+1], (size_img, size_img))
    x = x.reshape(n_samples, size_img ** 2)
    show_img(x)
from sklearn.feature_selection import VarianceThreshold
def get_threshold(x_train, x_test):
    selector = VarianceThreshold(threshold=0).fit(x_train)
    x_train = selector.transform(x_train)
    x_test = selector.transform(x_test)
    print(x_train.shape)
    return x_train, x_test
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#选择K个最好的特征，返回选择特征后的数据
# selector = SelectKBest(chi2, k=500).fit(x_train, y_train)
# x_train = selector.transform(x_train)
# x_test = selector.transform(x_test)
# print(x_train.shape)
from sklearn.decomposition import PCA
def get_pca(x_train, x_test):
    pca = PCA(n_components=0.95)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
    print(x_train.shape, x_test.shape)
    return x_train, x_test
# do the pre-process part

x_train = int2float_grey(x_train)
x_test = int2float_grey(x_test)
# stretch_image(x_train)
# stretch_image(x_test)
# x_train, x_test = get_threshold(x_train, x_test)
# x_train, x_test = get_pca(x_train, x_test)
def general_function(mod_name, model_name):
    y_pred = model_train_predict(mod_name, model_name)
    output_prediction(y_pred, model_name)
from sklearn.model_selection import cross_val_score
def model_train_predict(mod_name, model_name):
    import_mod = __import__(mod_name, fromlist = str(True))
    if hasattr(import_mod, model_name):
         f = getattr(import_mod, model_name)
    else:
        print("404")
        return []
    clf = f()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_train)
    get_acc(y_pred, y_train)
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    y_pred = clf.predict(x_test)
    return y_pred
def get_acc(y_pred, y_train):
    right_num = (y_train == y_pred).sum()
    print("acc: ", right_num/n_samples_train)
def output_prediction(y_pred, model_name):
    print(y_pred)
    data_predict = {"ImageId":range(1, n_samples_test+1), "Label":y_pred}
    data_predict = pd.DataFrame(data_predict)
    data_predict.to_csv("dr output %s.csv" %model_name, index = False)
# from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
mod_name = "sklearn.naive_bayes"
model_name = "GaussianNB"
# model_name = "MultinomialNB"
# general_function(mod_name, model_name)
# from sklearn.neighbors import KNeighborsClassifier
mod_name = "sklearn.neighbors"
model_name = "KNeighborsClassifier"
# general_function(mod_name, model_name)
# from sklearn.cluster import KMeans
mod_name = "sklearn.cluster"
model_name = "KMeans"
# general_function(mod_name, model_name)
# from sklearn.svm import SVC
mod_name = "sklearn.svm"
model_name = "SVC"
# general_function(mod_name, model_name)
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
mod_name = "sklearn.tree"
model_name = "DecisionTreeClassifier"
# general_function(mod_name, model_name)
# from sklearn.ensemble import RandomForestClassifier
mod_name = "sklearn.ensemble"
model_name = "RandomForestClassifier"
# general_function(mod_name, model_name)
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape, Conv2D, AveragePooling2D, Flatten
from keras.layers import MaxPooling2D
from keras.optimizers import adam

y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_all_pred = np.zeros((3, n_samples_test)).astype(np.int64)
print(y_all_pred.dtype)
model_name = "LaNet5"
model = Sequential()
model.add(Reshape(target_shape=(1, 28, 28), input_shape=(784,)))
model.add(Conv2D(kernel_size=(3, 3), filters=6, padding="same", data_format="channels_first", kernel_initializer="uniform", use_bias=False))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
model.add(Conv2D(kernel_size=(5, 5), filters=16, padding="same", data_format="channels_first", kernel_initializer="uniform", use_bias=False))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
model.add(Conv2D(kernel_size=(5, 5), filters=120, padding="same", data_format="channels_first", kernel_initializer="uniform", use_bias=False))
model.add(Flatten())
model.add(Dense(output_dim=120, activation='relu'))
model.add(Dense(output_dim=120, activation='relu'))
model.add(Dense(output_dim=10, activation='softmax'))

adam = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=30, batch_size=64)
y_pred = model.predict_classes(x_test)
output_prediction(y_pred, model_name)
y_all_pred[0] = y_pred
model_name = "CNN"
model = Sequential()

model.add(Reshape(target_shape=(1, 28, 28), input_shape=(784,)))
model.add(Conv2D(kernel_size=(3, 3), filters=32, padding="same", data_format="channels_first", kernel_initializer="uniform", use_bias=False))
model.add(AveragePooling2D(pool_size=(2, 2), data_format="channels_first"))
model.add(Conv2D(kernel_size=(3, 3), filters=32, padding="same", data_format="channels_first", kernel_initializer="uniform", use_bias=False))
model.add(AveragePooling2D(pool_size=(2, 2), data_format="channels_first"))
model.add(Conv2D(kernel_size=(3, 3), filters=64, padding="same", data_format="channels_first", kernel_initializer="uniform", use_bias=False))
model.add(AveragePooling2D(pool_size=(2, 2), data_format="channels_first"))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(output_dim=256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(output_dim=256, activation='relu'))
model.add(Dense(output_dim=10, activation='softmax'))

adam = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=50, batch_size=128)
y_pred = model.predict_classes(x_test)
output_prediction(y_pred, model_name)

y_all_pred[1] = y_pred
model_name = "ComplexCNN"

model = Sequential()
model.add(Reshape(target_shape=(1, 28, 28), input_shape=(784,)))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(output_dim=256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(output_dim=256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(output_dim=10, activation='softmax'))

adam = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=80, batch_size=128)
y_pred = model.predict_classes(x_test)
output_prediction(y_pred, model_name)

y_all_pred[2] = y_pred
model_name = "Ensemble"
print(y_pred.shape)
y_ensem_pred = np.zeros((n_samples_test,))
for i,line in enumerate(y_all_pred.T):
    y_ensem_pred[i] = np.argmax(np.bincount(line))
print(y_ensem_pred.shape, y_ensem_pred)
y_ensem_pred = y_ensem_pred.astype("int64")
output_prediction(y_ensem_pred, model_name)