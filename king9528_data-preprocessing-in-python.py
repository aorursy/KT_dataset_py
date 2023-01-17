import numpy as np
import scipy.io as sio 
import os
import cv2
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import matplotlib.pyplot as plt
def get_labels():
    annos = sio.loadmat('../input/cars_annos.mat')
    _, total_size = annos["annotations"].shape
    print("total sample size is ", total_size)
    labels = np.zeros((total_size, 5))
    for i in range(total_size):
        path = annos["annotations"][:,i][0][0][0].split(".")
        id = int(path[0][8:]) - 1
        for j in range(5):
            labels[id, j] = int(annos["annotations"][:,i][0][j + 1][0])
    return labels
labels = get_labels()
def peek_image(path, idx, labels):
    image_names = os.listdir("../input/" + path + "/" + path)
    im = cv2.imread("../input/" + path + "/" + path + "/" + image_names[idx])[:,:,::-1]
    print("image is", image_names[idx])
    name = image_names[idx].split('.')
    w, h, ch = im.shape
    print("orignal shape:" , w, h)
    h_resize = int(128*1.5)
    w_resize = 128
    im = cv2.resize(im,(h_resize,w_resize),interpolation=cv2.INTER_LINEAR)
    w, h, ch = im.shape
    print("resized shape:" , w, h)
    print("the label is " + str(labels[int(name[0]) - 1, 4]))
    plt.imshow(im)

peek_image("cars_train", 4445, labels)   
labels = []
def read_data(path, labels):
    x = []
    y = []
    counter = 0
    
    for file in os.listdir("../input/" + path + "/" + path):
        im = cv2.imread("../input/" + path + "/" + path + "/" + file)[:,:,::-1]
        name = file.split('.')
        w, h, ch = im.shape
        h_resize = int(128*1.5)
        w_resize = 128
        im = cv2.resize(im,(h_resize,w_resize),interpolation = cv2.INTER_LINEAR)
        x.append(im)
        y.append(labels[int(name[0]) - 1,4])
        if counter % 1000 == 0 and counter > 0:
            print("1000 images are loaded.")
        counter += 1
        #print(file, int(name[0]) - 1)
    return np.array(x), np.array(y).reshape([len(y),1])
        
def load_split_data():
    annos = sio.loadmat('../input/cars_annos.mat')
    _, total_size = annos["annotations"].shape
    print("total sample size is ", total_size)
    labels = np.zeros((total_size, 5))
    for i in range(total_size):
        path = annos["annotations"][:,i][0][0][0].split(".")
        id = int(path[0][8:]) - 1
        for j in range(5):
            labels[id, j] = annos["annotations"][:,i][0][j + 1][0]
    print("Annotation Loading completed.")
    x_train, y_train = read_data("cars_train", labels)
    print("Loading training data completed.")
    print("training dimension is",x_train.shape)
    x_val_test, y_val_test = read_data("cars_test", labels)
    val_test_size = x_val_test.shape[0]
    print("test and val dimension is",x_val_test.shape)
    print("Loading validation and test data completed.")
    
    #shuffle and splite vallidation data and test data
    p = np.random.permutation(val_test_size)
    x_val_test = x_val_test[p]
    y_val_test = y_val_test[p]
    x_val = x_val_test[0:int(val_test_size / 2),:,:,:]
    y_val = y_val_test[0:int(val_test_size / 2),:]
    print("validation size is",int(val_test_size / 2))
    x_test = x_val_test[int(val_test_size / 2):val_test_size,:,:,:]
    y_test = y_val_test[int(val_test_size / 2):val_test_size,:]
    print("test size is",val_test_size - int(val_test_size / 2))
    print("Spliting validation and test data completed.")
    return [x_train, x_val, x_test], [y_train, y_val, y_test]
X, Y = load_split_data()