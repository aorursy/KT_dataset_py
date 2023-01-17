import numpy as np 
import pandas as pd
import os

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
#from keras.applications.vgg19 import preprocess_input
#from keras.applications.mobilenet import preprocess_input

# time measuring
import time
print(os.listdir("../input"))
train_df = pd.read_csv("../input/train.csv")
train_df.head()
from joblib import Parallel, delayed
import multiprocessing

def preprocess_image(index, data, dataset):
    fig = data.iloc[index]['Image']
    #load images into images of size 100x100x3
    img = image.load_img("../input/"+dataset+"/"+fig, target_size=(100, 100, 3))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    
    return x
    

def prepareImages_parallel(data, m, dataset):
    print("Preparing images")
    X_train = np.zeros((m, 100, 100, 3))  
    X_train = Parallel(n_jobs=-1, prefer="threads") (delayed(preprocess_image) 
                                        (i, data, dataset) for i in range(len(data)))
    
    return X_train
start = time.time()
X = prepareImages_parallel(train_df, train_df.shape[0], "train")
X = np.array(X, dtype='float64')
X /= 255
print('multithreaded:', time.time() - start)
def prepareImages(data, m, dataset):
    print("Preparing images")
    X_train = np.zeros((m, 100, 100, 3))
    count = 0
    
    for fig in data['Image']:
        #load images into images of size 100x100x3
        img = image.load_img("../input/"+dataset+"/"+fig, target_size=(100, 100, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)

        X_train[count] = x
        if (count%500 == 0):
            print("Processing image: ", count+1, ", ", fig)
        count += 1
    
    return X_train
start = time.time()
X = prepareImages(train_df, train_df.shape[0], "train")
X /= 255
print('single-threaded:', time.time() - start)