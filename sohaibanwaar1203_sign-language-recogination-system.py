# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import warnings

import warnings

import random

# filter warnings

warnings.filterwarnings('ignore')

from subprocess import check_output

import numpy as np                     # numeric python lib



import matplotlib.image as mpimg       # reading images to numpy arrays

import matplotlib.pyplot as plt        # to plot any graph

import matplotlib.patches as mpatches  # to draw a circle at the mean contour



from skimage import measure            # to find shape contour

import scipy.ndimage as ndi            # to determine shape centrality

from skimage import filters

from skimage import feature



# matplotlib setup

%matplotlib inline

from pylab import rcParams





print(check_output(["ls", "../input"]).decode("utf8"))



!ls ../input/



# Any results you write to the current directory are saved as output.




# load data set

X=np.load("../input/sign-language-digits-dataset/Sign-language-digits-dataset/X.npy")

y=np.load("../input/sign-language-digits-dataset/Sign-language-digits-dataset/Y.npy")

print("The dataset loaded...")









# Image to display

display_images = 3

img_size = 64

count_displayed = 0





# Taking Random number from dataset

random_list = random.sample([i for i in range(len(X))], display_images * display_images)





# Displaying Small Sample of dataset

fig = plt.figure()

for i in range(display_images * display_images):

    plt.subplot(display_images, display_images, i+1)

    plt.imshow(X[random_list[i]].reshape(img_size, img_size))

plt.show()





def show_model_history(modelHistory, model_name):

    history=pd.DataFrame()

    history["Train Loss"]=modelHistory.history['loss']

    history["Validation Loss"]=modelHistory.history['val_loss']

    history["Train Accuracy"]=modelHistory.history['accuracy']

    history["Validation Accuracy"]=modelHistory.history['val_accuracy']

    

    fig, axarr=plt.subplots(nrows=2, ncols=1 ,figsize=(12,8))

    axarr[0].set_title("History of Loss in Train and Validation Datasets")

    history[["Train Loss", "Validation Loss"]].plot(ax=axarr[0])

    axarr[1].set_title("History of Accuracy in Train and Validation Datasets")

    history[["Train Accuracy", "Validation Accuracy"]].plot(ax=axarr[1]) 

    plt.suptitle(" Convulutional Model {} Loss and Accuracy in Train and Validation Datasets".format(model_name))

    plt.show()
def decode_OneHotEncoding(label):

    label_new=list()

    for target in label:

        label_new.append(np.argmax(target))

    label=np.array(label_new)

    

    return label

def correct_mismatches(label):

    label_map={0:9,1:0, 2:7, 3:6, 4:1, 5:8, 6:4, 7:3, 8:2, 9:5}

    label_new=list()

    for s in label:

        label_new.append(label_map[s])

    label_new=np.array(label_new)

    

    return label_new

    

def show_image_classes(image, label, n=10):

    label=decode_OneHotEncoding(label)

    label=correct_mismatches(label)

    fig, axarr=plt.subplots(nrows=n, ncols=n, figsize=(18, 18))

    axarr=axarr.flatten()

    plt_id=0

    start_index=0

    for sign in range(10):

        sign_indexes=np.where(label==sign)[0]

        for i in range(n):



            image_index=sign_indexes[i]

            axarr[plt_id].imshow(image[image_index], cmap='gray')

            axarr[plt_id].set_xticks([])

            axarr[plt_id].set_yticks([])

            axarr[plt_id].set_title("Sign :{}".format(sign))

            plt_id=plt_id+1

    plt.suptitle("{} Sample for Each Classes".format(n))

    plt.show()
show_image_classes(image=X, label=y.copy())
#importing the required libraries

import numpy as np

# from skimage.io import imread, imshow

from skimage.filters import prewitt_h,prewitt_v

import matplotlib.pyplot as plt

%matplotlib inline











#calculating horizontal edges using prewitt kernel

edges_prewitt_horizontal = prewitt_h(X[1])

#calculating vertical edges using prewitt kernel

edges_prewitt_vertical = prewitt_v(X[1])



plt.imshow(edges_prewitt_vertical, cmap='gray')

plt.show()

plt.imshow(edges_prewitt_horizontal, cmap='gray')

plt.show()









rcParams['figure.figsize'] = (6, 6)      # setting default size of plots

cy, cx = ndi.center_of_mass(X[1])

plt.imshow(X[1], cmap='Set3')  # show me the leaf

plt.scatter(cx, cy)           # show me its center

plt.show()











# Sobel Kernel

ed_sobel = filters.sobel(X[1])

plt.imshow(ed_sobel, cmap='gray');

plt.show()







#canny algorithm

can = feature.canny(X[1])

plt.imshow(can, cmap='gray');

plt.show()
from skimage.exposure import histogram

import cv2



image2 = X[1]



hist, hist_centers = histogram(image2)



#Plotting the Image and the Histogram of gray values

fig, axes = plt.subplots(1, 2, figsize=(8, 3))

axes[0].imshow(image2, cmap=plt.cm.gray)

axes[0].axis('off')

axes[1].plot(hist_centers, hist, lw=2)

axes[1].set_title('histogram of gray values')

plt.show()





# Gaussian Blur

kernel = np.ones((5,5),np.float32)/25

dst = cv2.filter2D(X[1],-1,kernel)

plt.title("Gaussian Blur")

plt.imshow(dst)

plt.show()





# Padding 

padding = 10

image = cv2.copyMakeBorder( X[1], padding, padding, padding, padding, cv2.BORDER_CONSTANT)

plt.imshow(image)

plt.show()



# Averaging Blur

kernel = np.ones((5,5),np.float32)/25

dst = cv2.filter2D(X[1],-1,kernel)

plt.title("Averaging Blur")

plt.imshow(dst)

plt.show()





# Box Blur

kernel = np.ones((5,5),np.float32)/55

dst = cv2.filter2D(X[1],-1,kernel)

plt.title("Box Blur")

plt.imshow(dst)

plt.show()
from keras.callbacks import EarlyStopping

from keras import optimizers

from sklearn.model_selection import train_test_split

def split_dataset(X, y, test_size=0.3, random_state=42):

    X_conv=X.reshape(X.shape[0], X.shape[1], X.shape[2],1)

    

    



    return train_test_split(X_conv,y, stratify=y,test_size=test_size,random_state=random_state)





optimizer=optimizers.RMSprop(lr=1e-4)# our default optimizer in evaluate_conv_model functio

def evaluate_conv_model(model, model_name, X, y, epochs=100,

                        optimizer=optimizers.RMSprop(lr=0.0001), callbacks=None):

    print("[INFO]:Convolutional Model {} created...".format(model_name))

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    

    

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    print("[INFO]:Convolutional Model {} compiled...".format(model_name))

    

    print("[INFO]:Convolutional Model {} training....".format(model_name))

    earlyStopping = EarlyStopping(monitor = 'val_loss', patience=20, verbose = 1) 

    if callbacks is None:

        callbacks = [earlyStopping]

    modelHistory=model.fit(X_train, y_train, 

             validation_data=(X_test, y_test),

             callbacks=callbacks,

             epochs=epochs,

             verbose=0)

    print("[INFO]:Convolutional Model {} trained....".format(model_name))



    test_scores=model.evaluate(X_test, y_test, verbose=0)

    train_scores=model.evaluate(X_train, y_train, verbose=0)

    print("[INFO]:Train Accuracy:{:.3f}".format(train_scores[1]))

    print("[INFO]:Validation Accuracy:{:.3f}".format(test_scores[1]))

    

    show_model_history(modelHistory=modelHistory, model_name=model_name)

    return model
from keras.models import Sequential

from keras import layers

from keras import optimizers



def build_conv_model_8():

    model = Sequential()

    model.add(layers.Convolution2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.BatchNormalization())

    model.add(layers.Dropout(0.25))



    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.BatchNormalization())

    model.add(layers.Dropout(0.25))



    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.BatchNormalization())

    model.add(layers.Dropout(0.25))



    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.BatchNormalization())

    model.add(layers.Dropout(0.25))

    

    model.add(layers.Flatten())

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(256, activation='relu'))

    model.add(layers.Dense(10, activation='softmax'))

        

    return model





model=build_conv_model_8()

trained_model_8_1=evaluate_conv_model(model=model, model_name=8, X=X, y=y)

for i in random.sample([i for i in range(len(X))],10):

    plt.imshow(X[i])

    plt.show()

    prediction = correct_mismatches(decode_OneHotEncoding(trained_model_8_1.predict(np.expand_dims(np.expand_dims(X[i],axis = 0),axis =3))))

    print(f"Prediction = {prediction}")