import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import cohen_kappa_score

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler

from keras.models import load_model

import matplotlib.pyplot as plt

import seaborn as sns

import cv2 #!pip3 install --user opencv-python

%matplotlib inline

from PIL import Image

from os import listdir

import os

import glob

from os.path import isfile, join
IMG_SIZE = 64 #length side of images after pre-processing (images are reshaped to squares)

img_size = IMG_SIZE * IMG_SIZE #size of the vector representing an image

list_features_names = ["pixel" + str(i) for i in range(0, img_size)] # list of features

DEBUG = False #To pre-process only a limited amount of images

pre_process_image = False #To pre-process train and test sets

train_model = False #To train the model(s). If false, load model(s) from disk
path_to_data  = '/kaggle/input/aptos2019-blindness-detection'

path_to_preprocessed_data_model = '/kaggle/input/aptos-2019' #if using pre-processed data and/or model

path_to_train_img = path_to_data + '/train_images'

path_to_test_img = path_to_data + '/test_images'

path_to_train_img_pre = path_to_data + '/train_images_preprocessed'
list_files = listdir(path_to_train_img)

#X_train.columns

list_files = [s for s in list_files if '.png' in s]
list_files[0:5]
def rgb2gray(rgb):

    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])



def crop_image_from_gray(img,tol=7):

    """

    Crop out black borders

    https://www.kaggle.com/ratthachat/aptos-updated-preprocessing-ben-s-cropping

    """  

    

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mask = gray_img>tol        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0):

            return img

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

            img = np.stack([img1,img2,img3],axis=-1)

        return img





def circle_crop(img):   

    """

    Create circular crop around image centre    

    """    

    

    img = cv2.imread(img)

    img = crop_image_from_gray(img)    

    

    height, width, depth = img.shape    

    

    x = int(width/2)

    y = int(height/2)

    r = np.amin((x,y))

    

    circle_img = np.zeros((height, width), np.uint8)

    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)

    img = cv2.bitwise_and(img, img, mask=circle_img)

    img = crop_image_from_gray(img)

    

    return img 



def circle_crop_v2(img, IMG_SIZE = 512):

    """

    Create circular crop around image centre

    """

    img = cv2.imread(img)

    img = crop_image_from_gray(img)



    #height, width, depth = img.shape

    #largest_side = np.max((height, width))

    #img = cv2.resize(img, (largest_side, largest_side))

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))



    height, width, depth = img.shape



    x = int(width / 2)

    y = int(height / 2)

    r = np.amin((x, y))



    circle_img = np.zeros((height, width), np.uint8)

    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)

    img = cv2.bitwise_and(img, img, mask=circle_img)

    img = crop_image_from_gray(img)

    # Make sure the image has the right size

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))



    return img
## try circle crop

fig = plt.figure(figsize=(25, 16))

for (index_img, img_name)  in enumerate(list_files):

    # DEBUG

    if index_img > 10:

        break

    ax = fig.add_subplot(5,5, index_img+1)

    path= path_to_train_img + '/' + img_name

    image = circle_crop_v2(path, IMG_SIZE = IMG_SIZE)

    image = rgb2gray(image)

    plt.imshow(image, cmap='gray')
image.shape
## Circle cropping

if pre_process_image == True:

    print("Pre-processing train images")

    df_train = pd.read_csv(path_to_data + '/train.csv')

    df_train.head()

    image_2D = None #To store images in a vector form

    # Read files following the order of df_train

    for (index_img, row) in df_train.iterrows():

    #for (index_img, img_name) in enumerate(list_files):

        img_name = row['id_code'] + '.png'

        # DEBUG

        if DEBUG==True:

            if index_img > 10:

                break

        path= path_to_train_img + '/' + img_name

        image = circle_crop_v2(path, IMG_SIZE = IMG_SIZE)

        image = rgb2gray(image)

        img_1D_vector = image.reshape(img_size, 1)

        if index_img == 0:

            image_2D = img_1D_vector

        else:

            image_2D = np.concatenate((image_2D, img_1D_vector), axis=1)

    print("Done")

    print("Merging df")

    data_train_img = pd.DataFrame(data=np.transpose(image_2D), # values

                  index=list(range(0, image_2D.shape[1])), # 1st column as index

                  columns=list_features_names) 



    df_merged = df_train.join(data_train_img, how='outer')

    print(df_merged.head())

    df_merged.to_csv('train_preprocessed.csv', index=False)

    print("Done")
# Freeing memory

df_merged = None

data_train_img = None

df_train = None
if pre_process_image == True:

    print("Pre-processing test images")

    df_test = pd.read_csv(path_to_data + '/test.csv')

    df_test.head()

    ## Apply similar transformation to the test dataset

    image_2D = None

    # Read files following the order of df_train

    for (index_img, row) in df_test.iterrows():

    #for (index_img, img_name) in enumerate(list_files):

        img_name = row['id_code'] + '.png'

        # DEBUG

        if DEBUG == True:

            if index_img > 10:

                break

        path= path_to_test_img + '/' + img_name

        image = circle_crop_v2(path, IMG_SIZE = IMG_SIZE)

        image = rgb2gray(image)

        img_1D_vector = image.reshape(img_size, 1)

        if index_img == 0:

            image_2D = img_1D_vector

        else:

            image_2D = np.concatenate((image_2D, img_1D_vector), axis=1)

    print("Done")

    print("Merging df")

    data_test_img = pd.DataFrame(data=np.transpose(image_2D), # values

              index=list(range(0, image_2D.shape[1])), # 1st column as index

              columns=list_features_names) 

    df_merged = df_test.join(data_test_img, how='outer')

    df_merged.head()

    df_merged.to_csv('test_preprocessed.csv', index=False)

    print("Done")
df_merged = None

data_test_img = None

df_test = None
if pre_process_image == True:

    train = pd.read_csv('train_preprocessed.csv')

else:

    train = pd.read_csv(path_to_preprocessed_data_model+'/train_preprocessed.csv')

train.head()
# PREPARE DATA FOR NEURAL NETWORK

train["label"] = train["diagnosis"]

Y_train = train["label"]

X_train = train.drop(labels = ["label", "diagnosis", "id_code"],axis = 1)

# Normalization:

X_train = X_train / 255.0

X_train = X_train.values.reshape(-1,IMG_SIZE,IMG_SIZE,1)

Y_train = to_categorical(Y_train, num_classes = 5)
# PREVIEW IMAGES

plt.figure(figsize=(15,4.5))

for i in range(20):  

    plt.subplot(3, 10, i+1)

    plt.imshow(X_train[i].reshape((IMG_SIZE,IMG_SIZE)),cmap=plt.cm.binary)

    plt.axis('off')

plt.subplots_adjust(wspace=-0.1, hspace=-0.1)

plt.show()
# CREATE MORE IMAGES VIA DATA AUGMENTATION

datagen = ImageDataGenerator(

        rotation_range=30) #let's only rotate images for the moment
# PREVIEW AUGMENTED IMAGES

X_train3 = X_train[9,].reshape((1,IMG_SIZE,IMG_SIZE,1))

Y_train3 = Y_train[9,].reshape((1,5))

plt.figure(figsize=(15,4.5))

for i in range(20):  

    plt.subplot(3, 10, i+1)

    X_train2, Y_train2 = datagen.flow(X_train3,Y_train3).next()

    #plt.imshow(X_train2[0].reshape((IMG_SIZE,IMG_SIZE)),cmap=plt.cm.binary)

    plt.imshow(X_train2[0].reshape((IMG_SIZE,IMG_SIZE)),cmap='gray')

    plt.axis('off')

    if i==9: X_train3 = X_train[11,].reshape((1,IMG_SIZE,IMG_SIZE,1))

    if i==19: X_train3 = X_train[18,].reshape((1,IMG_SIZE,IMG_SIZE,1))

plt.subplots_adjust(wspace=-0.1, hspace=-0.1)

plt.show()
# BUILD CONVOLUTIONAL NEURAL NETWORKS

nets = 2

model = [0] *nets

if train_model==True:

    print("Training model(s)")

    for j in range(nets):

        model[j] = Sequential()



        model[j].add(Conv2D(IMG_SIZE, kernel_size = 3, activation='relu', input_shape = (IMG_SIZE, IMG_SIZE, 1)))

        model[j].add(BatchNormalization())

        model[j].add(Conv2D(IMG_SIZE, kernel_size = 3, activation='relu'))

        model[j].add(BatchNormalization())

        model[j].add(Conv2D(IMG_SIZE, kernel_size = 5, strides=2, padding='same', activation='relu'))

        model[j].add(BatchNormalization())

        model[j].add(Dropout(0.4))



        model[j].add(Conv2D(int(IMG_SIZE*2), kernel_size = 3, activation='relu'))

        model[j].add(BatchNormalization())

        model[j].add(Conv2D(int(IMG_SIZE*2), kernel_size = 3, activation='relu'))

        model[j].add(BatchNormalization())

        model[j].add(Conv2D(int(IMG_SIZE*2), kernel_size = 5, strides=2, padding='same', activation='relu'))

        model[j].add(BatchNormalization())

        model[j].add(Dropout(0.4))



        model[j].add(Conv2D(int(IMG_SIZE*4), kernel_size = 4, activation='relu'))

        model[j].add(BatchNormalization())

        model[j].add(Flatten())

        model[j].add(Dropout(0.4))

        # Five categories

        model[j].add(Dense(5, activation='softmax'))



        # COMPILE WITH ADAM OPTIMIZER AND CROSS ENTROPY COST

        model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])



        # DECREASE LEARNING RATE EACH EPOCH

        annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

        # TRAIN NETWORKS

        history = [0] * nets

        epochs = 50

        for j in range(nets):

            X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size = 0.1)

            history[j] = model[j].fit_generator(datagen.flow(X_train2,Y_train2, batch_size=64),

                epochs = epochs, steps_per_epoch = X_train2.shape[0]//64,  

                validation_data = (X_val2,Y_val2), callbacks=[annealer], verbose=0)

            print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(

                j+1,epochs,max(history[j].history['acc']),max(history[j].history['val_acc']) ))



        # save model and architecture to single file

        for j in range(nets):

            model_name = "model{}.h5".format(j)

            print(model_name)

            model[j].save(model_name)

        print("Saved model(s) to disk")

else:

    print("Loading model(s) from disk")

    for j in range(nets):

        model_name = "/model{}.h5".format(j)

        # load model

        model[j] = load_model(path_to_preprocessed_data_model + model_name)

        # summarize model.

        model[j].summary()

    print("Done.")
# ENSEMBLE PREDICTIONS AND SUBMIT

results = np.zeros( (X_train.shape[0], 5) ) 

for j in range(nets):

    results = results + model[j].predict(X_train)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
# PREVIEW PREDICTIONS

plt.figure(figsize=(15,6))

for i in range(10):  

    plt.subplot(4, 10, i+1)

    plt.imshow(X_train[i].reshape((IMG_SIZE,IMG_SIZE)),cmap=plt.cm.binary)

    plt.title("{} ; {}".format(results[i],train["label"][i]),y=0.9)

    plt.axis('off')

plt.subplots_adjust(wspace=0.3, hspace=-0.1)

plt.show()
#confusion matrix

labels = ['0 - No DR', '1 - Mild', '2 - Moderate', '3 - Severe', '4 - Proliferative DR']

cnf_matrix = confusion_matrix(train["label"].astype('int'), results.astype('int'))

cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

df_cm = pd.DataFrame(cnf_matrix_norm, index=labels, columns=labels)

plt.figure(figsize=(16, 7))

sns.heatmap(df_cm, annot=True, fmt='.2f')

plt.show()
print("Train Cohen Kappa score: %.3f" % cohen_kappa_score(results.astype('int'), train['diagnosis'].astype('int'), weights='quadratic'))
if pre_process_image == True:

    test = pd.read_csv('test_preprocessed.csv')

else:

    test = pd.read_csv(path_to_preprocessed_data_model+'/test_preprocessed.csv')

train.head()
# PREPARE DATA FOR NEURAL NETWORK

X_id_code = test["id_code"]

X_test = test.drop(labels = ["id_code"],axis = 1)

# Normalization:

X_test = X_test / 255.0

X_test = X_test.values.reshape(-1,IMG_SIZE,IMG_SIZE,1)
# ENSEMBLE PREDICTIONS AND SUBMIT

results = np.zeros( (X_test.shape[0], 5) ) 

for j in range(nets):

    results = results + model[j].predict(X_test)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
# PREVIEW PREDICTIONS

plt.figure(figsize=(15,6))

for i in range(30):  

    plt.subplot(4, 10, i+1)

    plt.imshow(X_test[i].reshape((IMG_SIZE,IMG_SIZE)),cmap=plt.cm.binary)

    plt.title("predict=%d" % results[i],y=0.9)

    plt.axis('off')

plt.subplots_adjust(wspace=0.3, hspace=-0.1)

plt.show()
results = pd.DataFrame(results)

X_id_code = X_id_code.reset_index()
X_id_code["diagnosis"] = results["Label"]

X_id_code = X_id_code.drop(labels = ["index"],axis = 1)
X_id_code.head()
X_id_code.to_csv('submission.csv', index=False)