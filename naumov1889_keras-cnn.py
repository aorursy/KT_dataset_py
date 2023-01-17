IMG_SIZE = 300
import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

import cv2

import os



print('Tensorflow version:', tf.__version__)
train = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")

test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

#train_x = np.array([cv2.resize(np.array(cv2.imread("../input/aptos2019-blindness-detection/train_images/"+i+".png")),(IMG_SIZE,IMG_SIZE)) for i in train.id_code])

#train_y = np.array(train.diagnosis)

#test_x = np.array([cv2.resize(np.array(cv2.imread("../input/aptos2019-blindness-detection/test_images/"+i+".png")),(IMG_SIZE,IMG_SIZE)) for i in test.id_code])
"""

n = 10

cols = 5

rows = np.ceil(n/cols)

fig = plt.gcf()

fig.set_size_inches(cols * n, rows * n)

for i in range(n):

  plt.subplot(rows, cols, i+1)

  plt.imshow(test_x[i])

  #plt.title(train['diagnosis'][i], fontsize=40)

  plt.axis('off')

"""
"""

def crop_image1(img,tol=7):

    # img is image data

    # tol  is tolerance

        

    mask = img>tol

    return img[np.ix_(mask.any(1),mask.any(0))]

"""

def crop_image_from_gray(img,tol=7):

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img>tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

    #         print(img1.shape,img2.shape,img3.shape)

            img = np.stack([img1,img2,img3],axis=-1)

    #         print(img.shape)

        return img
def load_ben_color(data, img_size, sigmaX=10):

    if data.ndim == 4:  # array of images

        for i in range(len(data)):

            image = cv2.cvtColor(data[i], cv2.COLOR_BGR2RGB)

            image = crop_image_from_gray(image)

            image = cv2.resize(image, (img_size, img_size))

            data[i] = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), sigmaX), -4 ,128)

    elif data.ndim == 3:  # just a single image

        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

        data = crop_image_from_gray(data)

        data = cv2.resize(data, (img_size, img_size))

        data = cv2.addWeighted(data, 4, cv2.GaussianBlur(data, (0,0), sigmaX), -4 , 128)

    else: 

        return 0

    

    return data
def image_preprocessing(data, img_size, sigmaX=10):

    # cropping & Ben Graham's preprocessing method

    data = load_ben_color(data, img_size, sigmaX)

    

    # normalization (rescaling between 0 and 1)

    data = data.astype('float32')

    for i in range(len(data)):

        cv2.normalize(data[i],  data[i], 0, 1, cv2.NORM_MINMAX)

        

    return data
#train_x = image_preprocessing(train_x, IMG_SIZE, sigmaX=10)

#test_x = image_preprocessing(test_x, IMG_SIZE, sigmaX=10)
"""

n = 10

cols = 5

rows = np.ceil(n/cols)

fig = plt.gcf()

fig.set_size_inches(cols * n, rows * n)

for i in range(n):

  plt.subplot(rows, cols, i+1)

  plt.imshow(test_x[i])

  #plt.title(train['diagnosis'][i], fontsize=40)

  plt.axis('off')

"""
import pickle

"""

# pickle out

pickle_out_train_x = open('train_x_aptos2019.pickle', 'wb')

pickle.dump(train_x, pickle_out_train_x)

pickle_out_train_x.close()



pickle_out_train_y = open('train_y_aptos2019-blindness-detection.pickle', 'wb')

pickle.dump(train_y, pickle_out_train_y)

pickle_out_train_y.close()



pickle_out_test_x = open('test_x_aptos2019-blindness-detection.pickle', 'wb')

pickle.dump(test_x, pickle_out_test_x)

pickle_out_test_x.close()

"""



#"""

# pickle in

pickle_in_train_x = open('../input/preprocessed-data-aptos2019blindnessdetection/train_x_aptos2019.pickle', 'rb')

pickle_in_train_y = open('../input/preprocessed-data-aptos2019blindnessdetection/train_y_aptos2019.pickle', 'rb')

pickle_in_test_x = open('../input/preprocessed-data-aptos2019blindnessdetection/test_x_aptos2019.pickle', 'rb')



train_x = pickle.load(pickle_in_train_x)

train_y = pickle.load(pickle_in_train_y)

test_x = pickle.load(pickle_in_test_x)



print(train_x.shape, train_y.shape, test_x.shape)

#"""
n = 5

cols = 5

rows = np.ceil(n/cols)

fig = plt.gcf()

fig.set_size_inches(cols * n, rows * n)

for i in range(n):

  plt.subplot(rows, cols, i+1)

  plt.imshow(train_x[i])

  plt.title(train['diagnosis'][i], fontsize=40)

  plt.axis('off')
n = 5

cols = 5

rows = np.ceil(n/cols)

fig = plt.gcf()

fig.set_size_inches(cols * n, rows * n)

for i in range(n):

  plt.subplot(rows, cols, i+1)

  plt.imshow(test_x[i])

  #plt.title(train['diagnosis'][i], fontsize=40)

  plt.axis('off')
#"""

def create_model_1():

    layers_1 = [

        tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu, input_shape=train_x.shape[1:]),

        tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),

        tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu),

        tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),

        tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu),

        tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),

        tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', activation=tf.nn.relu),

        tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),

        tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='same', activation=tf.nn.relu),

        tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),

        tf.keras.layers.Flatten(),  

        tf.keras.layers.Dense(units=512, activation=tf.nn.relu),

        tf.keras.layers.Dense(units=256, activation=tf.nn.relu),

        tf.keras.layers.Dense(units=len(np.unique(train_y)), activation=tf.nn.softmax),

    ] 



    model_1 = tf.keras.Sequential(layers_1)

    model_1.compile(optimizer=tf.keras.optimizers.Adam(), #tf.optimizers.Adam(),

                 loss=tf.keras.losses.sparse_categorical_crossentropy, #tf.losses.SparseCategoricalCrossentropy(),

                 metrics=['accuracy'])

    

    return model_1

#"""
#"""

model_1 = create_model_1()

model_1.summary()

#"""
#"""

# https://www.youtube.com/watch?v=HxtBIwfy0kM

checkpoint_path = 'cp_model_1_aptos2019-blindness-detection.ckpt'

checkpoint_dir = os.path.dirname(checkpoint_path)



# Create checkpoint callback

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,

                                                 save_weights_only=True,

                                                 verbose=1)



model_1 = create_model_1()



model_1.fit(train_x, train_y, epochs=5, batch_size=32, 

            callbacks=[cp_callback])  # pass calback to training

#"""
#"""

train_predicted = model_1.predict(train_x)

train_predicted = [np.argmax(i) for i in train_predicted]



from sklearn.metrics import cohen_kappa_score

cohen_kappa_score(train_predicted, train_y, weights='quadratic')

#"""
""" Memory error here

# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator

datagen = tf.keras.preprocessing.image.ImageDataGenerator(

    rescale=1./255,

    rotation_range=30,

    brightness_range=[0.5, 1.5],

    zoom_range=[0.8, 1.2],

    horizontal_flip=True,

    vertical_flip=False)



datagen.fit(train_x)



checkpoint_path = 'cp_model_1_aptos2019-blindness-detection.ckpt'

checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,

                                                save_weights_only=True,

                                                verbose=1)



model_1 = create_model_1()



# fits the model on batches with real-time data augmentation:

model_1.fit_generator(datagen.flow(train_x, train_y, batch_size=32),

                      steps_per_epoch=len(train_x) / 32, epochs=5,

                      callbacks=[cp_callback])

"""
#model_1 = create_model_1()



#loss, acc = model_1.evaluate(x, y)

#print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
#model_1.load_weights('../input/blindness-1/cp_model_1_aptos2019-blindness-detection.ckpt')

#loss, acc = model_1.evaluate(x, y)

#print("Restored model, accuracy: {:5.2f}%".format(100*acc))
#"""

test_predicted = model_1.predict(test_x)

test_predicted = [np.argmax(i) for i in test_predicted]

test_result = pd.DataFrame({"id_code": test["id_code"].values, "diagnosis": test_predicted})

test_result.head()

#"""
test_result.to_csv('submission.csv', index=False)
n = 5

cols = 5

rows = np.ceil(n/cols)

fig = plt.gcf()

fig.set_size_inches(cols * n, rows * n)

for i in range(n):

    plt.subplot(rows, cols, i+1)

    plt.imshow(test_x[i])

    plt.title(test_predicted[i], fontsize=40)

    plt.axis('off') 
# How to count the occurrence of certain item in an ndarray (from numpy) in Python? 

# https://stackoverflow.com/questions/28663856/how-to-count-the-occurrence-of-certain-item-in-an-ndarray-in-python

unique, counts = np.unique(test_predicted, return_counts=True)

mydict = dict(zip(unique, counts))
plt.bar(unique, counts)

plt.show()