import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import zipfile
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set(style="darkgrid")

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from keras.utils import to_categorical
from keras import layers
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model

from keras import backend as K
K.tensorflow_backend._get_available_gpus()
from keras.models import Sequential

print(os.listdir("../input"))
import warnings
warnings.simplefilter("ignore")
## Taking a look at the beautiful pictures
path = '../input/train/'
path_test = '../input/test/'
fig = plt.figure(figsize = (80, 30))

for num, items in enumerate(os.listdir(path)[1500:1510]):
    y = fig.add_subplot(2, 5, num + 1)
    img = cv2.imread(path + items)
    new_img = cv2.resize(img, (128, 128), interpolation = cv2.INTER_AREA)
    y.imshow(new_img)

plt.show()
print("Number of images in the train set: ", len([iq for iq in os.scandir(path)]))
print("Number of images in the test set: ", len([iq for iq in os.scandir(path_test)]))
## Importing the data frame
df = pd.read_csv("../input/train.csv")
print(df.shape)
df.head().T
## Since, a test dataframe is not given... Using the image names from the test set will do
test_df = pd.DataFrame(columns = {"Image", "Id"})

for image_id in os.listdir(path_test):
    test_df = test_df.append({"Image" : image_id}, ignore_index = True)

print(test_df.shape) ## Make sure that the size of the dataframe is same as the size of the pictures in the directory
test_df.head().T
df.describe()
## Shuffling up the dataset using pandas sample function
n = 5 ## Set this number for shuffling the dataset these many number of times
for _ in range(n):
    df = df.sample(frac = 1)

df.shape
## Checking out the top 10 frequently occuring whale Ids
whales = df.groupby('Id')['Image'].nunique()
whales.sort_values(ascending=False)[0:10]
## Looking at these images with the top 10 Id
w_id = ['new_whale', 'w_23a388d', 'w_9b5109b', 'w_9c506f6', 'w_0369a5c',
        'w_700ebb4', 'w_3de579a', 'w_564a34b', 'w_fd3e556', 'w_88e4537']
fig = plt.figure(figsize = (80, 30))

for num, ids in enumerate(w_id):
    img_name = df.loc[df["Id"] == ids]['Image'].tolist() ## Getting the images names from the dataframe
    y = fig.add_subplot(2, 5, num + 1)
    img = cv2.imread(path + img_name[3]) ## Read only the first type of each id
    new_img = cv2.resize(img, (128, 128), interpolation = cv2.INTER_AREA)
    y.imshow(new_img)

plt.show()
## All we bascially need are the Fins of the whales, not the background of the images i.e. the water
## Doing some edge detection to pick out the whale fins only
fig = plt.figure(figsize = (80, 30))

for num, ids in enumerate(w_id):
    img_name = df.loc[df["Id"] == ids]['Image'].tolist() ## Getting the images names from the dataframe
    y = fig.add_subplot(2, 5, num + 1)
    img = cv2.imread(path + img_name[3]) ## Read only the first type of each id
    new_img = cv2.resize(img, (128, 128), interpolation = cv2.INTER_AREA)
    
    # Blurring the image
    median = cv2.medianBlur(new_img, 5)
    # Applying canny edge detection
    canny = cv2.Canny(median, 150, 150)
    y.imshow(canny)

## There are various other ways to do this -- but this is Baseline
plt.show()
## Converting the training images into numpy arrays
def convert_images(df):
    
    # Parameters for our images
    number_of_images = df.shape[0]
    IMAGE_WIDTH = 128
    IMAGE_HEIGHT = 128
    N_CHANNELS = 1
    
    # Creating an empty array of size 25361 x 128 x 128 x 3
    dataset = np.ndarray((number_of_images, IMAGE_HEIGHT, IMAGE_WIDTH), dtype = np.float32)
    print("Size of dataset: ", dataset.shape)
    
    i = 0
    # Iterate through the images and convert them to arrays
    for image_names, df_items in zip(os.listdir(path), df.itertuples()):
        img = cv2.imread(path + df_items[1], 1)
        new_img = cv2.resize(img, (128, 128), interpolation = cv2.INTER_AREA)
        # Blurring the image
        median = cv2.medianBlur(new_img, 5)
        # Applying canny edge detection
        canny = cv2.Canny(median, 150, 150)
        dataset[i] = canny
        i += 1
        if i % 2500 == 0: # Print all along the way
            print("Size of the last image was: ", dataset.shape, "")
            print("{} number of images are converted\n".format(i))
            
    print("--- All images are converted ---\nSize of the dataset is: ", dataset.shape)
    
    return dataset
dataset = convert_images(df)
## Getting our labels
Y = df["Id"].values
print(Y.shape)

## Need to one-hot encode the targets -- from "machinelearningmastery"
data = array(Y)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(data)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)

# invert first example
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
print(inverted)

Y = onehot_encoded
print("\nTarget variable size: ", Y.shape)
## Converting the dataset to a range btw 0 - 1
final_dataset = dataset / 255 # Since, the values in the pixels range from 0 - 255
final_dataset = final_dataset.reshape(df.shape[0], 128, 128, 1)
print(final_dataset.shape) ## All three channels of the first pixel of the first picture
# define the CNN model
def cnn_model():
    # create model
    model = Sequential()
    
    model.add(Conv2D(64, (5, 5), input_shape=(128, 128, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Dropout(0.2))
    model.add(Flatten())
    
    model.add(Dense(500, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(Y.shape[1], activation='softmax'))
    
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
model = cnn_model()
model.summary()
# %%time
# ## Fitting the model
# history = model.fit(final_dataset, Y, epochs = 10, batch_size = 1024, verbose = 1)
# ## Getting the accuracy scores
# scores = model.evaluate(final_dataset, Y, verbose=0)
# print('Final CNN accuracy: ', scores[1]*100, "%")
# gc.collect()
# # Plotting the history of accuracy 
# plt.plot(history.history['acc'])
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train'], loc = 'upper left')
# plt.show()
# # Summarizing the history of loss
# plt.plot(history.history['loss'])
# plt.title('Model Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend(['Train'], loc = 'upper left')
# plt.show()
# test_dataset = convert_images(test_df)
# ## Converting the dataset to a range btw 0 - 1
# test_final_dataset = test_dataset / 255 # Since, the values in the pixels range from 0 - 255
# test_final_dataset = test_final_dataset.reshape(df.shape[0], 128, 128, 1)
# print(test_final_dataset.shape) 
# predictions = model.predict(test_final_dataset, verbose = 1)
# for i, pred in enumerate(predictions):
#     test_df.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))
# test_df.head(10)
# # test_df.to_csv('submission.csv', index=False)
