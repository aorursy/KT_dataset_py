import datetime as dt

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

sns.set_style('whitegrid')





import os

from keras.applications import xception

from keras.preprocessing import image

from mpl_toolkits.axes_grid1 import ImageGrid

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



import cv2

from scipy.stats import uniform



from tqdm import tqdm

from glob import glob





from keras.models import Model, Sequential

from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Masking

from keras.utils import np_utils, to_categorical







from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
## import Keras and its module for image processing and model building

import keras

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img



from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
#copying the pretrained models to the cache directory

cache_dir = os.path.expanduser(os.path.join('~', '.keras'))

if not os.path.exists(cache_dir):

    os.makedirs(cache_dir)

models_dir = os.path.join(cache_dir, 'models')

if not os.path.exists(models_dir):

    os.makedirs(models_dir)



#copy the Xception models

!cp ../input/keras-pretrained-models/xception* ~/.keras/models/

#show

!ls ~/.keras/models
img = load_img('../input/sheep-goat/sheep_goat/goat/goat28.jpg')  # this is a PIL image

x = img_to_array(img)  # this is a Numpy array 

print('image shape: ', x.shape)



print('Goat Image')

plt.imshow(img)

plt.show()





img = load_img('../input/sheep-goat/sheep_goat/sheep/sheep3.jpg')  # this is a PIL image

x = img_to_array(img)  # this is a Numpy array 

print('Sheep Image')

plt.imshow(img)

plt.show()





dir_kaggle ='../input/sheep-goat/'

data_kaggle ='../input/sheep-goat/sheep_goat/'

goat  ='../input/sheep-goat/sheep_goat/goat/'

sheep ='../input/sheep-goat/sheep_goat/sheep/'





class_data= ['goat','sheep']

len_class_data = len(class_data)
image_count = {}

train_data = []



for i , class_data in tqdm(enumerate(class_data)):

    class_folder = os.path.join(data_kaggle,class_data)

    label = class_data

    image_count[class_data] = []

    

    for path in os.listdir(os.path.join(class_folder)):

        image_count[class_data].append(class_data)

        train_data.append(['{}/{}'.format(class_data, path), i, class_data])
#show image count

for key, value in image_count.items():

    print('{0} -> {1}'.format(key, len(value)))
#create a dataframe

df = pd.DataFrame(train_data, columns=['file', 'id', 'label'])

df.shape

df.head()
cnt_pro = df['label'].value_counts()

plt.figure(figsize=(6,4))

sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('label', fontsize=12)

plt.xticks(rotation=80)

plt.show();
#masking function

def create_mask_for_plant(image):

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)



    lower_hsv = np.array([0,0,250])

    upper_hsv = np.array([250,255,255])

    

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    

    return mask



#image segmentation function

def segment_image(image):

    mask = create_mask_for_plant(image)

    output = cv2.bitwise_and(image, image, mask = mask)

    return output/255



#sharpen the image

def sharpen_image(image):

    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)

    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)

    return image_sharp



# function to get an image

def read_img(filepath, size):

    img = image.load_img(os.path.join(data_kaggle, filepath), target_size=size)

    #convert image to array

    img = image.img_to_array(img)

    return img
nb_rows = 3

nb_cols = 5

fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(10, 5));

plt.suptitle('SAMPLE IMAGES');

for i in range(0, nb_rows):

    for j in range(0, nb_cols):

        axs[i, j].xaxis.set_ticklabels([]);

        axs[i, j].yaxis.set_ticklabels([]);

        axs[i, j].imshow((read_img(df['file'][np.random.randint(100)], (255,255)))/255.);

plt.show();
#get an image

img = read_img(df['file'][12],(255,255))

#mask

image_mask = create_mask_for_plant(img)

#segmentation

image_segmented = segment_image(img)

#sharpen the image

image_sharpen = sharpen_image(image_segmented)



fig, ax = plt.subplots(1, 4, figsize=(10, 5));

plt.suptitle('SAMPLE PROCESSED IMAGE', x=0.5, y=0.8)

plt.tight_layout(1)



ax[0].set_title('ORIGINAL', fontsize=12)

ax[1].set_title('MASK', fontsize=12)

ax[2].set_title('SEGMENTED', fontsize=12)

ax[3].set_title('SHARPEN', fontsize=12)





ax[0].imshow(img/255);

ax[1].imshow(image_mask);

ax[2].imshow(image_segmented);

ax[3].imshow(image_sharpen);



INPUT_SIZE=255



##preprocess the input

X_train = np.zeros((len(df), INPUT_SIZE, INPUT_SIZE, df.shape[1]), dtype='float')

for i, file in tqdm(enumerate(df['file'])):

    #read image

    img = read_img(file,(INPUT_SIZE,INPUT_SIZE))

    #masking and segmentation

    image_segmented = segment_image(img)

    #sharpen

    image_sharpen = sharpen_image(image_segmented)

    x = xception.preprocess_input(np.expand_dims(image_sharpen.copy(), axis=0))

    X_train[i] = x
print('Train Image Shape: ', X_train.shape)

print('Train Image Size: ', X_train.size)
y = df['id']

train_x, train_val, y_train, y_val = train_test_split(X_train, y, test_size=0.1, random_state=101)
print('GOAT IMAGES ON TRAINING DATA: ',y_train[y_train==0].shape[0])

print('SHEEP IMAGES ON TRAINING DATA: ',y_train[y_train==1].shape[0])
##get the features

xception_bf = xception.Xception(weights='imagenet', include_top=False, pooling='avg')

bf_train_x = xception_bf.predict(train_x, batch_size=32, verbose=1)

bf_train_val = xception_bf.predict(train_val, batch_size=32, verbose=1)
#print shape of feature and size

print('Train Shape: ', bf_train_x.shape)

print('Train Size: ', bf_train_x.size)



print('Validation Shape: ', bf_train_val.shape)

print('Validation Size: ', bf_train_val.size)
from keras.models import Sequential

from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, Input

from keras.optimizers import SGD, Adam

from keras.utils import np_utils

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

#keras Sequential model

model = Sequential()

model.add(Dense(units = 75 , activation = 'relu', input_dim=bf_train_x.shape[1]))

model.add(Dense(units = 2, activation = 'softmax'))



model.add(Flatten())

model.add(Dropout(0.15))

model.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])

model.summary()
#train the model 

history = model.fit(bf_train_x, y_train, epochs=1000, batch_size=32);
from keras.utils import plot_model

plot_model(model, to_file='model.png')
plt.plot(history.history['accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epochs')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

plt.savefig('model_accuracy.png')

# summarize history for loss

plt.plot(history.history['loss'])

#plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epochs')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

plt.savefig('model_loss.png')
#predict the validation data

predictions = model.predict_classes(bf_train_val)
confusion_mat = confusion_matrix(y_val, predictions)



plt.figure(figsize=(4,4))

sns.heatmap(confusion_mat, square=True, annot=True,

            yticklabels=['Goat', 'Sheep'],

            xticklabels=['Goat', 'Sheep']);

plt.title('CONFUSION MATRIX');

plt.xlabel('Y_TRUE');

plt.ylabel("PREDICTIONS");
print(classification_report(y_val, predictions))