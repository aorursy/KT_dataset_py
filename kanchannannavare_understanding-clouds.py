import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from skimage.color import rgb2gray

import cv2

import matplotlib.pyplot as plt

%matplotlib inline

from scipy import ndimage

from sklearn.model_selection import train_test_split

#from PIL import Image

import cv2

from sklearn.preprocessing import LabelBinarizer

import keras

from keras import backend as K

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation

from keras.losses import categorical_crossentropy, binary_crossentropy

from keras.optimizers import Adam

from keras.utils import to_categorical
path = '../input/understanding_cloud_organization/'

data = pd.read_csv(path + 'train.csv')

data = data.fillna(-1)

tr, test = train_test_split(data, test_size=0.5)

tr.head(10)
train = tr[tr['EncodedPixels']!= -1]

train['ImageId'] = tr['Image_Label'].apply(lambda x : x.split('_')[0])

train['ClassId'] = tr['Image_Label'].apply(lambda x : x.split('_')[1])

# train = train[train['ImageId'].unique()]

train = train[['ImageId', 'ClassId', 'EncodedPixels']]

print(train.shape)

train.head(10)
images_path = "../input/understanding_cloud_organization/train_images/"

img = train.iloc[0]["ImageId"]

classId = train.iloc[0]["ClassId"]

pixels = train.iloc[0]["EncodedPixels"]



print("class", classId)

image = plt.imread(images_path + img)

image.shape

plt.imshow(image)



print('encoded pixels', image)
classes = train['ClassId'].value_counts()

plt.bar(classes.index, classes)

plt.show()
def rle_to_mask(rle_string, img):

    rows, cols = img.shape[0], img.shape[1]

    img = np.zeros(rows*cols, dtype=np.uint8)

    if rle_string == -1:

        return img

    else:

        rle_numbers = [int(x) for x in rle_string.split(' ')]

        rle_pairs = np.array(rle_numbers).reshape(-1,2)



        for index, length in rle_pairs:

            index -= 1

            img[index:index+length] = 255

        img = img.reshape(cols,rows)

        img = img.T

        img = image = np.expand_dims(img, axis=2)



        return img



# print(train.iloc[1]['EncodedPixels'])

# print(train.iloc[1]['ImageId'])



print(rle_to_mask(train.iloc[1]['EncodedPixels'], cv2.imread(images_path + train.iloc[1]['ImageId'])))
image_ids = list(train['ImageId'])

pixels = []

print(len(image_ids))

c = 0

for i in image_ids:

    c = c + 1

#     print(c)

    img = cv2.imread(images_path + i)

    img = cv2.resize(img, (64, 64))

    tmp = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    types = list(train[train['ImageId']==i]['EncodedPixels'])

    for j in types:

        tmp = tmp + img + rle_to_mask(j, img)/255.0

    pixels.append(tmp)
encoder = LabelBinarizer()

transfomed_label = encoder.fit_transform(train['ClassId'])

print(train['ClassId'][:5])

print(transfomed_label[:5])
def dice_coef(y_true, y_pred, smooth=1):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def dice_loss(y_true, y_pred):

    smooth = 1.

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = y_true_f * y_pred_f

    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return 1. - score



def bce_dice_loss(y_true, y_pred):

    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
model = Sequential()



input_shape = (64, 64, 3)





model.add(Conv2D(32, kernel_size=(2, 2), strides=(1, 1), activation='relu', input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



model.add(Conv2D(32, kernel_size=(2, 2), strides=(1, 1), activation='relu', input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



model.add(Conv2D(32, kernel_size=(2, 2), strides=(1, 1), activation='relu', input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



model.add(Conv2D(16, kernel_size=(2, 2), strides=(1, 1), activation='relu', input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



model.add(Conv2D(16, kernel_size=(2, 2), strides=(1, 1), activation='relu', input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



model.add(Conv2D(8, kernel_size=(2, 2), strides=(1, 1), activation='relu', input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



model.add(Flatten())

model.add(Dense(8))



model.add(Dense(4, activation='sigmoid'))

model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_coef])

model.fit([pixels], transfomed_label, batch_size=1, epochs=24, validation_split=0.1)
model.summary()
def rle2mask(rle, width, height):

    if rle == '-1':

        return np.zeros((width, height))

    mask = np.zeros(width * height)

    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]

    lengths = array[1::2]



    current_position = 0

    for index, start in enumerate(starts):

        current_position += start

        mask[current_position:current_position+lengths[index]] = 255

        current_position += lengths[index]



    return mask.reshape(width, height)
print(test.iloc[77])

pixels_test = []

imgs = test.iloc[77]['Image_Label'].split('_')

img = cv2.imread(images_path + imgs[0])

img = cv2.resize(img, (64, 64))

tmp = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

types = list(train[train['ImageId']==i]['EncodedPixels'])

# print('image', img)

tmp = tmp + img + rle_to_mask(test.iloc[21]['EncodedPixels'], img)/255.0

pixels_test.append(tmp)
test_img = cv2.imread(images_path + imgs[0])

testimg = cv2.resize(test_img, (64, 64))

plt.imshow(testimg)

# tmp =  np.zeros((testimg.shape[0], testimg.shape[1], 1), dtype=np.uint8) + testimg + rle_to_mask(test.iloc[20]['EncodedPixels'], img)

# plt.imshow(tmp)



# px = rle_to_mask(test.iloc[82]['EncodedPixels'], img)

# print(px)
test_res = model.predict([pixels_test])
classes = ['fish', 'flower', 'gravel', 'sugar']



print(classes[np.argmax(test_res)])