import numpy as np 

import pandas as pd 

import os

import matplotlib.pyplot as plt

import matplotlib.image as mplimg

from matplotlib.pyplot import imshow



from keras import optimizers

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder



from keras import layers

from keras.preprocessing import image

from keras.applications.imagenet_utils import preprocess_input

from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D

from keras.layers import AveragePooling2D, MaxPooling2D, Dropout

from keras.models import Model



import keras.backend as K

from keras.models import Sequential



import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)
train_df = pd.read_csv("../input/humpback-whale-identification/train.csv")

train_df.head()


plt.figure(figsize=(25,25))

ax1=plt.subplot(5, 2, 1)

img = image.load_img("../input/humpback-whale-identification/train/"+train_df['Image'][0])

print(train_df['Id'][0])

imshow(img)



ax1=plt.subplot(5, 2, 2)

img = image.load_img("../input/humpback-whale-identification/train/"+train_df['Image'][1])

print(train_df['Id'][1])

imshow(img)





ax1=plt.subplot(5, 2, 3)

img = image.load_img("../input/humpback-whale-identification/train/"+train_df['Image'][2])

print(train_df['Id'][2])

imshow(img)





ax1=plt.subplot(5, 2, 4)

img = image.load_img("../input/humpback-whale-identification/train/"+train_df['Image'][3])

print(train_df['Id'][3])

imshow(img)



ax1=plt.subplot(5, 2, 5)

img = image.load_img("../input/humpback-whale-identification/train/"+train_df['Image'][4])

print(train_df['Id'][4])

imshow(img)





ax1=plt.subplot(5, 2, 6)

img = image.load_img("../input/humpback-whale-identification/train/"+train_df['Image'][5])

print(train_df['Id'][5])

imshow(img)





ax1=plt.subplot(5, 2, 7)

img = image.load_img("../input/humpback-whale-identification/train/"+train_df['Image'][6])

print(train_df['Id'][6])

imshow(img)







ax1=plt.subplot(5, 2, 8)

img = image.load_img("../input/humpback-whale-identification/train/"+train_df['Image'][7])

print(train_df['Id'][7])

imshow(img)





ax1=plt.subplot(5, 2, 9)

img = image.load_img("../input/humpback-whale-identification/train/"+train_df['Image'][8])

print(train_df['Id'][8])

imshow(img)





ax1=plt.subplot(5, 2, 10)

img = image.load_img("../input/humpback-whale-identification/train/"+train_df['Image'][9])

print(train_df['Id'][9])

imshow(img)
from progressbar import ProgressBar





def prepareImages(data, m, dataset):

    pbar = ProgressBar()

    X_train = np.zeros((m, 100, 100, 3))

    count = 0

    

    for fig in pbar(data['Image']):

        #load images into images of size 100x100x3

        img = image.load_img("../input/humpback-whale-identification/"+dataset+"/"+fig, target_size=(100, 100, 3))

        x = image.img_to_array(img)

        x = preprocess_input(x)



        X_train[count] = x

        count += 1

    

    return X_train



def prepare_labels(y):

    values = np.array(y)

    label_encoder = LabelEncoder()

    integer_encoded = label_encoder.fit_transform(values)

    # print(integer_encoded)



    onehot_encoder = OneHotEncoder(sparse=False)

    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    # print(onehot_encoded)



    y = onehot_encoded

    # print(y.shape)

    return y, label_encoder
X = prepareImages(train_df, train_df.shape[0], "train")

X /= 255
y_train, label_encoder = prepare_labels(train_df['Id'])
y_train.shape
from pathlib import Path



from tqdm import tqdm

from keras import applications



weights = Path('../input/xception-weight/xception_weights_tf_dim_ordering_tf_kernels_notop (1).h5')

# base_model = applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights=weights, input_shape=(100, 100, 3) )#input_tensor=None, input_shape=None, pooling=None, classes=1000)

base_model =applications.xception.Xception(input_shape=(100, 100, 3), include_top=False, weights=weights)## set model architechture

x = base_model.output

x = Flatten()(x)

x = Dense(256, activation='relu')(x)

x = Dense(256, activation='relu')(x)

predictions = Dense(y_train.shape[1], activation='softmax')(x) 

model = Model(input=base_model.input, output=predictions)



model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-4),

              metrics=['accuracy'])



model.summary()
from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(

        rotation_range=0,

        width_shift_range=0,

        height_shift_range=0,

        shear_range=0,

        zoom_range=0,

        vertical_flip=True,

        horizontal_flip=True)



train_datagen.fit(X)
history = model.fit_generator(

    train_datagen.flow(X, y_train, batch_size=100),

    steps_per_epoch=10,

    epochs=1000

)
plt.plot(history.history['loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.show()
plt.plot(history.history['acc'])

plt.title('Model categorical accuracy')

plt.ylabel('categorical accuracy')

plt.xlabel('Epoch')

plt.show()
test = os.listdir("../input/humpback-whale-identification/test/")

print(len(test))
col = ['Image']

test_df = pd.DataFrame(test, columns=col)

test_df['Id'] = ''
X = prepareImages(test_df, test_df.shape[0], "test")

X /= 255
predictions = model.predict(np.array(X), verbose=1)
for i, pred in enumerate(predictions):

    test_df.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))
test_df.head(10)
test_df.to_csv('ans.csv', index=False)


plt.figure(figsize=(25,25))

ax1=plt.subplot(5, 2, 1)

img = image.load_img("../input/humpback-whale-identification/test/"+test_df['Image'][0])

print(test_df['Id'][0])

imshow(img)



ax1=plt.subplot(5, 2, 2)

img = image.load_img("../input/humpback-whale-identification/test/"+test_df['Image'][1])

print(test_df['Id'][1])

imshow(img)





ax1=plt.subplot(5, 2, 3)

img = image.load_img("../input/humpback-whale-identification/test/"+test_df['Image'][2])

print(test_df['Id'][2])

imshow(img)





ax1=plt.subplot(5, 2, 4)

img = image.load_img("../input/humpback-whale-identification/test/"+test_df['Image'][3])

print(test_df['Id'][3])

imshow(img)



ax1=plt.subplot(5, 2, 5)

img = image.load_img("../input/humpback-whale-identification/test/"+test_df['Image'][4])

print(test_df['Id'][4])

imshow(img)





ax1=plt.subplot(5, 2, 6)

img = image.load_img("../input/humpback-whale-identification/test/"+test_df['Image'][5])

print(test_df['Id'][5])

imshow(img)





ax1=plt.subplot(5, 2, 7)

img = image.load_img("../input/humpback-whale-identification/test/"+test_df['Image'][6])

print(test_df['Id'][6])

imshow(img)







ax1=plt.subplot(5, 2, 8)

img = image.load_img("../input/humpback-whale-identification/test/"+test_df['Image'][7])

print(test_df['Id'][7])

imshow(img)





ax1=plt.subplot(5, 2, 9)

img = image.load_img("../input/humpback-whale-identification/test/"+test_df['Image'][8])

print(test_df['Id'][8])

imshow(img)





ax1=plt.subplot(5, 2, 10)

img = image.load_img("../input/humpback-whale-identification/test/"+test_df['Image'][9])

print(test_df['Id'][9])

imshow(img)