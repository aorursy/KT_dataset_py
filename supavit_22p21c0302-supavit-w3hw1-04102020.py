import numpy as np 

import pandas as pd 

import os

import matplotlib.pyplot as plt

import matplotlib.image as mplimg

from matplotlib.pyplot import imshow



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder



from keras import layers

from keras.preprocessing import image

from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D

from keras.layers import AveragePooling2D, MaxPooling2D, Dropout

from keras.models import Model



import keras.backend as K

from keras.models import Sequential



import imagehash



from keras.applications.resnet50 import ResNet50

from keras.applications.resnet50 import preprocess_input



import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)
train_df = pd.read_csv("../input/super-ai-image-classification/train/train/train.csv")

train_df.head()
img = image.load_img("../input/super-ai-image-classification/train/train/images/01b73b6d-4c99-4b47-a4a0-c15e9cebc252.jpg")

x = image.img_to_array(img)

x = preprocess_input(x)

plt.imshow(x)

plt.show()
def prepareImages(data, m, dataset):

    print("Preparing images")

    X_train = np.zeros((m, 224, 224, 3))

    count = 0

    

    for fig in data['id']:

        #load images into images of size 100x100x3

        img = image.load_img("../input/super-ai-image-classification/train/train/"+dataset+"/"+fig, target_size=(224, 224, 3))

        x = image.img_to_array(img)

        x = preprocess_input(x)



        X_train[count] = x

        if (count%500 == 0):

            print("Processing image: ", count+1, ", ", fig)

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
X = prepareImages(train_df, train_df.shape[0], "images")

X /= 255
plt.imshow(X[1,:,:,:])

plt.show()
y, label_encoder = prepare_labels(train_df['category'])
model = Sequential()



model.add(ResNet50(weights="imagenet", input_shape=(224,224,3)))



model.add(Dense(y.shape[1], activation='softmax', name='sm'))

model.layers[0].trainable = True



model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

model.summary()
history = model.fit(X, y, epochs=200, batch_size=100, verbose=1)
plt.plot(history.history['accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.show()
plt.plot(history.history['loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.show()
test = os.listdir("../input/super-ai-image-classification/val/val/images")



col = ['id']

test_df = pd.DataFrame(test, columns=col)

test_df['category'] = ''
def prepareImagesTest(data, m, dataset):

    print("Preparing images")

    Z_test = np.zeros((m, 224, 224, 3))

    count = 0

    

    for fig in data['id']:

        #load images into images of size 100x100x3

        img = image.load_img("../input/super-ai-image-classification/val/val/"+dataset+"/"+fig, target_size=(224, 224, 3))

        z = image.img_to_array(img)

        z = preprocess_input(z)



        Z_test[count] = z

        if (count%500 == 0):

            print("Processing image: ", count+1, ", ", fig)

        count += 1

    

    return Z_test
Z = prepareImagesTest(test_df, test_df.shape[0], "images")

Z /= 255
predictions = model.predict(np.array(Z), verbose=1)
for i, pred in enumerate(predictions):

    print(f"{i}, {pred[0]:.2f}, {pred[1]:.2f}, {pred.argmax()}")

    test_df.loc[i, 'category'] = pred.argmax()
test_df.head()

test_df.to_csv('./val.csv', index=False)