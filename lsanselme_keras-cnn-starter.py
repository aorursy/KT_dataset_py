import numpy as np 

import pandas as pd 

import os

import matplotlib.pyplot as plt

import matplotlib.image as mplimg

from matplotlib.pyplot import imshow



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split



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
os.listdir("../input/")
train_df = pd.read_csv("../input/train.csv")

train_df.head()
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
X.shape
y, label_encoder = prepare_labels(train_df['Id'])
y.shape
X, X_val, y, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, )
print(X.shape)

print(X_val.shape)

print(y.shape)

print(y_val.shape)

model = Sequential()



model.add(Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0', input_shape = (100, 100, 3)))



model.add(BatchNormalization(axis = 3, name = 'bn0'))

model.add(Activation('relu'))



model.add(MaxPooling2D((2, 2), name='max_pool'))

model.add(Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0.5'))



model.add(BatchNormalization(axis = 3, name = 'bn0.5'))

model.add(Activation('relu'))



model.add(MaxPooling2D((2, 2), name='max_pool1'))

model.add(Conv2D(64, (3, 3), strides = (1,1), name="conv1"))

model.add(Activation('relu'))

model.add(AveragePooling2D((3, 3), name='avg_pool'))



model.add(Flatten())

model.add(Dense(500, activation="relu", name='rl'))

model.add(Dropout(0.8))

model.add(Dense(y.shape[1], activation='softmax', name='sm'))



model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

model.summary()
history = model.fit(X, y, epochs=150, batch_size=100, verbose=1, validation_data=(X_val, y_val))
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.show()
test = os.listdir("../input/test/")

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

test_df.to_csv('submission.csv', index=False)
image_hash = [test[14]]

im_df = pd.DataFrame(image_hash, columns=col)

im_df['Id'] = ''

X = prepareImages(im_df, im_df.shape[0], "test")

X /= 255

prediction = model.predict(np.array(X), verbose=0)[0]



im_df.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(prediction.argsort()[-1:][::-1]))



print(im_df)