## load the libraries 

from keras.layers import Dense, Input, Conv2D, LSTM, MaxPooling2D, UpSampling2D , Flatten ,MaxPool2D ,BatchNormalization , Dropout

from keras.optimizers import RMSprop ,Adam



from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping

from keras.utils import to_categorical

from numpy import argmax, array_equal

import matplotlib.pyplot as plt

from keras.models import Model

from imgaug import augmenters

from random import randint

import pandas as pd

from keras.utils.np_utils import to_categorical



import numpy as np

### read dataset 

train = pd.read_csv("../input/fashion-mnist_train.csv")

train_x = train[list(train.columns)[1:]].values

train_y = train['label'].values



## normalize and reshape the predictors  

train_x = train_x / 255



## create train and validation datasets

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2)



## reshape the inputs

train_x = train_x.reshape(-1, 784)

val_x = val_x.reshape(-1, 784)
## input layer

input_layer = Input(shape=(784,))



## encoding architecture

encode_layer1 = Dense(1500, activation='relu')(input_layer)

encode_layer2 = Dense(1000, activation='relu')(encode_layer1)

encode_layer3 = Dense(500, activation='relu')(encode_layer2)



## latent view

latent_view   = Dense(10, activation='sigmoid')(encode_layer3)



## decoding architecture

decode_layer1 = Dense(500, activation='relu')(latent_view)

decode_layer2 = Dense(1000, activation='relu')(decode_layer1)

decode_layer3 = Dense(1500, activation='relu')(decode_layer2)



## output layer

output_layer  = Dense(784)(decode_layer3)



model = Model(input_layer, output_layer)
model.summary()
model.compile(optimizer='adam', loss='mse')

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

model.fit(train_x, train_x, epochs=20, batch_size=2048, validation_data=(val_x, val_x), callbacks=[early_stopping])
val_x.shape
preds = model.predict(val_x)
from PIL import Image 

f, ax = plt.subplots(1,5)

f.set_size_inches(80, 40)

for i in range(5):

    ax[i].imshow(val_x[i].reshape(28, 28))

plt.show()
f, ax = plt.subplots(1,5)

f.set_size_inches(80, 40)

for i in range(5):

    ax[i].imshow(preds[i].reshape(28, 28))

plt.show()
## recreate the train_x array and val_x array

train_x = train[list(train.columns)[1:]].values

train_x,valid_x,train_ground,valid_ground = train_test_split(train_x, train_x,test_size=0.2,random_state=13)



## normalize and reshape

train_x = train_x/255.

valid_x = valid_x/255.

train_ground = train_ground/255.

valid_ground = valid_ground/255.
train_x = train_x.reshape(-1, 28, 28, 1)

valid_x = valid_x.reshape(-1, 28, 28, 1)

train_ground = train_ground.reshape(-1, 28, 28, 1)

valid_ground = valid_ground.reshape(-1, 28, 28, 1)
# Lets add sample noise - Salt and Pepper

noise = augmenters.SaltAndPepper(0.1)

seq_object = augmenters.Sequential([noise])



train_x_n = seq_object.augment_images(train_x * 255) / 255

val_x_n = seq_object.augment_images(valid_x * 255) / 255
f, ax = plt.subplots(1,5)

f.set_size_inches(80, 40)

for i in range(5,10):

    ax[i-5].imshow(train_x[i].reshape(28, 28))

plt.show()
f, ax = plt.subplots(1,5)

f.set_size_inches(80, 40)

for i in range(5,10):

    ax[i-5].imshow(train_x_n[i].reshape(28, 28))

plt.show()
# input layer

input_layer = Input(shape=(28, 28, 1))



# encoding architecture

encoded_layer1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)

encoded_layer1 = MaxPool2D( (2, 2), padding='same')(encoded_layer1)

encoded_layer2 = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded_layer1)

encoded_layer2 = MaxPool2D( (2, 2), padding='same')(encoded_layer2)

encoded_layer3 = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded_layer2)

latent_view    = MaxPool2D( (2, 2), padding='same')(encoded_layer3)



# decoding architecture

decoded_layer1 = Conv2D(16, (3, 3), activation='relu', padding='same')(latent_view)

decoded_layer1 = UpSampling2D((2, 2))(decoded_layer1)

decoded_layer2 = Conv2D(32, (3, 3), activation='relu', padding='same')(decoded_layer1)

decoded_layer2 = UpSampling2D((2, 2))(decoded_layer2)

decoded_layer3 = Conv2D(64, (3, 3), activation='relu')(decoded_layer2)

decoded_layer3 = UpSampling2D((2, 2))(decoded_layer3)

output_layer   = Conv2D(1, (3, 3), padding='same')(decoded_layer3)



# compile the model

model_2 = Model(input_layer, output_layer)

model_2.compile(optimizer='adam', loss='mse')
model_2.summary()
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=5, mode='auto')

history = model_2.fit(train_x_n, train_x, epochs=10, batch_size=2048, validation_data=(val_x_n, valid_x), callbacks=[early_stopping])
preds = model_2.predict(val_x_n)
f, ax = plt.subplots(1,5)

f.set_size_inches(80, 40)

for i in range(5):

    ax[i].imshow(val_x_n[i].reshape(28, 28))

plt.show()
f, ax = plt.subplots(1,5)

f.set_size_inches(80, 40)

for i in range(5):

    ax[i].imshow(preds[i].reshape(28, 28))

plt.show()
### read dataset 

train = pd.read_csv("../input/fashion-mnist_train.csv")

test  = pd.read_csv("../input/fashion-mnist_test.csv")

train_x = train[list(train.columns)[1:]].values

train_y = train['label'].values

test_y=test["label"].values

test=test[list(test.columns)[1:]].values





## normalize and reshape the predictors  

train_x = train_x / 255

test= test/255



## create train and validation datasets

#train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2)



## reshape the inputs

train_x = train_x.reshape(-1, 28,28,1)

test = test.reshape(-1, 28,28,1)
train_x.shape , test.shape
label_dict = {

 0: 'A',

 1: 'B',

 2: 'C',

 3: 'D',

 4: 'E',

 5: 'F',

 6: 'G',

 7: 'H',

 8: 'I',

 9: 'J',

}


plt.figure(figsize=[5,5])



# Display the first image in training data

plt.subplot(121)

curr_img = np.reshape(train_x[0], (28,28))

curr_lbl = train_y[0]

plt.imshow(curr_img, cmap='gray')

plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")



# Display the first image in testing data

plt.subplot(122)

curr_img = np.reshape(test[0], (28,28))

curr_lbl = test_y[0]

plt.imshow(curr_img, cmap='gray')

plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")
from sklearn.model_selection import train_test_split

train_X,valid_X,train_ground,valid_ground = train_test_split(train_x,

                                                             train_x,

                                                             test_size=0.2,

                                                             random_state=13)
batch_size = 64

epochs = 200

inChannel = 1

x, y = 28, 28

input_img = Input(shape = (x, y, inChannel))

num_classes = 10
def encoder(input_img):

    #encoder

    #input = 28 x 28 x 1 (wide and thin)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32

    conv1 = BatchNormalization()(conv1)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)

    conv1 = BatchNormalization()(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64

    conv2 = BatchNormalization()(conv2)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)

    conv2 = BatchNormalization()(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)

    conv3 = BatchNormalization()(conv3)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    conv3 = BatchNormalization()(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 256 (small and thick)

    conv4 = BatchNormalization()(conv4)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)

    conv4 = BatchNormalization()(conv4)

    return conv4



def decoder(conv4):    

    #decoder

    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4) #7 x 7 x 128

    conv5 = BatchNormalization()(conv5)

    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)

    conv5 = BatchNormalization()(conv5)

    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5) #7 x 7 x 64

    conv6 = BatchNormalization()(conv6)

    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)

    conv6 = BatchNormalization()(conv6)

    up1 = UpSampling2D((2,2))(conv6) #14 x 14 x 64

    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 32

    conv7 = BatchNormalization()(conv7)

    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)

    conv7 = BatchNormalization()(conv7)

    up2 = UpSampling2D((2,2))(conv7) # 28 x 28 x 32

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1

    return decoded

autoencoder = Model(input_layer, decoder(encoder(input_layer)))

autoencoder.compile(loss='mean_squared_error', optimizer = 'rmsprop')
autoencoder.summary()
autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_ground))
loss = autoencoder_train.history['loss']

val_loss = autoencoder_train.history['val_loss']

epochs = range(200)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
autoencoder.save_weights('autoencoder.h5')
# Change the labels from categorical to one-hot encoding

train_Y_one_hot = to_categorical(train_y)

test_Y_one_hot = to_categorical(test_y)



# Display the change for category label using one-hot encoding

print('Original label:', train_y[1000])

print('After conversion to one-hot:', train_Y_one_hot[0])
train_X,valid_X,train_label,valid_label = train_test_split(train_x,train_Y_one_hot,test_size=0.2,random_state=13)

train_X.shape , valid_X.shape , train_label.shape , valid_label.shape
def fc(enco):

    flat = Flatten()(enco)

    x = Dense(50, activation='relu')(flat)

    x = Dropout(0.1)(x)

    x = Dense(30, activation='relu')(x)

    x = Dropout(0.1)(x)



    out = Dense(10, activation='softmax')(x)

    return out
encode = encoder(input_layer)

full_model = Model(input_layer,fc(encode))

for l1,l2 in zip(full_model.layers[:19],autoencoder.layers[0:19]):

    l1.set_weights(l2.get_weights())

autoencoder.get_weights()[0][1]

full_model.get_weights()[0][1]

for layer in full_model.layers[0:19]:

    layer.trainable = False

full_model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=1e-5),metrics=['accuracy'])

full_model.summary()
classify_train = full_model.fit(train_X, train_label, batch_size=64,epochs=40,verbose=1,validation_data=(valid_X, valid_label))

accuracy = classify_train.history['acc']

val_accuracy = classify_train.history['val_acc']

loss = classify_train.history['loss']

val_loss = classify_train.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training accuracy')

plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
test_eval = full_model.evaluate(test, test_Y_one_hot, verbose=0)
print('Test loss:', test_eval[0])

print('Test accuracy:', test_eval[1])

predicted_classes = full_model.predict(test)

predicted_classes = np.argmax(np.round(predicted_classes),axis=1)

predicted_classes.shape, test_y.shape
correct = np.where(predicted_classes==test_y)[0]

print ("Found %d correct labels" % len(correct))

for i, correct in enumerate(correct[:9]):

    plt.subplot(3,3,i+1)

    plt.imshow(test[correct].reshape(28,28), cmap='gray', interpolation='none')

    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_y[correct]))

    plt.tight_layout()
incorrect = np.where(predicted_classes!=test_y)[0]

print ("Found %d incorrect labels" % len(incorrect))

for i, incorrect in enumerate(incorrect[:9]):

    plt.subplot(3,3,i+1)

    plt.imshow(test[incorrect].reshape(28,28), cmap='gray', interpolation='none')

    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_y[incorrect]))

    plt.tight_layout()
from sklearn.metrics import classification_report

target_names = ["Class {}".format(i) for i in range(num_classes)]

print(classification_report(test_y, predicted_classes, target_names=target_names))