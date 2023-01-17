#All Imports here

import os

import numpy as np

from numpy.random import seed

seed(0)

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg #For displaying images

import keras



from keras.models import Sequential

from keras.layers import LeakyReLU, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, ReLU, Dropout

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import cross_val_score

from skimage.transform import resize
meta_data = pd.read_csv("../input/HAM10000_metadata.csv")

print(meta_data.info())

meta_data.age.fillna(np.mean(meta_data.age), inplace = True)

meta_data.loc[:, ['dx', 'dx_type', 'sex', 'localization']] = meta_data[['dx', 'dx_type', 'sex', 'localization']].astype('category')

print(meta_data.info())

meta_data.sort_values(by='image_id', inplace = True)

meta_data.reset_index(inplace = True, drop = True)
img_list = os.listdir("../input/ham10000_images_part_1")

img_list = img_list+os.listdir("../input/ham10000_images_part_2")

img_list = sorted(img_list)



def get_img(name):

    try:

        return mpimg.imread("../input/ham10000_images_part_1/"+name)

    except:

        return mpimg.imread("../input/ham10000_images_part_2/"+name)



print('Shape', get_img(img_list[0]).shape)
#Creating the datasets

X = []

y = []

for i in range(len(img_list)):

    img = get_img(img_list[i])

    img = resize(img, (75, 100 ))

    X.append(img)

    y.append(meta_data.iloc[i, 2])

X = np.array(X)

y = np.array(y)

y = OneHotEncoder().fit_transform(y.reshape(-1,1)).toarray()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify = y)
print("Before Normalizing")

plt.imshow(X_test[4])
mean = []

std = []

for i in range(3):

    mean = np.mean(X_train[:,:,:,i])

    std = np.std(X_train[:,:,:,i])

    X_train[:,:,:,i] = (X_train[:,:,:,i] - mean)/std;

    X_test[:,:,:,i] = (X_test[:,:,:,i] - mean)/std;
print("After normalizing")

plt.imshow(X_test[4])


class ResNet(keras.Model):



    def __init__(self, num_classes=3):

        super(ResNet, self).__init__(name='ResNet')

        self.num_classes = num_classes



        self.pool = MaxPooling2D()

        

        self.cnn_input = Conv2D(64, kernel_size=(3,3), strides=2, activation='relu', input_shape=(75,100,3))

        self.cnn64_1 = Conv2D(64, kernel_size=(3,3), strides=1, activation='relu', padding='Same')

        self.cnn64_2 = Conv2D(64, kernel_size=(3,3), strides=1, activation='relu', padding='Same')

        self.cnn64_3 = Conv2D(64, kernel_size=(3,3), strides=1, activation='relu', padding='Same')

        self.cnn64_4 = Conv2D(64, kernel_size=(3,3), strides=1, activation='relu', padding='Same')

        self.cnn64_deact_1 = Conv2D(64, kernel_size=(3,3), strides=1, activation=None, padding='Same')

        self.cnn64_deact_2 = Conv2D(64, kernel_size=(3,3), strides=1, activation=None, padding='Same')

        

        self.cnn128by2 = Conv2D(128, kernel_size=(3,3), strides=2, activation='relu')

        self.cnn128_1 = Conv2D(128, kernel_size=(3,3), strides=1, activation='relu', padding = 'Same')

        self.cnn128_2 = Conv2D(128, kernel_size=(3,3), strides=1, activation='relu', padding = 'Same')

        self.cnn128_3 = Conv2D(128, kernel_size=(3,3), strides=1, activation='relu', padding = 'Same')

        self.cnn128_4 = Conv2D(128, kernel_size=(3,3), strides=1, activation='relu', padding = 'Same')

        self.cnn128_deact_1 = Conv2D(128, kernel_size=(3,3), strides=1, activation=None, padding='Same')

        self.cnn128_deact_2 = Conv2D(128, kernel_size=(3,3), strides=1, activation=None, padding='Same')

        

        self.cnn256by2 = Conv2D(256, kernel_size=(3,3), strides=2, activation='relu')

        self.cnn256_1 = Conv2D(256, kernel_size=(3,3), strides=1, activation='relu', padding = 'Same')

        self.cnn256_2 = Conv2D(256, kernel_size=(3,3), strides=1, activation='relu', padding = 'Same')

        self.cnn256_3 = Conv2D(256, kernel_size=(3,3), strides=1, activation='relu', padding = 'Same')

        self.cnn256_4 = Conv2D(256, kernel_size=(3,3), strides=1, activation='relu', padding = 'Same')

        self.cnn256_deact_1 = Conv2D(256, kernel_size=(3,3), strides=1, activation=None, padding='Same')

        self.cnn256_deact_2 = Conv2D(256, kernel_size=(3,3), strides=1, activation=None, padding='Same')

        

        self.cnn512by2 = Conv2D(512, kernel_size=(3,3), strides=2, activation='relu')

        self.cnn512_1 = Conv2D(512, kernel_size=(3,3), strides=1, activation='relu', padding='Same')

        self.cnn512_2 = Conv2D(512, kernel_size=(3,3), strides=1, activation='relu', padding='Same')

        self.cnn512_3 = Conv2D(512, kernel_size=(3,3), strides=1, activation='relu', padding='Same')

        self.cnn512_4 = Conv2D(512, kernel_size=(3,3), strides=1, activation='relu', padding='Same')

        self.cnn512_deact_1 = Conv2D(512, kernel_size=(3,3), strides=1, activation=None, padding='Same')

        self.cnn512_deact_2 = Conv2D(512, kernel_size=(3,3), strides=1, activation=None, padding='Same')

        

        self.flatten = Flatten()

        self.dense_1 = Dense(1000, activation='relu')

        self.dense_2 = Dense(500, activation='relu')

        self.dense_3 = Dense(250, activation='relu')

        self.dense_output = Dense(self.num_classes, activation='softmax')

        

        self.drop = Dropout(0.2)

        self.normalize_1 = BatchNormalization(axis=-1)

        self.normalize_2 = BatchNormalization(axis=-1)

        self.normalize_3 = BatchNormalization(axis=-1)

        self.normalize_4 = BatchNormalization(axis=-1)

        self.normalize_5 = BatchNormalization(axis=-1)

        self.normalize_6 = BatchNormalization(axis=-1)

        self.normalize_7 = BatchNormalization(axis=-1)

        self.normalize_8 = BatchNormalization(axis=-1)

        self.normalize_9 = BatchNormalization(axis=-1)

        self.normalize_10 = BatchNormalization(axis=-1)

        self.normalize_11 = BatchNormalization(axis=-1)

        self.normalize_12 = BatchNormalization(axis=-1)

        self.normalize_13 = BatchNormalization(axis=-1)

        self.normalize_14 = BatchNormalization(axis=-1)

        self.normalize_15 = BatchNormalization(axis=-1)

        self.normalize_16 = BatchNormalization(axis=-1)

        self.normalize_17 = BatchNormalization(axis=-1)

        self.normalize_18 = BatchNormalization(axis=-1)

        self.normalize_19 = BatchNormalization(axis=-1)

        self.normalize_20 = BatchNormalization(axis=-1)

        self.normalize_21 = BatchNormalization(axis=-1)

        self.normalize_22 = BatchNormalization(axis=-1)

        self.normalize_23 = BatchNormalization(axis=-1)

        self.normalize_24 = BatchNormalization(axis=-1)

        self.normalize_25 = BatchNormalization(axis=-1)

        self.normalize_26 = BatchNormalization(axis=-1)

        self.normalize_27 = BatchNormalization(axis=-1)

        

        self.relu = LeakyReLU(alpha=0.3)

        

    def call(self, inputs):

        x = self.cnn_input(inputs)

        res = self.pool(x);x=res

        #Passing to 3x3x64 filter

        x = self.cnn64_1(x)

        x = self.normalize_1(x)

        x = self.cnn64_2(x)

        x = self.normalize_2(x)

        res = self.relu(self.cnn64_deact_1(x)+res);x=res

        x = self.normalize_3(x)

        x = self.cnn64_3(x)

        x = self.normalize_4(x)

        x = self.cnn64_4(x)

        x = self.normalize_5(x)

        x = self.relu(self.cnn64_deact_2(x)+res)

        x = self.normalize_6(x)

        res = self.cnn128by2(x);x=res

        x = self.normalize_7(x)

        #Passing to 3x3x128 filter

        x = self.cnn128_1(x)

        x = self.normalize_8(x)

        x = self.cnn128_2(x)

        x = self.normalize_9(x)

        res = self.relu(self.cnn128_deact_1(x)+res);x=res

        x = self.normalize_10(x)

        x = self.cnn128_3(x)

        x = self.normalize_11(x)

        x = self.cnn128_4(x)

        x = self.normalize_12(x)

        x = self.relu(self.cnn128_deact_2(x)+res)

        x = self.normalize_13(x)

        res = self.cnn256by2(x);x=res

        x = self.normalize_14(x)

        #Passing to 3x3x256 filter

        x = self.cnn256_1(x)

        x = self.normalize_15(x)

        x = self.cnn256_2(x)

        x = self.normalize_16(x)

        res = self.relu(self.cnn256_deact_1(x)+res);x=res

        x = self.normalize_17(x)

        x = self.cnn256_3(x)

        x = self.normalize_18(x)

        x = self.cnn256_4(x)

        x = self.normalize_19(x)

        x = self.relu(self.cnn256_deact_2(x)+res)

        x = self.normalize_20(x)

        res = self.cnn512by2(x);x=res

        x = self.normalize_21(x)

        #Passing to 3x3x512 filter

        x = self.cnn512_1(x)

        x = self.normalize_22(x)

        x = self.cnn512_2(x)

        x = self.normalize_23(x)

        res = self.relu(self.cnn512_deact_1(x)+res);x=res

        x = self.normalize_24(x)

        x = self.cnn512_3(x)

        x = self.normalize_25(x)

        x = self.cnn512_4(x)

        x = self.normalize_26(x)

        x = self.relu(self.cnn512_deact_2(x)+res)

        x = self.normalize_27(x)

        x = self.flatten(x)

        x = self.dense_1(x)

        x = self.dense_2(x)

        x = self.dense_3(x)

        return self.dense_output(x)



model = ResNet(num_classes=y.shape[1])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train,y_train, epochs=80, batch_size=19)
model.evaluate(X_test, y_test)
cm = [[0,0,0,0,0,0,0],

     [0,0,0,0,0,0,0],

     [0,0,0,0,0,0,0],

     [0,0,0,0,0,0,0],

     [0,0,0,0,0,0,0],

     [0,0,0,0,0,0,0],

     [0,0,0,0,0,0,0],]

for y_t, y in zip(model.predict(X_test), y_test):

    cm[np.argmax(y)][np.argmax(y_t)]+=1
"""from keras import applications



base_model = applications.resnet50.ResNet50(weights= 'imagenet',

                                            include_top=False,

                                            input_shape= (75, 100, 3))

x = base_model.output

x = MaxPooling2D()(x)

x = Flatten()(x)

x = Dense(500, activation = 'relu')(x)

x = Dense(250, activation = 'relu')(x)

x = Dense(50, activation = 'relu')(x)

predictions = Dense(1, activation= 'sigmoid')(x)

model = Model(inputs = base_model.input, outputs = predictions)"""
cm
print(cm[0][0]/sum(cm[0]))

print(cm[1][1]/sum(cm[1]))

print(cm[2][2]/sum(cm[2]))

print(cm[3][3]/sum(cm[3]))

print(cm[4][4]/sum(cm[4]))

print(cm[5][5]/sum(cm[5]))

print(cm[6][6]/sum(cm[6]))
print(sum(model.predict(X_test)))

print(sum(y_test))
acc = history.history['acc']

loss = history.history['loss']



fig, ax_l = plt.subplots(1,2, figsize = (10,6))

ax_l[0].plot(acc, 'b', label='Training accuracy')

ax_l[0].set_title('Training accuracy')

ax_l[0].set_xlabel('Epochs')

ax_l[0].set_ylabel('Accuracy')

ax_l[0].legend()

ax_l[0].grid()

ax_l[1].plot(loss, 'b', label='Training loss')

ax_l[1].set_title('Training loss')

ax_l[1].set_xlabel('Epochs')

ax_l[1].set_ylabel('Loss')

ax_l[1].legend()

ax_l[1].grid()
model_trial = Sequential([

    Conv2D(64, input_shape=(75,100,3), kernel_size=(3,3), activation='relu'),

    #BatchNormalization(axis = -1),

    #Conv2D(64, input_shape=(450,600,3), kernel_size=(3,3), strides=2, activation='relu'),

    BatchNormalization(axis = -1),

    #Conv2D(128, kernel_size=(3,3), activation='relu'),

    #BatchNormalization(axis = -1),

    Conv2D(128, kernel_size=(3,3), strides=2, activation='relu'),

    BatchNormalization(axis = -1),

    #Conv2D(256, kernel_size=(3,3), activation='relu'),

    #BatchNormalization(axis = -1),

    Conv2D(256, kernel_size=(3,3), strides=2, activation='relu'),

    BatchNormalization(axis = -1),

    #Conv2D(512, kernel_size=(3,3), activation='relu'),

    #BatchNormalization(axis = -1),

    Conv2D(512, kernel_size=(3,3), strides=2, activation='relu'),

    BatchNormalization(axis = -1),

    Flatten(),

    Dense(128, activation='relu'),

    Dropout(0.5),

    Dense(64, activation='relu'),

    Dense(7, activation = 'softmax')

])



model_trial.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model_trial.fit(X_train,y_train, epochs=20, batch_size=20)
"""acc = history.history['acc']

loss = history.history['loss']



val_acc = history.history['val_acc']

val_loss = history.history['val_loss']



fig, ax_l = plt.subplots(1,2, figsize = (10,6))

ax_l[0].plot(acc, 'b', label='Training accuracy')

ax_l[0].plot(val_acc, 'r', label='Validation accuracy')

ax_l[0].set_title('Training accuracy vs Validation accuracy')

ax_l[0].set_xlabel('Epochs')

ax_l[0].set_ylabel('Accuracy')

ax_l[0].legend()

ax_l[0].grid()

ax_l[1].plot(loss, 'b', label='Training loss')

ax_l[1].plot(val_loss, 'r', label='Validation loss')

ax_l[1].set_title('Training loss vs Validation loss')

ax_l[1].set_xlabel('Epochs')

ax_l[1].set_ylabel('Loss')

ax_l[1].legend()

ax_l[1].grid()"""