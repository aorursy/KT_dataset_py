import numpy as np

import pandas as pd 

import random as rn



# tensorflow

import tensorflow.random as tfr

import tensorflow.keras as keras

from tensorflow.keras.models import Sequential, load_model

from tensorflow.keras.layers import Dense, Dropout, Flatten

from tensorflow.keras.layers import Conv2D, MaxPool2D, MaxPooling2D, BatchNormalization

from tensorflow.keras import backend as K

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.optimizers import RMSprop, Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint



# Chart

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



import seaborn as sns

import glob



from skimage import color, exposure

from sklearn.metrics import classification_report



import os

import cv2



# Setting the same seed for repeatability



seed = 0



np.random.seed(seed) 

rn.seed(seed)

tfr.set_seed(seed)



# Display graphs in a Jupyter

%matplotlib inline



print("Imported")
data_path = '../input/chest-xray-pneumonia/chest_xray/chest_xray/'

data_path



train_path = data_path + 'train/'

test_path = data_path + 'test/'

val_path = data_path + 'val/'
img_size = 200
def read_data(data_paths):

    for data_path in data_paths:

        labels = ['PNEUMONIA', 'NORMAL']

        images = []

        y = []

        for label in labels:

            curr_path = data_path + label

            for img in os.listdir(curr_path):

                if ('DS' not in img):

                    image_path = os.path.join(curr_path, img)

                    image =  cv2.resize(cv2.imread(image_path), (img_size, img_size))

                    if image is not None:

                        images.append([image, label])

                

    images = np.asarray(images)

    return images
train = read_data([train_path])

test = read_data([val_path, test_path])
for i in range(10):

    np.random.shuffle(train)

    np.random.shuffle(test)
train_df = pd.DataFrame(train, columns=['image', 'label'])

test_df = pd.DataFrame(test, columns = ['image', 'label'])
train_df['label'].head()
plt.figure(figsize=(18, 8))

sns.set_style("darkgrid")



plt.subplot(1,2,1)

sns.countplot(train_df['label'], palette = 'coolwarm')

plt.title('Train data')



plt.subplot(1,2,2)

sns.countplot(test_df['label'], palette = "hls")

plt.title('Test data')



plt.show()
def Show_example_image():

    fig = plt.figure(figsize = (16, 16))

    for idx in range(15):

        plt.subplot(5, 5, idx+1)

        plt.imshow(train_df.iloc[idx]['image'])

        plt.title("{}".format(train_df.iloc[idx]['label']))

        

    plt.tight_layout()

    

Show_example_image()
def lung_condition(label):

    if label == 'NORMAL':

        return 0

    else:

        return 1
def splitdata(data):

    X = []

    y = []

    for i, (val, label) in enumerate(data):

        X.append(val)

        y.append(lung_condition(label))

    return np.array(X), np.array(y)
np.random.shuffle(train)

np.random.shuffle(test)

X_train, y_train = splitdata(train)

X_test, y_test = splitdata(test)
def preprocesing_to_mlp(data):

    data1 = color.rgb2gray(data).reshape(-1, img_size * img_size).astype('float32')

    

    # Data Normalization [0, 1]

    data1 /= 255

    

    return data1
X_train = preprocesing_to_mlp(X_train)

X_test = preprocesing_to_mlp(X_test)
num_pixels = X_train.shape[1] 



# one-hot encoding for target column

y_train = to_categorical(y_train)

y_test = to_categorical(y_test)



num_classes = y_train.shape[1]
def draw_learning_curve(history, keys=['accuracy', 'loss']):

    plt.figure(figsize=(20,8))

    for i, key in enumerate(keys):

        plt.subplot(1, 2, i + 1)

        sns.lineplot(x = history.epoch, y = history.history[key])

        sns.lineplot(x = history.epoch, y = history.history['val_' + key])

        plt.title('Learning Curve')

        plt.ylabel(key.title())

        plt.xlabel('Epoch')

#         plt.ylim(ylim)

        plt.legend(['train', 'test'], loc='best')

    plt.show()
callbacks1 = [ 

    EarlyStopping(monitor = 'loss', patience = 6), 

    ReduceLROnPlateau(monitor = 'loss', patience = 3), 

    ModelCheckpoint('../working/model.best1.hdf5',monitor='loss', save_best_only=True) # saving the best model

]
def get_mlp():

    

    return Sequential([

        #input layer is automatic generation by keras

        

        #hidden layer

        Dense(1024, input_dim = num_pixels, activation='relu'),

        

        #output layer

        Dense(num_classes, activation='softmax')

    ])
model = get_mlp()

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.summary()
learning_history = model.fit(X_train, y_train,

          batch_size = 64, epochs = 40, verbose = 2,

          callbacks = callbacks1,

          validation_data=(X_test, y_test));
model = load_model('model.best1.hdf5')
score = model.evaluate(X_test, y_test, verbose = 0)

print('Test loss: {}%'.format(score[0] * 100))

print('Test accuracy: {}%'.format(score[1] * 100))



print("MLP Error: %.2f%%" % (100 - score[1] * 100))
draw_learning_curve(learning_history)
callbacks2 = [ 

    EarlyStopping(monitor = 'loss', patience = 6), 

    ReduceLROnPlateau(monitor = 'loss', patience = 3), 

    ModelCheckpoint('../working/model.best2.hdf5', monitor='loss' , save_best_only=True) # saving the best model

]
def get_mlpv2():

    

    return Sequential([

        Dense(1024, input_dim=num_pixels, activation='relu'),

        Dropout(0.4),

        Dense(512, activation='relu'),

        Dropout(0.3),

        Dense(128, activation='relu'),

        Dropout(0.3),

        Dense(num_classes, activation='softmax')

    ])
model = get_mlpv2()

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.summary()
learning_history = model.fit(X_train, y_train,

          batch_size = 64, epochs = 100, verbose = 1,

          callbacks = callbacks2,

          validation_data=(X_test, y_test));
model = load_model('model.best2.hdf5')
score = model.evaluate(X_test, y_test, verbose = 0)

print('Test loss: {}%'.format(score[0] * 100))

print('Test accuracy: {}%'.format(score[1] * 100))



print("MLP Error: %.2f%%" % (100 - score[1] * 100))
draw_learning_curve(learning_history)
X_train, y_train = splitdata(train)

X_test, y_test = splitdata(test)
def preprocesing_to_cnn(data):

    data1 = color.rgb2gray(data).reshape(-1, img_size, img_size, 1).astype('float32')

    data1 /= 255

    return data1
X_train = preprocesing_to_cnn(X_train)

X_test = preprocesing_to_cnn(X_test)



# one-hot encoding for target column

y_train = to_categorical(y_train)

y_test = to_categorical(y_test)
num_classes = y_train.shape[1]
input_shape = (img_size, img_size, 1)
callbacks3 = [ 

    EarlyStopping(monitor = 'loss', patience = 6), 

    ReduceLROnPlateau(monitor = 'loss', patience = 3), 

    ModelCheckpoint('../working/model.best3.hdf5', monitor='loss' , save_best_only=True) # saving the best model

]
num_pixels 
def get_modelcnn():

    return Sequential([

        

        Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', input_shape = input_shape),

        Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'),

        BatchNormalization(),

        MaxPool2D(pool_size=(2, 2)),

        Dropout(0.25),

        

        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),

        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),

        BatchNormalization(),

        MaxPool2D(pool_size=(2, 2)),

        Dropout(0.25),

        

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),

        BatchNormalization(),

        MaxPool2D(pool_size=(2, 2)),

        Dropout(0.25),

        

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same' ),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),

        BatchNormalization(),

        MaxPool2D(pool_size=(2, 2)),

        Dropout(0.25),

        

        Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same' ),

        Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),

        BatchNormalization(),

        MaxPool2D(pool_size=(2, 2)),

        Dropout(0.25),

        

        

        Flatten(),

        

        Dense(512, activation='relu'),

        Dropout(0.5),

        

        Dense(256, activation='relu'),

        Dropout(0.5),

        

        Dense(64, activation='relu'),

        Dropout(0.5),

        Dense(num_classes, activation = "softmax")

        

    ])
model = get_modelcnn()

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

model.summary()
learning_history = model.fit(X_train, y_train,

          batch_size = 64,

          epochs = 100,

          verbose = 1,

          callbacks = callbacks3,

          validation_data = (X_test, y_test))
model = load_model('model.best3.hdf5')
score = model.evaluate(X_test, y_test, verbose = 0)

print('Test loss: {}%'.format(score[0] * 100))

print('Test accuracy: {}%'.format(score[1] * 100))



print("MLP Error: %.2f%%" % (100 - score[1] * 100))
draw_learning_curve(learning_history)
datagen = ImageDataGenerator(

        featurewise_center = False,

        samplewise_center = False,

        featurewise_std_normalization = False, 

        samplewise_std_normalization = False,

        zca_whitening = False,

        horizontal_flip = False,

        vertical_flip = False,

        rotation_range = 10,  

        zoom_range = 0.1, 

        width_shift_range = 0.1, 

        height_shift_range = 0.1)



datagen.fit(X_train)

train_gen = datagen.flow(X_train, y_train, batch_size = 32)
callbacks4 = [ 

    EarlyStopping(monitor = 'loss', patience = 7), 

    ReduceLROnPlateau(monitor = 'loss', patience = 4), 

    ModelCheckpoint('../working/model.best4.hdf5', monitor='loss' , save_best_only=True) # saving the best model

]
def get_modelcnn_v2():

    return Sequential([

        Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', input_shape = input_shape),

        Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'),

        BatchNormalization(),

        MaxPool2D(pool_size=(2, 2)),

        Dropout(0.2),

        

        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),

        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),

        BatchNormalization(),

        MaxPool2D(pool_size=(2, 2)),

        Dropout(0.2),

        

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),

        BatchNormalization(),

        MaxPool2D(pool_size=(2, 2)),

        Dropout(0.2),

        

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),

        BatchNormalization(),

        MaxPool2D(pool_size=(2, 2)),

        Dropout(0.2),

        

        Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),

        Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),

        BatchNormalization(),

        MaxPool2D(pool_size=(2, 2)),

        Dropout(0.2),

        

        Flatten(),

       

        Dense(1024, activation='relu'),

        BatchNormalization(),

        Dropout(0.5),

        

        Dense(512, activation='relu'),

        BatchNormalization(),

        Dropout(0.4),

        

        Dense(256, activation='relu'),

        BatchNormalization(),

        Dropout(0.3),

        

        Dense(64, activation='relu'),

        BatchNormalization(),

        Dropout(0.2),

        

        Dense(num_classes, activation = "softmax")

        

    ])
model = get_modelcnn_v2()

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

model.summary()
learning_history = model.fit_generator((train_gen), 

                               epochs = 100, 

                               steps_per_epoch = X_train.shape[0] // 32,

                               validation_data = (X_test, y_test),

                               callbacks = callbacks4,

                        )
model = load_model('model.best4.hdf5')
score = model.evaluate(X_test, y_test, verbose = 0)

print('Test loss: {}%'.format(score[0] * 100))

print('Test accuracy: {}%'.format(score[1] * 100))



print("MLP Error: %.2f%%" % (100 - score[1] * 100))
draw_learning_curve(learning_history)
y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred, axis = 1)
y_pre_test = np.argmax(y_test, axis = 1)
def show_condition(num):

    if num == 0:

        return 'NORMAL'

    return 'PNEUMONIA'
cnt_error = []

for idx, (a, b) in enumerate(zip(y_pre_test, y_pred)):

    if a == b: continue

    cnt_error.append(a)# test



cnt_error = np.unique(cnt_error, return_counts = True)

sns.set_style("darkgrid")

plt.figure(figsize = (15, 7))

sns.barplot([show_condition(x) for x in cnt_error[0]], cnt_error[1], palette="muted")

plt.show()
cnt_ind = 1

list_idx = []

fig = plt.figure(figsize=(14, 14))

X_test_plot = X_test.reshape(-1, img_size, img_size)

for idx, (a, b) in enumerate(zip(y_pre_test, y_pred)):

    if(cnt_ind > 16):break

    if a == b: continue

    plt.subplot(4, 4, cnt_ind)

    plt.imshow(X_test_plot[idx], cmap='gray', interpolation='none')

    plt.title('y_true = {0}\ny_pred = {1}\n ind = {2}'.format(show_condition(a), show_condition(b), idx))

    plt.tight_layout()

    list_idx.append(idx)

    cnt_ind += 1
print(classification_report(y_pre_test, y_pred))