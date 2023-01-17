import numpy as np

import pandas as pd



np.random.seed(0) 

import random



import tensorflow.keras as keras

from tensorflow.keras.models import Sequential, load_model

from tensorflow.keras.layers import Dense, Dropout, Flatten

from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization

from tensorflow.keras import backend as K

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.datasets import mnist

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint



from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")





X_train = train.drop(labels = ["label"], axis = 1)

y_train = train['label']



X_test = test



print(X_train.shape, X_test.shape)
X_train_plot = X_train.values.reshape(-1, 28, 28)
def Show_example_digits(mono = 'gray'):

    fig = plt.figure(figsize = (16, 16))

    for idx in range(15):

        plt.subplot(5, 5,idx+1)

        plt.imshow(X_train_plot[idx], cmap = mono)

        plt.title("Digit {}".format(y_train[idx]))

        

    plt.tight_layout()

    

Show_example_digits()
# Function return digit in grayscale

def plot_digit(digit, dem = 28, font_size = 12):

    max_ax = font_size * dem

    

    fig = plt.figure(figsize=(13, 13))

    plt.xlim([0, max_ax])

    plt.ylim([0, max_ax])

    plt.axis('off')

    black = '#000000'

    

    for idx in range(dem):

        for jdx in range(dem):



            t = plt.text(idx * font_size, max_ax - jdx*font_size, digit[jdx][idx], fontsize = font_size, color = black)

            c = digit[jdx][idx] / 255.

            t.set_bbox(dict(facecolor=(c, c, c), alpha = 0.5, edgecolor = 'black'))

            

    plt.show()
rand_number = random.randint(0, len(y_train))

print(y_train[rand_number])

plot_digit(X_train_plot[rand_number])
digit_range = np.arange(10)



val = y_train.value_counts().index

cnt = y_train.value_counts().values

mycolors = ['red', 'blue', 'green', 'orange', 'brown', 'grey', 'pink', 'olive', 'deeppink', 'steelblue']



plt.figure(figsize = (15, 7))

plt.title("The number of digits in the data", fontsize = 20)

plt.xticks(range(10))

plt.bar(val, cnt, color = mycolors);
img_rows, img_cols = 28, 28



num_pixels = X_train.shape[1] 



input_shape = (img_rows, img_cols)
# Data Normalization [0, 1]

X_train /= 255

X_test /= 255



# one-hot encoding for target column

y_train = to_categorical(y_train)



# | [0, 1, 2, ... , 9] | = 10

num_classes = y_train.shape[1]



# Number of objects, vector size (28 * 28)

print(X_train.shape, X_test.shape)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state = 2)
def draw_learning_curve(history, key='accuracy', ylim=(0.8, 1.01)):

    plt.figure(figsize=(12,6))

    plt.plot(history.history[key])

    plt.plot(history.history['val_' + key])

    plt.title('Learning Curve')

    plt.ylabel(key.title())

    plt.xlabel('Epoch')

    plt.ylim(ylim)

    plt.legend(['train', 'test'], loc='best')

    plt.show()
def get_mlp():

    

    return Sequential([

        #input layer is automatic generation by keras

        

        #hidden layer

        Dense(512, input_dim = num_pixels, activation='relu'),

        

        #output layer

        Dense(num_classes, activation='softmax')

    ])
model = get_mlp()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
X_train.shape
learning_history = model.fit(X_train, y_train,

          batch_size = 1024, epochs = 40, verbose = 2,

          validation_data=(X_val, y_val));
score = model.evaluate(X_val, y_val, verbose = 0)

print('Test loss: {}%'.format(score[0] * 100))

print('Test accuracy: {}%'.format(score[1] * 100))



print("MLP Error: %.2f%%" % (100 - score[1] * 100))
draw_learning_curve(learning_history, 'accuracy', ylim = (0.95, 1.001))
def get_mlpv2():

    

    return Sequential([

        Dense(512, input_dim=num_pixels, activation='relu'),

        Dropout(0.3),

        Dense(256, activation='relu'),

        Dropout(0.2),

        Dense(128, activation='relu'),

        Dense(num_classes, kernel_initializer='normal', activation='softmax')

    ])
model = get_mlpv2()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
learning_history = model.fit(X_train, y_train,

          batch_size = 1024, epochs = 40, verbose = 2,

          validation_data=(X_val, y_val));
draw_learning_curve(learning_history, 'accuracy', ylim = (0.97,1.))
score = model.evaluate(X_val, y_val, verbose = 0)

print('Test loss: {}%'.format(score[0] * 100))

print('Test accuracy: {}%'.format(score[1] * 100))



print("MLP Error: %.2f%%" % (100 - score[1] * 100))
X_train.shape
X_train = X_train.values.reshape(-1, 28, 28, 1)

X_val = X_val.values.reshape(-1, 28, 28, 1)

X_test = X_test.values.reshape(-1, 28, 28, 1)

input_shape = (28, 28, 1)
def get_triplecnn():

    return Sequential([

        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape = input_shape),

        Conv2D(32, kernel_size=(3, 3), activation='relu' ),

        MaxPool2D(pool_size=(2, 2)),

        Dropout(0.25),

        

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),

        Conv2D(64, kernel_size=(3, 3), activation='relu' ),

        MaxPool2D(pool_size=(2, 2)),

        Dropout(0.25),

        

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same' ),

        Conv2D(128, kernel_size=(3, 3), activation='relu' ),

        MaxPool2D(pool_size=(2, 2)),

        Dropout(0.25),

        

        

        Flatten(),

        

        Dense(256, activation='relu'),

        Dropout(0.5),

        Dense(num_classes, activation = "softmax")

        

    ])
model = get_triplecnn()

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

model.summary()
learning_history = model.fit(X_train, y_train,

          batch_size = 128,

          epochs = 50,

          verbose = 1,

          validation_data = (X_val, y_val))
score = model.evaluate(X_val, y_val, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])



print("CNN Error: %.2f%%" % (100-score[1]*100))
draw_learning_curve(learning_history, 'accuracy', ylim = (0.98,1.))
y_pred = model.predict(X_val)
def draw_output(idx_nums):

    plt.figure(figsize = (20, 20))

    plt.xticks( range(10) )

    x = np.ceil(np.sqrt(len(idx_nums)))

    cnt = 1

    for ph in idx_nums:

        plt.subplot(x, x, cnt)

        curr_photo = y_val[ph]

        

        plt.xlim(0, 10)

        plt.title("Digit: {0}\n idx: {1} ".format(np.argmax(y_val[ph]), ph), fontsize = 10) 

        plt.bar(range(10), y_pred[ph])

        

        cnt += 1
cnt_error = []

for idx, (a, b) in enumerate(zip(y_val, y_pred)):

    if np.argmax(a) == np.argmax(b): continue

    cnt_error.append( (np.argmax(a)) )



cnt_error = np.unique(cnt_error, return_counts = True)

sns.set_style("darkgrid")

plt.figure(figsize = (15, 7))

bar_plot = sns.barplot(cnt_error[0], cnt_error[1], palette="muted")

plt.show()
cnt_ind = 1

list_idx = []

X_val_plot = X_val.reshape( X_val.shape[:-1] )

fig = plt.figure(figsize=(14, 14))



for idx, (a, b) in enumerate(zip(y_val, y_pred)):

    if np.argmax(a) == np.argmax(b): continue

    if (np.argmax(a) == 2 or np.argmax(a) == 9):    

        plt.subplot(5, 5, cnt_ind)

        plt.imshow(X_val_plot[idx], cmap='gray', interpolation='none')

        plt.title('y_true={0}\ny_pred={1}\n ind = {2}'.format(np.argmax(a), np.argmax(b), idx))

        plt.tight_layout()

        list_idx.append(idx)

        cnt_ind += 1
draw_output(list_idx)
train_aug = ImageDataGenerator(

        featurewise_center = False,

        samplewise_center = False,

        featurewise_std_normalization = False, 

        samplewise_std_normalization = False,

        zca_whitening = False,

        horizontal_flip = False,

        vertical_flip = False,

        fill_mode = 'nearest',

        rotation_range = 10,  

        zoom_range = 0.1, 

        width_shift_range = 0.1, 

        height_shift_range = 0.1)

        



train_aug.fit(X_train)

train_gen = train_aug.flow(X_train, y_train, batch_size=64)
def get_newtriplecnn():

    return Sequential([

        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape = input_shape),

        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),

        BatchNormalization(),

        MaxPool2D(pool_size=(2, 2)),

        Dropout(0.25),

        

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same' ),

        BatchNormalization(),

        MaxPool2D(pool_size=(2, 2)),

        Dropout(0.25),

        

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same' ),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same' ),

        BatchNormalization(),

        MaxPool2D(pool_size=(2, 2)),

        Dropout(0.25),

        

        

        Flatten(),

          

        Dense(512, activation='relu'),

        BatchNormalization(),

        Dropout(0.5),

        

        Dense(256, activation='relu'),

        BatchNormalization(),

        Dropout(0.4),

        

        Dense(64, activation='relu'),

        BatchNormalization(),

        Dropout(0.3),

        

        Dense(num_classes, activation = "softmax")

        

    ])
model = get_newtriplecnn()

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

model.summary()
callbacks1 = [ 

    EarlyStopping(monitor = 'loss', patience = 6), 

    ReduceLROnPlateau(monitor = 'loss', patience = 3), 

    ModelCheckpoint('../working/model.best.hdf5', save_best_only=True) # saving the best model

]
history = model.fit_generator((train_gen), epochs = 100, 

                               steps_per_epoch = X_train.shape[0] // 64,

                               validation_data = (X_val, y_val),

                               callbacks = callbacks1,

                             )
model = load_model('../working/model.best.hdf5')
score = model.evaluate(X_val, y_val, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])



print("CNN Error: %.2f%%" % (100-score[1]*100))
draw_learning_curve(history, 'accuracy', ylim = (0.985,1.))
output = model.predict(X_test)



output = np.argmax(output, axis = 1)



output = pd.Series(output, name="Label")



submission = pd.concat([pd.Series(range(1,28001), name = "ImageId"), output], axis = 1)



submission.to_csv("submission.csv", index=False)
def load_data(path):

    with np.load(path) as f:

        x_train, y_train = f['x_train'], f['y_train']

        x_test, y_test = f['x_test'], f['y_test']

        return (x_train, y_train), (x_test, y_test)



(x_train1, y_train1), (x_test1, y_test1) = load_data('../input/mnist-numpy/mnist.npz')
x_train1 = x_train1 / 255

x_test1 = x_test1 / 255



x_train1 = x_train1.reshape(-1, 28, 28, 1)

x_test1 = x_test1.reshape(-1, 28, 28, 1)



y_train1 = y_train1.reshape(y_train1.shape[0], 1)

y_test1 = y_test1.reshape(y_test1.shape[0], 1)
Add_X = np.vstack((x_train1, x_test1))



Add_y = np.vstack((y_train1, y_test1))



Add_y = to_categorical(Add_y)
train = pd.read_csv("../input/digit-recognizer/train.csv")



X_train = train.drop(labels = ["label"], axis = 1)

y_train = train['label']

y_train = to_categorical(y_train)



X_train /= 255

X_train = X_train.values.reshape(-1, 28, 28, 1)
add_train_aug = ImageDataGenerator(

        featurewise_center = False,

        samplewise_center = False,

        featurewise_std_normalization = False, 

        samplewise_std_normalization = False,

        zca_whitening = False,

        horizontal_flip = False,

        vertical_flip = False,

        fill_mode = 'nearest',

        rotation_range = 10,  

        zoom_range = 0.1, 

        width_shift_range = 0.1, 

        height_shift_range = 0.1)

        



add_train_aug.fit(Add_X)

add_train_gen = add_train_aug.flow(Add_X, Add_y, batch_size=64)
add_callbacks = [ 

    EarlyStopping(monitor = 'loss', patience = 6), 

    ReduceLROnPlateau(monitor = 'loss', patience = 3), 

    ModelCheckpoint('../working/additional_model.best.hdf5', save_best_only=True) # saving the best model

]
def get_addcnn():

    return Sequential([

        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape = input_shape),

        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),

        BatchNormalization(),

        MaxPool2D(pool_size=(2, 2)),

        Dropout(0.25),

        

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same' ),

        BatchNormalization(),

        MaxPool2D(pool_size=(2, 2)),

        Dropout(0.25),

        

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same' ),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same' ),

        BatchNormalization(),

        MaxPool2D(pool_size=(2, 2)),

        Dropout(0.25),

        

        

        Flatten(),

          

        Dense(512, activation='relu'),

        BatchNormalization(),

        Dropout(0.5),

        

        Dense(256, activation='relu'),

        BatchNormalization(),

        Dropout(0.4),

        

        Dense(64, activation='relu'),

        BatchNormalization(),

        Dropout(0.3),

        

        Dense(num_classes, activation = "softmax")

        

    ])
model = get_addcnn()

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

model.summary()
history = model.fit_generator((add_train_gen), epochs = 100, 

                               steps_per_epoch = x_train1.shape[0] // 64,

                               validation_data = (X_val, y_val),

                               callbacks = add_callbacks,

                             )
model = load_model('additional_model.best.hdf5')
score = model.evaluate(X_val, y_val, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])



print("CNN Error: %.2f%%" % (100-score[1]*100))
draw_learning_curve(history, 'accuracy', ylim = (0.985,1.))
output = model.predict(X_test)



output = np.argmax(output, axis = 1)



output = pd.Series(output, name="Label")



submission = pd.concat([pd.Series(range(1,28001), name = "ImageId"), output], axis = 1)



submission.to_csv("bonus_submission.csv", index=False)