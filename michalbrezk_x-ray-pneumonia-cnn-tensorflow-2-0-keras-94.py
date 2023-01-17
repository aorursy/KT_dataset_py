import glob

import random as rn

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import cv2



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import RMSprop, Adam

from tensorflow.keras.utils import to_categorical

from sklearn.metrics import classification_report, confusion_matrix





%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
path = '/kaggle/input/chest-xray-pneumonia/chest_xray/'





# define paths

train_normal_dir = path + 'train/NORMAL/'

train_pneu_dir = path + 'train/PNEUMONIA/'



test_normal_dir = path + 'test/NORMAL/'

test_pneu_dir = path + 'test/PNEUMONIA/'



val_normal_dir = path + 'val/NORMAL/'

val_pneu_dir = path + 'val/PNEUMONIA/'





# find all files, our files has extension jpeg

train_normal_cases = glob.glob(train_normal_dir + '*jpeg')

train_pneu_cases = glob.glob(train_pneu_dir + '*jpeg')



test_normal_cases = glob.glob(test_normal_dir + '*jpeg')

test_pneu_cases = glob.glob(test_pneu_dir + '*jpeg')



val_normal_cases = glob.glob(val_normal_dir + '*jpeg')

val_pneu_cases = glob.glob(val_pneu_dir + '*jpeg')





# make path using / instead of \\ ... this may be redudant step

train_normal_cases = [x.replace('\\', '/') for x in train_normal_cases]

train_pneu_cases = [x.replace('\\', '/') for x in train_pneu_cases]

test_normal_cases = [x.replace('\\', '/') for x in test_normal_cases]

test_pneu_cases = [x.replace('\\', '/') for x in test_pneu_cases]

val_normal_cases = [x.replace('\\', '/') for x in val_normal_cases]

val_pneu_cases = [x.replace('\\', '/') for x in val_pneu_cases]





# create lists for train, test & validation cases, create labels as well

train_list = []

test_list = []

val_list = []



for x in train_normal_cases:

    train_list.append([x, 0])

    

for x in train_pneu_cases:

    train_list.append([x, 1])

    

for x in test_normal_cases:

    test_list.append([x, 0])

    

for x in test_pneu_cases:

    test_list.append([x, 1])

    

for x in val_normal_cases:

    val_list.append([x, 0])

    

for x in val_pneu_cases:

    val_list.append([x, 1])





# shuffle/randomize data as they were loaded in order: normal cases, then pneumonia cases

rn.shuffle(train_list)

rn.shuffle(test_list)

rn.shuffle(val_list)





# create dataframes

train_df = pd.DataFrame(train_list, columns=['image', 'label'])

test_df = pd.DataFrame(test_list, columns=['image', 'label'])

val_df = pd.DataFrame(val_list, columns=['image', 'label'])
plt.figure(figsize=(20,5))



plt.subplot(1,3,1)

sns.countplot(train_df['label'])

plt.title('Train data')



plt.subplot(1,3,2)

sns.countplot(test_df['label'])

plt.title('Test data')



plt.subplot(1,3,3)

sns.countplot(val_df['label'])

plt.title('Validation data')



plt.show()
plt.figure(figsize=(20,8))

for i,img_path in enumerate(train_df[train_df['label'] == 1][0:4]['image']):

    plt.subplot(2,4,i+1)

    plt.axis('off')

    img = plt.imread(img_path)

    plt.imshow(img, cmap='gray')

    plt.title('Pneumonia')

    

for i,img_path in enumerate(train_df[train_df['label'] == 0][0:4]['image']):

    plt.subplot(2,4,4+i+1)

    plt.axis('off')

    img = plt.imread(img_path)

    plt.imshow(img, cmap='gray')

    plt.title('Healthy / Normal')
def process_data(img_path):

    img = cv2.imread(img_path)

    img = cv2.resize(img, (196, 196))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img/255.0

    img = np.reshape(img, (196,196,1))

    

    return img



def compose_dataset(df):

    data = []

    labels = []



    for img_path, label in df.values:

        data.append(process_data(img_path))

        labels.append(label)

        

    return np.array(data), np.array(labels)
X_train, y_train = compose_dataset(train_df)

X_test, y_test = compose_dataset(test_df)

X_val, y_val = compose_dataset(val_df)



print('Train data shape: {}, Labels shape: {}'.format(X_train.shape, y_train.shape))

print('Test data shape: {}, Labels shape: {}'.format(X_test.shape, y_test.shape))

print('Validation data shape: {}, Labels shape: {}'.format(X_val.shape, y_val.shape))
# define generator

datagen = ImageDataGenerator(

    rotation_range=10,

    zoom_range = 0.1,

    width_shift_range=0.1,

    height_shift_range=0.1,

    horizontal_flip=False,

    vertical_flip=False

)



# fit generator on our train features

datagen.fit(X_train)
y_train = to_categorical(y_train)

y_test = to_categorical(y_test)

y_val = to_categorical(y_val)
model = Sequential()



model.add(Conv2D(filters=8, kernel_size=(7,7), padding='same', activation='relu', input_shape=(196, 196, 1)))

model.add(Conv2D(filters=8, kernel_size=(7,7), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(3,3)))



model.add(Conv2D(filters=16, kernel_size=(5,5), padding='same', activation='relu'))

model.add(Conv2D(filters=16, kernel_size=(5,5), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(3,3)))



model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'))

model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Flatten())



model.add(Dense(128, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))



optimizer = Adam(lr=0.0001, decay=1e-5)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])



callback = EarlyStopping(monitor='loss', patience=6)

history = model.fit(datagen.flow(X_train,y_train, batch_size=4), validation_data=(X_test, y_test), epochs = 100, verbose = 1, callbacks=[callback], class_weight={0:6.0, 1:0.5})
plt.figure(figsize=(20,5))



# plot loss & val loss

plt.subplot(1,2,1)

sns.lineplot(x=history.epoch, y=history.history['loss'], color='red', label='Loss')

sns.lineplot(x=history.epoch, y=history.history['val_loss'], color='orange', label='Val Loss')

plt.title('Loss on train vs test')

plt.legend(loc='best')



# plot accuracy and val accuracy

plt.subplot(1,2,2)

sns.lineplot(x=history.epoch, y=history.history['accuracy'], color='blue', label='Accuracy')

sns.lineplot(x=history.epoch, y=history.history['val_accuracy'], color='green', label='Val Accuracy')

plt.title('Accuracy on train vs test')

plt.legend(loc='best')



plt.show()
y_test_hat = model.predict(X_test, batch_size=4)

y_test_hat = np.argmax(y_test_hat, axis=1)

y_test = np.argmax(y_test, axis=1)
# calculate confusion matrix & classification report

conf_m = confusion_matrix(y_test, y_test_hat)

clas_r = classification_report(y_test, y_test_hat)



# plot confusion matrix as heatmap

plt.figure(figsize=(5,3))

sns.set(font_scale=1.2)

ax = sns.heatmap(conf_m, annot=True,xticklabels=['H', 'P'], yticklabels=['H', 'P'], cbar=False, cmap='Blues',linewidths=1, linecolor='black', fmt='.0f')

plt.yticks(rotation=0)

plt.xlabel('Predicted labels')

plt.ylabel('True labels')

ax.xaxis.set_ticks_position('top') 

plt.title('Confusion matrix - test data\n(H - healthy/normal, P - pneumonia)')

plt.show()



# print classification report

print('Classification report on test data')

print(clas_r)
y_val_hat = model.predict(X_val, batch_size=4)

y_val_hat = np.argmax(y_val_hat, axis=1)

y_val = np.argmax(y_val, axis=1)
plt.figure(figsize=(20,15))

for i,x in enumerate(X_val):

    plt.subplot(4,4,i+1)

    plt.imshow(x.reshape(196, 196), cmap='gray')

    plt.axis('off')

    plt.title('Predicted: {}, Real: {}'.format(y_val_hat[i], y_val[i]))  