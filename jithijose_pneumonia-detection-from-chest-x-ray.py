# Import Required Libraries



import os

import cv2

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten, BatchNormalization

from tensorflow.keras import regularizers

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix



%matplotlib inline

sns.set_style('whitegrid')
# function to read data from the directory and resize images



labels = ['NORMAL', 'PNEUMONIA']

IMG_SIZE = 150



def get_data(img_path):

    data = []

    

    for label in labels:

        path = os.path.join(img_path, label)

        target_label = labels.index(label)

        

        for img in os.listdir(path):

            try:

                img_file = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

                img_resize = cv2.resize(img_file, (IMG_SIZE, IMG_SIZE))

                data.append([img_resize, target_label])

            except Exception as e:

                print(e)

    

    return np.array(data)
# load train, test and validation datasets



train_data = get_data('../input/chest-xray-pneumonia/chest_xray/train')

test_data = get_data('../input/chest-xray-pneumonia/chest_xray/test')

val_data = get_data('../input/chest-xray-pneumonia/chest_xray/val')



# check datasets size



print(train_data.shape)

print(test_data.shape)

print(val_data.shape)
# Check the number of samples in each datasets



def get_counts(data):

    

    lbl = []

    for rec in data:

        if(rec[1] == 1):

            lbl.append('Pneumonia')

        else:

            lbl.append('Normal')

    

    return lbl

    



plt.figure(figsize=(10, 6))

plt.subplot(1, 3, 1)

train_count = get_counts(train_data)

sns.countplot(train_count)

plt.title('Train Data')



plt.subplot(1, 3, 2)

test_count = get_counts(test_data)

sns.countplot(test_count)

plt.title('Test Data')



plt.subplot(1, 3, 3)

val_count = get_counts(val_data)

sns.countplot(test_count)

plt.title('Val Data')



plt.show()
# Display few sample images from both targets



plt.figure(figsize=(12, 5))



plt.subplot(1, 2, 1)

plt.imshow(train_data[0][0], cmap='gray')

plt.title(labels[train_data[0][1]])



plt.subplot(1, 2, 2)

plt.imshow(train_data[-1][0], cmap='gray')

plt.title(labels[train_data[-1][1]])



plt.show()
# feature and target separations



def feature_exctract(data):

    features = []

    targets = []

    

    for feature, target in data:

        features.append(feature)

        targets.append(target)

        

    return features, targets



X_train, y_train = feature_exctract(train_data)

X_test, y_test = feature_exctract(test_data)

X_val, y_val = feature_exctract(val_data)
# data normalization



X_train = np.array(X_train) / 255

X_test = np.array(X_test) / 255

X_val = np.array(X_val) / 255



y_train = np.array(y_train)

y_test = np.array(y_test)

y_val = np.array(y_val)



print(X_train.shape)

print(X_test.shape)

print(X_val.shape)



print(len(y_train))

print(len(y_test))

print(len(y_val))
# resize the data for neural network input



X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

X_test =X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

X_val = X_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1)



print(X_train.shape)

print(X_test.shape)

print(X_val.shape)
# Data augmentation



datagen = ImageDataGenerator(featurewise_center=False,

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.22, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip = True,  # randomly flip images

        vertical_flip=False)



datagen.fit(X_train)
# create model



model = Sequential()



model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=X_train[1].shape))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))



model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))

model.add(Dropout(0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))



model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))

model.add(Dropout(0.2))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))



model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='normal', activation='relu'))

model.add(Dropout(0.2))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))



model.add(Flatten())

model.add(Dense(units=128, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(units=64, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))

model.add(Dropout(0.2))



model.add(Dense(units=1 , activation='sigmoid'))



adam = Adam(lr=0.001)



model.compile(optimizer=adam , loss='binary_crossentropy' , metrics=['accuracy'])



model.summary()
# learning rate reduction and early stopping



learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1,factor=0.3, min_lr=0.000001)



early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),

                    validation_data=datagen.flow(X_val, y_val),

                    epochs=20,

                   callbacks=[learning_rate_reduction, early_stop])
print('Loss of the model: ', model.evaluate(X_test, y_test)[0])

print('Accuracy of the model: ', model.evaluate(X_test, y_test)[1])
# save the model



model.save('pneumonia_detection.h5')
# plot the accuracy and loss metrics



loss = pd.DataFrame(history.history)



loss.plot(figsize=(10, 6))

plt.show()
predictions = model.predict_classes(X_test)

predictions = predictions.reshape(1, -1)[0]
print('Classification Report \n\n', classification_report(y_test, predictions, target_names=['Pneumonia (Class 0)','Normal (Class 1)']))
cm = confusion_matrix(y_test, predictions)

print('Confusion Metrics \n\n', cm)
cm = pd.DataFrame(cm , index = ['0','1'] , columns = ['0','1'])



sns.heatmap(cm, annot=True, cmap= "viridis", fmt='', linewidths=.5, xticklabels = labels,yticklabels = labels)
# Samples predictions



correct_prediction = np.nonzero(predictions == y_test)[0]

incorrect_prediction = np.nonzero(predictions != y_test)[0]


plt.figure(figsize=(10, 15))

k=0

for i in correct_prediction[:6]:

    plt.subplot(3, 2, k+1)

    plt.imshow(X_test[i].reshape(150,150), cmap='gray', interpolation='none')

    plt.title("Predicted Class {},Actual Class {}".format(predictions[i], y_test[i]))

    plt.tight_layout()

    k += 1



plt.show()
plt.figure(figsize=(10, 15))

k=0

for i in incorrect_prediction[:6]:

    plt.subplot(3, 2, k+1)

    plt.imshow(X_test[i].reshape(150,150), cmap='gray')

    plt.title("Predicted Class {},Actual Class {}".format(predictions[i], y_test[i]))

    plt.tight_layout()

    k += 1



#plt.show()