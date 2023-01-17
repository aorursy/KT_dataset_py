#load data directories

dir='/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/'



test='/kaggle/input/chest-xray-pneumonia/chest_xray/test'

train='/kaggle/input/chest-xray-pneumonia/chest_xray/train'

val='/kaggle/input/chest-xray-pneumonia/chest_xray/val'



norm_test='/kaggle/input/chest-xray-pneumonia/chest_xray/test/NORMAL'

pneum_test='/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA'



norm_train='/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL'

pneum_train='/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA'



norm_val='/kaggle/input/chest-xray-pneumonia/chest_xray/val/NORMAL'

pneum_val='/kaggle/input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA'
import pandas as pd

from pathlib import Path
#Load train dataset

train_data_norm=pd.DataFrame(Path(norm_train).glob('*.jpeg'))

train_data_pneum=pd.DataFrame(Path(pneum_train).glob('*.jpeg'))



train_data_norm[1]=0

train_data_pneum[1]=1



train_data=train_data_norm.append(train_data_pneum)
#Load test dataset

test_data_norm=pd.DataFrame(Path(norm_test).glob('*.jpeg'))

test_data_pneum=pd.DataFrame(Path(pneum_test).glob('*.jpeg'))



test_data_norm[1]=0

test_data_pneum[1]=1



test_data=test_data_norm.append(test_data_pneum)
#Load validation dataset

val_data_norm=pd.DataFrame(Path(norm_val).glob('*.jpeg'))

val_data_pneum=pd.DataFrame(Path(pneum_val).glob('*.jpeg'))



val_data_norm[1]=0

val_data_pneum[1]=1



val_data=val_data_norm.append(val_data_pneum)
#Let's explore the data

count_tr=len(train_data)

count_tr_n=len(train_data[train_data[1]==0])

count_tr_p=len(train_data[train_data[1]==1])



count_ts=len(test_data)

count_ts_n=len(test_data[test_data[1]==0])

count_ts_p=len(test_data[test_data[1]==1])



count_val=len(val_data)

count_val_n=len(val_data[val_data[1]==0])

count_val_p=len(val_data[val_data[1]==1])
print('Train data')



print(f'Normal cases    {count_tr_n}   ({round(count_tr_n/count_tr,2)*100}%)')

print(f'Pneunomia cases {count_tr_p}   ({round(count_tr_p/count_tr,2)*100}%)')

print(f'Total cases:    {count_tr} ')

print('')

print('Test data')

print(f'Normal cases    {count_ts_n}   ({round(count_ts_n/count_ts,2)*100}%)')

print(f'Pneunomia cases {count_ts_p}   ({round(count_ts_p/count_ts,2)*100}%)')

print(f'Total cases:    {count_ts} ')

print('')

print('Validation data')

print(f'Normal cases    {count_val_n}   ({round(count_val_n/count_val,2)*100}%)')

print(f'Pneunomia cases {count_val_p}   ({round(count_val_p/count_val,2)*100}%)')

print(f'Total cases:    {count_val} ')
# Let's shuffle the data

from sklearn.utils import shuffle
train_data =shuffle(train_data)

test_data = shuffle(test_data)

val_data = shuffle(val_data)
import cv2

import numpy as np
#loading train data

train_img = []

train_label = []

train_shapes =[]

blacwhite_counter=0

for i, imgfile in enumerate(train_data[0]):

    img = cv2.imread(str(imgfile))

    train_shapes.append(np.shape(img))

    img = cv2.resize(img, (224,224))

    if img.shape[2] ==1:

        img = np.dstack([img, img, img])

        blacwhite_counter= blacwhite_counter+1

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)/255.

    train_img.append(img)

    train_label.append(train_data.iloc[i,1])

#loading test data

test_img = []

test_label = []

test_shapes=[]

blacwhite_counter_t=0

for i, imgfile in enumerate(test_data[0]):

    img = cv2.imread(str(imgfile))

    test_shapes.append(np.shape(img))

    img = cv2.resize(img, (224,224))

    if img.shape[2] ==1:

        img = np.dstack([img, img, img])

        blacwhite_counter_t= blacwhite_counter_t+1

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)/255.

    test_img.append(img)

    test_label.append(test_data.iloc[i,1])
#loading val data

val_img = []

val_label = []

val_shapes=[]

blacwhite_counter_v=0

for i, imgfile in enumerate(val_data[0]):

    img = cv2.imread(str(imgfile))

    val_shapes.append(np.shape(img))

    img = cv2.resize(img, (224,224))

    if img.shape[2] ==1:

      img = np.dstack([img, img, img])

      blacwhite_counter_v= blacwhite_counter_v+1

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)/255.

    val_img.append(img)

    val_label.append(val_data.iloc[i,1])

print(blacwhite_counter, blacwhite_counter_t,blacwhite_counter_v)
from keras.utils import to_categorical
#I rename the datasets with easier names and turn the labels into categorical data

trainX = np.array(train_img)

trainY = to_categorical(np.array(train_label))

valX = np.array(val_img)

valY = to_categorical(np.array(val_label))

testX = np.array(test_img)

testY = to_categorical(np.array(test_label))
# You can delete the old variables to clean some ram

del train_img, train_label, val_img, val_label, test_img, test_label
from keras.models import Sequential, Model

from keras.applications.vgg16 import VGG16, preprocess_input

from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D

from keras.layers import GlobalMaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras.layers.merge import Concatenate

from keras.optimizers import Adam, SGD, RMSprop

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping



from keras.layers import GaussianNoise

from keras.layers import Activation

import keras.metrics

from sklearn.metrics import precision_score , recall_score



from mlxtend.plotting import plot_confusion_matrix

from sklearn.metrics import confusion_matrix



from sklearn.metrics import classification_report



import matplotlib.pyplot as plt
# construct the training image generator for data augmentation

aug = ImageDataGenerator(#zoom_range=[0.9,1.1],

                         brightness_range=[0.9,1.1]

                         #horizontal_flip=True ,

                         #rotation_range=90,

                         #height_shift_range=0.15,

                         #width_shift_range=[-0.125,0.125]

                         )



aug.fit(trainX)
for X_batch, y_batch in aug.flow(trainX, trainY, batch_size=9):

	# create a grid of 3x3 images

	for i in range(0, 9):

		plt.subplot(330 + 1 + i)

		plt.imshow(X_batch[i])

	# show the plot

	plt.show()

	break
# construct the training image generator for data augmentation

aug = ImageDataGenerator(#zoom_range=[0.9,1.1],

                         #brightness_range=[0.,1.]

                         #horizontal_flip=True ,

                         #rotation_range=90,

                         #height_shift_range=0.15,

                         #width_shift_range=[-0.125,0.125]

                         )



aug.fit(trainX)
for X_batch, y_batch in aug.flow(trainX, trainY, batch_size=9):

	# create a grid of 3x3 images

	for i in range(0, 9):

		plt.subplot(330 + 1 + i)

		plt.imshow(X_batch[i])

	# show the plot

	plt.show()

	break
EPOCHS =50

BS = 64



def build_model():

  input_img=Input(shape=(224,224,3), name='ImageInput')



  x = Conv2D(16, (3,3),activation='relu', padding='same', name='Conv1_1')(input_img)

  x = Conv2D(16, (3,3), activation='relu', padding='same', name='Conv1_2')(x)

  

  x = MaxPooling2D((2,2), name='pool1')(x)

  x = Conv2D(32, (3,3),activation='relu', padding='same', name='Conv2_1')(x)

  x = Conv2D(32, (3,3), activation='relu', padding='same', name='Conv2_2')(x)

  x = MaxPooling2D((2,2), name='pool2')(x)



  x = Flatten(name='flatten')(x)

  x = Dense(128, activation='relu', name='fc1')(x)

  x = Dense(64, activation='relu', name='fc2')(x)

  x = Dense(2, activation='softmax', name='fc3')(x)



  model = Model(inputs = input_img, output=x)

  return model
model =  build_model()

model.summary()
opt = Adam(lr=0.0001, decay=1e-5)

chkpt = ModelCheckpoint(filepath='best_aug_model_todate2loss.h5',monitor='val_loss', save_best_only=True, save_weights_only=True)

chkpt2 = ModelCheckpoint(filepath='best_aug_model_todate2acc.h5',monitor='val_accuracy', save_best_only=True, save_weights_only=True)

callbacks_list=[chkpt,chkpt2]

model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer=opt)
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),callbacks=callbacks_list,

	validation_data=(valX, valY), steps_per_epoch=len(trainX) // BS,

	epochs=EPOCHS)
print(H.history.keys())

#  "Accuracy"

plt.plot(H.history['accuracy'])

plt.plot(H.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()

# "Loss"

plt.plot(H.history['loss'])

plt.plot(H.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
model.save_weights("last_epoch.h5")
test_loss_last, test_score_last = model.evaluate(testX, testY, batch_size=32)
print("Loss on test set: ", test_loss_last)

print("Accuracy on test set: ", test_score_last)
model.load_weights("best_aug_model_todate2loss.h5")
test_loss_bestloss, test_score_bestloss = model.evaluate(testX, testY, batch_size=32)
print("Loss on test set: ", test_loss_bestloss)

print("Accuracy on test set: ", test_score_bestloss)
model.load_weights("best_aug_model_todate2acc.h5")
test_loss_bestacc, test_score_bestacc = model.evaluate(testX, testY, batch_size=32)
print("Loss on test set: ", test_loss_bestacc)

print("Accuracy on test set: ", test_score_bestacc)
max_model=np.argmax([test_score_last,test_score_bestloss,test_score_bestacc])

max_model
if max_model == 0:

    model.load_weights("last_epoch.h5")

elif max_model == 1:

    model.load_weights("best_aug_model_todate2loss.h5")

elif max_model == 2:

    model.load_weights("best_aug_model_todate2acc.h5") 

    

    

    
# Get predictions

preds = model.predict(testX, batch_size=16)

preds = np.argmax(preds, axis=-1)



# Original labels

orig_test_labels = np.argmax(testY, axis=-1)



print(orig_test_labels.shape)

print(preds.shape)
# Get the confusion matrix

cm  = confusion_matrix(orig_test_labels, preds)

plt.figure()

plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True,cmap=plt.cm.Blues)

plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)

plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)

plt.show()
print(classification_report(orig_test_labels, preds))
# construct the training image generator for data augmentation

aug = ImageDataGenerator(zoom_range=[0.9,1.1],

                         #brightness_range=[0.,1.]

                         #horizontal_flip=True ,

                         #rotation_range=90,

                         #height_shift_range=0.15,

                         #width_shift_range=[-0.125,0.125]

                         )

aug.fit(trainX)
for X_batch, y_batch in aug.flow(trainX, trainY, batch_size=9):

	# create a grid of 3x3 images

	for i in range(0, 9):

		plt.subplot(330 + 1 + i)

		plt.imshow(X_batch[i])

	# show the plot

	plt.show()

	break
EPOCHS =50

BS = 64



def build_model():

  input_img=Input(shape=(224,224,3), name='ImageInput')



  x = Conv2D(16, (3,3),activation='relu', padding='same', name='Conv1_1')(input_img)

  x = Conv2D(16, (3,3), activation='relu', padding='same', name='Conv1_2')(x)

  x = MaxPooling2D((2,2), name='pool1')(x)



  x = Conv2D(32, (3,3),activation='relu', padding='same', name='Conv2_1')(x)

  x = Conv2D(32, (3,3), activation='relu', padding='same', name='Conv2_2')(x)

  x = MaxPooling2D((2,2), name='pool2')(x)



  x = Flatten(name='flatten')(x)

  x = Dense(128, activation='relu', name='fc1')(x)

  x = Dense(64, activation='relu', name='fc2')(x)

  x = Dense(2, activation='softmax', name='fc3')(x)



  model = Model(inputs = input_img, output=x)

  return model
model =  build_model()

model.summary()
opt = Adam(lr=0.0001, decay=1e-5)

chkpt = ModelCheckpoint(filepath='best_aug_model_todate2loss.h5',monitor='val_loss', save_best_only=True, save_weights_only=True)

chkpt2 = ModelCheckpoint(filepath='best_aug_model_todate2acc.h5',monitor='val_accuracy', save_best_only=True, save_weights_only=True)

callbacks_list=[chkpt,chkpt2]

model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer=opt)
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),callbacks=callbacks_list,

	validation_data=(valX, valY), steps_per_epoch=len(trainX) // BS,

	epochs=EPOCHS)
print(H.history.keys())

#  "Accuracy"

plt.plot(H.history['accuracy'])

plt.plot(H.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()

# "Loss"

plt.plot(H.history['loss'])

plt.plot(H.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
model.save_weights("last_epoch.h5")
test_loss_last, test_score_last = model.evaluate(testX, testY, batch_size=32)
print("Loss on test set: ", test_loss_last)

print("Accuracy on test set: ", test_score_last)
model.load_weights("best_aug_model_todate2loss.h5")
test_loss_bestloss, test_score_bestloss = model.evaluate(testX, testY, batch_size=32)
print("Loss on test set: ", test_loss_bestloss)

print("Accuracy on test set: ", test_score_bestloss)
model.load_weights("best_aug_model_todate2acc.h5")
test_loss_bestacc, test_score_bestacc = model.evaluate(testX, testY, batch_size=32)
print("Loss on test set: ", test_loss_bestacc)

print("Accuracy on test set: ", test_score_bestacc)
max_model=np.argmax([test_score_last,test_score_bestloss,test_score_bestacc])

max_model
if max_model == 0:

    model.load_weights("last_epoch.h5")

elif max_model == 1:

    model.load_weights("best_aug_model_todate2loss.h5")

elif max_model == 2:

    model.load_weights("best_aug_model_todate2acc.h5")

    

    

    
# Get predictions

preds = model.predict(testX, batch_size=16)

preds = np.argmax(preds, axis=-1)



# Original labels

orig_test_labels = np.argmax(testY, axis=-1)



print(orig_test_labels.shape)

print(preds.shape)
# Get the confusion matrix

cm  = confusion_matrix(orig_test_labels, preds)

plt.figure()

plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True,cmap=plt.cm.Blues)

plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)

plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)

plt.show()
print(classification_report(orig_test_labels, preds))
# construct the training image generator for data augmentation

aug = ImageDataGenerator(zoom_range=[0.75,1.25],

                         #brightness_range=[0.,1.]

                         #horizontal_flip=True ,

                         #rotation_range=90,

                         #height_shift_range=0.15,

                         #width_shift_range=[-0.125,0.125]

                         )

aug.fit(trainX)
for X_batch, y_batch in aug.flow(trainX, trainY, batch_size=9):

	# create a grid of 3x3 images

	for i in range(0, 9):

		plt.subplot(330 + 1 + i)

		plt.imshow(X_batch[i])

	# show the plot

	plt.show()

	break
EPOCHS =100

BS = 64



def build_model():

  input_img=Input(shape=(224,224,3), name='ImageInput')



  x = Conv2D(16, (3,3),activation='relu', padding='same', name='Conv1_1')(input_img)

  x = Conv2D(16, (3,3), activation='relu', padding='same', name='Conv1_2')(x)

  x = MaxPooling2D((2,2), name='pool1')(x)



  x = Conv2D(32, (3,3),activation='relu', padding='same', name='Conv2_1')(x)

  x = Conv2D(32, (3,3), activation='relu', padding='same', name='Conv2_2')(x)

  x = MaxPooling2D((2,2), name='pool2')(x)



  x = Flatten(name='flatten')(x)

  x = Dense(128, activation='relu', name='fc1')(x)

  x = Dense(64, activation='relu', name='fc2')(x)

  x = Dense(2, activation='softmax', name='fc3')(x)



  model = Model(inputs = input_img, output=x)

  return model
model =  build_model()

model.summary()
opt = Adam(lr=0.0001, decay=1e-5)

chkpt = ModelCheckpoint(filepath='best_aug_model_todate2loss.h5',monitor='val_loss', save_best_only=True, save_weights_only=True)

chkpt2 = ModelCheckpoint(filepath='best_aug_model_todate2acc.h5',monitor='val_accuracy', save_best_only=True, save_weights_only=True)

callbacks_list=[chkpt,chkpt2]

model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer=opt)
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),callbacks=callbacks_list,

	validation_data=(valX, valY), steps_per_epoch=len(trainX) // BS,

	epochs=EPOCHS)
print(H.history.keys())

#  "Accuracy"

plt.plot(H.history['accuracy'])

plt.plot(H.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()

# "Loss"

plt.plot(H.history['loss'])

plt.plot(H.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
model.save_weights("last_epoch.h5")
test_loss_last, test_score_last = model.evaluate(testX, testY, batch_size=32)
print("Loss on test set: ", test_loss_last)

print("Accuracy on test set: ", test_score_last)
model.load_weights("best_aug_model_todate2loss.h5")
test_loss_bestloss, test_score_bestloss = model.evaluate(testX, testY, batch_size=32)
print("Loss on test set: ", test_loss_bestloss)

print("Accuracy on test set: ", test_score_bestloss)
model.load_weights("best_aug_model_todate2acc.h5")
test_loss_bestacc, test_score_bestacc = model.evaluate(testX, testY, batch_size=32)
print("Loss on test set: ", test_loss_bestacc)

print("Accuracy on test set: ", test_score_bestacc)
max_model=np.argmax([test_score_last,test_score_bestloss,test_score_bestacc])

max_model
if max_model == 0:

    model.load_weights("last_epoch.h5")

elif max_model == 1:

    model.load_weights("best_aug_model_todate2loss.h5")

elif max_model == 2:

    model.load_weights("best_aug_model_todate2acc.h5")
# Get predictions

preds = model.predict(testX, batch_size=16)

preds = np.argmax(preds, axis=-1)



# Original labels

orig_test_labels = np.argmax(testY, axis=-1)



print(orig_test_labels.shape)

print(preds.shape)
# Get the confusion matrix

cm  = confusion_matrix(orig_test_labels, preds)

plt.figure()

plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True,cmap=plt.cm.Blues)

plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)

plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)

plt.show()
print(classification_report(orig_test_labels, preds))