%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import os

from glob import glob

import seaborn as sns

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from sklearn.preprocessing import label_binarize

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

import itertools

import keras

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, MaxPooling2D

from keras import backend as K

import itertools

from keras.layers.normalization import BatchNormalization

from keras.utils.np_utils import to_categorical 

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

base_dir = os.path.join('..', 'input/skin-cancer-mnist-ham10000')

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(base_dir, '*', '*.jpg'))}





type_dict = {

    'nv': 'Melanocytic nevi',

    'mel': 'Melanoma',

    'bkl': 'Benign keratosis-like lesions ',

    'bcc': 'Basal cell carcinoma',

    'akiec': 'Actinic keratoses',

    'vasc': 'Vascular lesions',

    'df': 'Dermatofibroma'

}

metadata = pd.read_csv("/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv")

metadata.head()
metadata['cell'] = metadata['dx'].map(type_dict.get) 

metadata['cell_id'] = pd.Categorical(metadata['cell']).codes

metadata['path'] = metadata['image_id'].map(imageid_path_dict.get)
metadata.head()
metadata.tail()
metadata.info()
metadata.isnull().sum()
metadata = metadata.dropna(how='any')

metadata.isnull().sum()
data = metadata.cell

plt.subplots(figsize=(30,15))

wordcloud = WordCloud(

                          background_color='white',

                          width=1024,

                          height=512

                         ).generate(" ".join(data))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph_cell.png')



plt.show()
fig, ax1 = plt.subplots(1, 1, figsize= (10, 5))

metadata['cell'].value_counts().plot(kind='bar', ax=ax1 , color="red" )

plt.savefig('bar1.png')

plt.show()



fig, ax1 = plt.subplots(1, 1, figsize= (10, 5))

metadata['dx_type'].value_counts().plot(kind='bar' , ax=ax1 , color="green")

plt.savefig('bar2.png')

plt.show()



fig, ax1 = plt.subplots(1, 1, figsize= (10, 5))

metadata['localization'].value_counts().plot(kind='bar' , ax=ax1 , color="blue")

plt.savefig('bar3.png')

plt.show()

import plotly.express as px

fig = px.scatter_3d(metadata, x='localization', y='cell', z='sex', color='cell')

fig.show()

import plotly.express as px

fig = px.scatter_3d(metadata, x='localization', y='cell', z='age', color='cell')

fig.show()

fig, ax1 = plt.subplots(2, 2, figsize= (50, 50) )

percent = metadata['sex'].value_counts() / 100

labels = "male" , "female" , "unknown"

ax1[0,0].pie(percent , labels=labels , startangle=180 , autopct='%1.1f%%' ,textprops={ 'fontsize': 19 , 'rotation':0}, shadow=True, radius=1.25)

percent = metadata['cell'].value_counts() / 100

labels = "Melanocytic nevi" , "Melanoma " , "Benign keratosis-like lesions " , "Basal cell carcinoma" , "Actinic keratoses" , "Vascular lesions"  ,"Dermatofibroma"

ax1[0,1].pie(percent , labels=labels , startangle=180 , autopct='%1.1f%%'  ,textprops={'fontsize': 19 , 'rotation':0}, shadow=True, radius=1.25)

percent = metadata['dx_type'].value_counts() / 100

labels = "histo" , "follow_up" , "consensus" , "confocal"

ax1[1,0].pie(percent , labels=labels , startangle=180 , autopct='%1.1f%%'  ,textprops={'fontsize': 19 , 'rotation':0}, shadow=True, radius=1.25)

percent = metadata['localization'].value_counts() / 100

labels = "back" , "lower extremity" , "trunk" , "upper extremity " ,"abdomen" , "face" ,"chest" , "foot" , "unknown" , "neck" , "scalp" , "hand" , "ear" , "genital" , "acral"

ax1[1,1].pie(percent , labels=labels , startangle=180, autopct='%1.1f%%' ,textprops={'fontsize': 19 , 'rotation':0}, shadow=True, radius=1.25)



fig.savefig('pie.png', dpi=300)

plt.show()
metadata['image'] = metadata['path'].map(lambda x: np.asarray(Image.open(x).resize((100,100))))
metadata.head()
n_samples = 7

fig, m_axs = plt.subplots(7, n_samples, figsize = (7*n_samples, 7*7))

for n_axs, (type_name, type_rows) in zip(m_axs,metadata.sort_values(['cell']).groupby('cell')):

    n_axs[0].set_title(type_name)

    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=77).iterrows()):

        c_ax.imshow(c_row['image'])

fig.savefig('category_samples.png', dpi=300)
X_train = metadata.drop(columns=['cell_id'],axis=1)

Y_train = metadata['cell_id']



x_orjinal_train , x_orjinal_test, y_orjinal_train, y_orjinal_test = train_test_split(X_train, Y_train, test_size=0.2,random_state=77)





y_train = to_categorical(y_orjinal_train, num_classes = 7)

y_test = to_categorical(y_orjinal_test, num_classes = 7)



x_train = np.asarray(x_orjinal_train['image'].tolist())

x_test = np.asarray(x_orjinal_test['image'].tolist())



x_train_mean = np.mean(x_train)

x_train_std = np.std(x_train)



x_test_mean = np.mean(x_test)

x_test_std = np.std(x_test)



x_train = (x_train - x_train_mean)/x_train_std

x_test = (x_test - x_test_mean)/x_test_std





x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.1, random_state = 2)



x_train = x_train.reshape(x_train.shape[0], *(100, 100, 3))

x_test = x_test.reshape(x_test.shape[0], *(100, 100, 3))

x_validate = x_validate.reshape(x_validate.shape[0], *(100, 100, 3))
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
from keras.utils.generic_utils import get_custom_objects

from keras.layers import Activation

def swish(x):

    return (K.sigmoid(x) * x)



get_custom_objects().update({'swish': Activation(swish)})



classifier = Sequential()



classifier.add(Conv2D(32, kernel_size= (3,3), activation= 'swish', padding= 'Same', input_shape = (100, 100, 3)))

classifier.add(Conv2D(32, kernel_size= (3,3), activation= 'swish', padding= 'Same'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Dropout(0.25))



classifier.add(Conv2D(64, kernel_size= (3,3), activation= 'swish', padding= 'Same'))

classifier.add(Conv2D(64, kernel_size= (3,3), activation= 'swish', padding= 'Same'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Dropout(0.25))





classifier.add(Flatten())

classifier.add(Dense(128, activation= 'swish'))

classifier.add(Dropout(0.5))



classifier.add(Dense(7, activation= 'softmax'))

classifier.summary()



classifier.compile(optimizer= keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False), loss='categorical_crossentropy', metrics = ['accuracy'])

epochs = 100

batch_size = 32

history = classifier.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_validate,y_validate),

                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size, 

                              callbacks=[learning_rate_reduction])
loss, accuracy = classifier.evaluate(x_test, y_test, verbose=1)

loss_v, accuracy_v = classifier.evaluate(x_validate, y_validate, verbose=1)

print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))

print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))

fig, ax1 = plt.subplots(figsize= (15, 10) )

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.savefig("acc_swish_adam.png")

plt.show()



fig, ax1 = plt.subplots(figsize= (15, 10) )

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.savefig("loss_swish_adam.png")

plt.show()
Y_pred = classifier.predict(x_validate)

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

Y_true = np.argmax(y_validate,axis = 1) 

cm = confusion_matrix(Y_true, Y_pred_classes)



label = pd.unique(metadata['cell'])

f,ax = plt.subplots(figsize=(18, 18))

ax = sns.heatmap(cm, xticklabels=label, yticklabels=label, linewidths=.5 , cbar=False , fmt="d" , annot=True)

plt.savefig("corr_cnn_swish_adam.png")

plt.show()
label_error = 1 - np.diag(cm) / np.sum(cm, axis=1)

plt.figure(figsize=(25,10))

plt.bar(label,label_error)

plt.xlabel('True Label')

plt.ylabel('Classified incorrectly')

plt.savefig("cnn_bar_swish_adam.png")
classifier = Sequential()



classifier.add(Conv2D(32, kernel_size= (3,3), activation= 'relu', padding= 'Same', input_shape = (100, 100, 3)))

classifier.add(Conv2D(32, kernel_size= (3,3), activation= 'relu', padding= 'Same'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Dropout(0.25))



classifier.add(Conv2D(64, kernel_size= (3,3), activation= 'relu', padding= 'Same'))

classifier.add(Conv2D(64, kernel_size= (3,3), activation= 'relu', padding= 'Same'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Dropout(0.25))





classifier.add(Flatten())

classifier.add(Dense(128, activation= 'relu'))

classifier.add(Dropout(0.5))



classifier.add(Dense(7, activation= 'softmax'))

classifier.summary()



classifier.compile(optimizer= keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False), loss='categorical_crossentropy', metrics = ['accuracy'])

epochs = 100

batch_size = 32

history = classifier.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_validate,y_validate),

                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size, 

                              callbacks=[learning_rate_reduction])
loss, accuracy = classifier.evaluate(x_test, y_test, verbose=1)

loss_v, accuracy_v = classifier.evaluate(x_validate, y_validate, verbose=1)

print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))

print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))

fig, ax1 = plt.subplots(figsize= (15, 10) )

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.savefig("acc_relu_adam.png")

plt.show()



fig, ax1 = plt.subplots(figsize= (15, 10) )

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.savefig("loss_relu_adam.png")

plt.show()
Y_pred = classifier.predict(x_validate)

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

Y_true = np.argmax(y_validate,axis = 1) 

cm = confusion_matrix(Y_true, Y_pred_classes)



label = pd.unique(metadata['cell'])

f,ax = plt.subplots(figsize=(18, 18))

ax = sns.heatmap(cm, xticklabels=label, yticklabels=label, linewidths=.5 , cbar=False , fmt="d" , annot=True)

plt.savefig("corr_cnn_relu_adam.png")

plt.show()
label_error = 1 - np.diag(cm) / np.sum(cm, axis=1)

plt.figure(figsize=(25,10))

plt.bar(label,label_error)

plt.xlabel('True Label')

plt.ylabel('Classified incorrectly')

plt.savefig("cnn_bar_relu_adam.png")
classifier = Sequential()



classifier.add(Conv2D(32, kernel_size= (3,3), activation= 'tanh', padding= 'Same', input_shape = (100, 100, 3)))

classifier.add(Conv2D(32, kernel_size= (3,3), activation= 'tanh', padding= 'Same'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Dropout(0.25))



classifier.add(Conv2D(64, kernel_size= (3,3), activation= 'tanh', padding= 'Same'))

classifier.add(Conv2D(64, kernel_size= (3,3), activation= 'tanh', padding= 'Same'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Dropout(0.25))





classifier.add(Flatten())

classifier.add(Dense(128, activation= 'tanh'))

classifier.add(Dropout(0.5))



classifier.add(Dense(7, activation= 'softmax'))

classifier.summary()



classifier.compile(optimizer= keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False), loss='categorical_crossentropy', metrics = ['accuracy'])



epochs = 100

batch_size = 32

history = classifier.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_validate,y_validate),

                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size, 

                              callbacks=[learning_rate_reduction])


loss, accuracy = classifier.evaluate(x_test, y_test, verbose=1)

loss_v, accuracy_v = classifier.evaluate(x_validate, y_validate, verbose=1)

print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))

print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))

fig, ax1 = plt.subplots(figsize= (15, 10) )

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.savefig("acc_tanh_adam.png")

plt.show()



fig, ax1 = plt.subplots(figsize= (15, 10) )

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.savefig("loss_tanh_adam.png")

plt.show()
Y_pred = classifier.predict(x_validate)

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

Y_true = np.argmax(y_validate,axis = 1) 

cm = confusion_matrix(Y_true, Y_pred_classes)



label = pd.unique(metadata['cell'])

f,ax = plt.subplots(figsize=(18, 18))

ax = sns.heatmap(cm, xticklabels=label, yticklabels=label, linewidths=.5 , cbar=False , fmt="d" , annot=True)

plt.savefig("corr_cnn_tanh_adam.png")

plt.show()
label_error = 1 - np.diag(cm) / np.sum(cm, axis=1)

plt.figure(figsize=(25,10))

plt.bar(label,label_error)

plt.xlabel('True Label')

plt.ylabel('Classified incorrectly')

plt.savefig("cnn_bar_tanh_adam.png")
from keras.utils.generic_utils import get_custom_objects

from keras.layers import Activation

def swish(x):

    return (K.sigmoid(x) * x)



get_custom_objects().update({'swish': Activation(swish)})



classifier = Sequential()



classifier.add(Conv2D(32, kernel_size= (3,3), activation= 'swish', padding= 'Same', input_shape = (100, 100, 3)))

classifier.add(Conv2D(32, kernel_size= (3,3), activation= 'swish', padding= 'Same'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Dropout(0.25))



classifier.add(Conv2D(64, kernel_size= (3,3), activation= 'swish', padding= 'Same'))

classifier.add(Conv2D(64, kernel_size= (3,3), activation= 'swish', padding= 'Same'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Dropout(0.25))





classifier.add(Flatten())

classifier.add(Dense(128, activation= 'swish'))

classifier.add(Dropout(0.5))



classifier.add(Dense(7, activation= 'softmax'))

classifier.summary()



classifier.compile(optimizer= keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6), loss='categorical_crossentropy', metrics = ['accuracy'])



epochs = 100

batch_size = 32

history = classifier.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_validate,y_validate),

                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size, 

                              callbacks=[learning_rate_reduction])
loss, accuracy = classifier.evaluate(x_test, y_test, verbose=1)

loss_v, accuracy_v = classifier.evaluate(x_validate, y_validate, verbose=1)

print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))

print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))
fig, ax1 = plt.subplots(figsize= (15, 10) )

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.savefig("acc_swish_rms.png")

plt.show()



fig, ax1 = plt.subplots(figsize= (15, 10) )

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.savefig("loss_swish_rms.png")

plt.show()

Y_pred = classifier.predict(x_validate)

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

Y_true = np.argmax(y_validate,axis = 1) 

cm = confusion_matrix(Y_true, Y_pred_classes)



label = pd.unique(metadata['cell'])

f,ax = plt.subplots(figsize=(18, 18))

ax = sns.heatmap(cm, xticklabels=label, yticklabels=label, linewidths=.5 , cbar=False , fmt="d" , annot=True)

plt.savefig("corr_cnn_swish_rms.png")

plt.show()

label_error = 1 - np.diag(cm) / np.sum(cm, axis=1)

plt.figure(figsize=(25,10))

plt.bar(label,label_error)

plt.xlabel('True Label')

plt.ylabel('Classified incorrectly')

plt.savefig("cnn_bar_swish_rms.png")
classifier = Sequential()



classifier.add(Conv2D(32, kernel_size= (3,3), activation= 'relu', padding= 'Same', input_shape = (100, 100, 3)))

classifier.add(Conv2D(32, kernel_size= (3,3), activation= 'relu', padding= 'Same'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Dropout(0.25))



classifier.add(Conv2D(64, kernel_size= (3,3), activation= 'relu', padding= 'Same'))

classifier.add(Conv2D(64, kernel_size= (3,3), activation= 'relu', padding= 'Same'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Dropout(0.25))





classifier.add(Flatten())

classifier.add(Dense(128, activation= 'relu'))

classifier.add(Dropout(0.5))



classifier.add(Dense(7, activation= 'softmax'))

classifier.summary()



classifier.compile(optimizer= keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6), loss='categorical_crossentropy', metrics = ['accuracy'])



epochs = 100

batch_size = 32

history = classifier.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_validate,y_validate),

                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size, 

                              callbacks=[learning_rate_reduction])
loss, accuracy = classifier.evaluate(x_test, y_test, verbose=1)

loss_v, accuracy_v = classifier.evaluate(x_validate, y_validate, verbose=1)

print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))

print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))
fig, ax1 = plt.subplots(figsize= (15, 10) )

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.savefig("acc_relu_rms.png")

plt.show()



fig, ax1 = plt.subplots(figsize= (15, 10) )

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.savefig("loss_relu_rms.png")

plt.show()
Y_pred = classifier.predict(x_validate)

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

Y_true = np.argmax(y_validate,axis = 1) 

cm = confusion_matrix(Y_true, Y_pred_classes)



label = pd.unique(metadata['cell'])

f,ax = plt.subplots(figsize=(18, 18))

ax = sns.heatmap(cm, xticklabels=label, yticklabels=label, linewidths=.5 , cbar=False , fmt="d" , annot=True)

plt.savefig("corr_cnn_relu_rms.png")

plt.show()
label_error = 1 - np.diag(cm) / np.sum(cm, axis=1)

plt.figure(figsize=(25,10))

plt.bar(label,label_error)

plt.xlabel('True Label')

plt.ylabel('Classified incorrectly')

plt.savefig("cnn_bar_relu_rms.png")
classifier = Sequential()



classifier.add(Conv2D(32, kernel_size= (3,3), activation= 'tanh', padding= 'Same', input_shape = (100, 100, 3)))

classifier.add(Conv2D(32, kernel_size= (3,3), activation= 'tanh', padding= 'Same'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Dropout(0.25))



classifier.add(Conv2D(64, kernel_size= (3,3), activation= 'tanh', padding= 'Same'))

classifier.add(Conv2D(64, kernel_size= (3,3), activation= 'tanh', padding= 'Same'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Dropout(0.25))





classifier.add(Flatten())

classifier.add(Dense(128, activation= 'tanh'))

classifier.add(Dropout(0.5))



classifier.add(Dense(7, activation= 'softmax'))

classifier.summary()



classifier.compile(optimizer= keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6), loss='categorical_crossentropy', metrics = ['accuracy'])

epochs = 100

batch_size = 32

history = classifier.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_validate,y_validate),

                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size, 

                              callbacks=[learning_rate_reduction])
loss, accuracy = classifier.evaluate(x_test, y_test, verbose=1)

loss_v, accuracy_v = classifier.evaluate(x_validate, y_validate, verbose=1)

print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))

print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))
fig, ax1 = plt.subplots(figsize= (15, 10) )

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.savefig("acc_tanh_rms.png")

plt.show()



fig, ax1 = plt.subplots(figsize= (15, 10) )

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.savefig("loss_tanh_rms.png")

plt.show()



Y_pred = classifier.predict(x_validate)

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

Y_true = np.argmax(y_validate,axis = 1) 

cm = confusion_matrix(Y_true, Y_pred_classes)



label = pd.unique(metadata['cell'])

f,ax = plt.subplots(figsize=(18, 18))

ax = sns.heatmap(cm, xticklabels=label, yticklabels=label, linewidths=.5 , cbar=False , fmt="d" , annot=True)

plt.savefig("corr_cnn_tanh_rms.png")

plt.show()

label_error = 1 - np.diag(cm) / np.sum(cm, axis=1)

plt.figure(figsize=(25,10))

plt.bar(label,label_error)

plt.xlabel('True Label')

plt.ylabel('Classified incorrectly')

plt.savefig("cnn_bar_tanh_rms.png")
from keras.utils import plot_model

plot_model(classifier)