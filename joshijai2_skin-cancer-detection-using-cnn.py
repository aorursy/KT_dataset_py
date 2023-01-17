#importing required libraries

%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import os

from glob import glob

import seaborn as sns

from PIL import Image

from sklearn.preprocessing import label_binarize

from sklearn.metrics import confusion_matrix

import itertools



import keras

from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras import backend as K

import itertools

from keras.layers.normalization import BatchNormalization

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding



from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
#1. Function to plot model's validation loss and validation accuracy

def plot_model_history(model_history):

    fig, axs = plt.subplots(1,2,figsize=(15,5))

    # summarize history for accuracy

    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])

    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])

    axs[0].set_title('Model Accuracy')

    axs[0].set_ylabel('Accuracy')

    axs[0].set_xlabel('Epoch')

#     axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)

    axs[0].legend(['train', 'val'], loc='best')

    # summarize history for loss

    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])

    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])

    axs[1].set_title('Model Loss')

    axs[1].set_ylabel('Loss')

    axs[1].set_xlabel('Epoch')

#     axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)

    axs[1].legend(['train', 'val'], loc='best')

    plt.show()
#2. Function to plot confusion matrix    

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



liesons=['akiec','bcc','bkl','nv','df','mel','vasc']

lesion_type_dict = {

    'nv': 'Melanocytic nevi',

    'mel': 'Melanoma',

    'bkl': 'Benign keratosis-like lesions ',

    'bcc': 'Basal cell carcinoma',

    'akiec': 'Actinic keratoses',

    'vasc': 'Vascular lesions',

    'df': 'Dermatofibroma'

}
#set image path in variables

base_skin_dir = os.path.join('..', 'input/skin-cancer-mnist-ham10000')

image_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}

#read the dataset

df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))

df['path'] = df['image_id'].map(image_path_dict.get)

df['cell_type'] = df['dx'].map(lesion_type_dict.get) 

df['cell_type_idx'] = pd.Categorical(df['cell_type']).codes

df.head()
#add the images from the path to dataset

df['image'] = df['path'].map(lambda x: np.asarray(Image.open(x).resize((100,75))))
df.head()
#view some data samples

n_samples = 5

fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))

for n_axs, (type_name, type_rows) in zip(m_axs, 

                                         df.sort_values(['cell_type']).groupby('cell_type')):

    n_axs[0].set_title(type_name)

    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=0).iterrows()):

        c_ax.imshow(c_row['image'])

        c_ax.axis('off')

fig.savefig('category_samples.png', dpi=300)
#view the shape of the data

df['image'].map(lambda x: x.shape).value_counts()
#Train Test Split

features=df.drop(columns=['cell_type_idx'],axis=1)

target=df['cell_type_idx']



x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(features, target, test_size=0.20,random_state=0)
#find missing values

print("x_train:\n",x_train_o.isnull().sum())

print("\nx_test:\n",x_test_o.isnull().sum())
#impute missing values

x_train_o['age'].fillna((x_train_o['age'].mean()), inplace=True)

x_test_o['age'].fillna((x_test_o['age'].mean()), inplace=True)
#Normalization

x_train = np.asarray(x_train_o['image'].tolist())

x_test = np.asarray(x_test_o['image'].tolist())



x_train_mean = np.mean(x_train)

x_train_std = np.std(x_train)



x_test_mean = np.mean(x_test)

x_test_std = np.std(x_test)



x_train = (x_train - x_train_mean)/x_train_std

x_test = (x_test - x_test_mean)/x_test_std
# Perform one-hot encoding on the labels

y_train = to_categorical(y_train_o)

y_test = to_categorical(y_test_o)
print(y_train.shape)
#Splitting training and validation split

x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.1, random_state = 2)
print(y_train.shape)
# Reshape image in 3 dimensions (height = 75px, width = 100px , canal = 3)

x_train = x_train.reshape(x_train.shape[0], *(75, 100, 3))

x_test = x_test.reshape(x_test.shape[0], *(75, 100, 3))

x_validate = x_validate.reshape(x_validate.shape[0], *(75, 100, 3))
# Set the CNN model 

# my CNN architechture is In -> [[Conv2D->relu] -> MaxPool2D -> Dropout]*3 -> Flatten -> Dense -> Dropout -> Out

input_shape = (75, 100, 3)

num_classes = 7



model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding = 'Same',input_shape=input_shape))

model.add(MaxPool2D(pool_size = (2, 2)))

model.add(Dropout(0.2))



model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.3))



model.add(Conv2D(128,kernel_size=(3, 3), activation='relu',padding = 'Same',))

model.add(MaxPool2D(pool_size = (2, 2)))

model.add(Dropout(0.4))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

model.summary()
#compiling the model with adam optimizer

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1,factor=0.5, min_lr=0.00001)
# With data augmentation to prevent overfitting 



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
# Fit the model

epochs = 50 

batch_size = 10

cnn_history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_validate,y_validate),

                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size,

                              callbacks=[learning_rate_reduction])
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)

loss_v, accuracy_v = model.evaluate(x_validate, y_validate, verbose=1)

print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))

print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))

model.save("model.h5")
#generatingthemodel

ann_model=Sequential()

ann_model.add(Flatten())

#firstlayerwith500neuronsandreluactivationfunction

ann_model.add(Dense(500,kernel_initializer="glorot_normal",

bias_initializer="glorot_normal",activation="relu"))

#secondlayerwith500neuronsandreluactivationfunction

ann_model.add(Dense(500,kernel_initializer="glorot_normal",

bias_initializer="glorot_normal",activation="relu"))

#thirdlayerwith500neuronsandreluactivationfunction

ann_model.add(Dense(500,kernel_initializer="glorot_normal",

bias_initializer="glorot_normal",activation="relu"))

#fourthlayerwith500neuronsandreluactivationfunction

ann_model.add(Dense(500,kernel_initializer="glorot_normal",

bias_initializer="glorot_normal",activation="relu"))

#fifthlayerwith500neuronsandreluactivationfunction

ann_model.add(Dense(500,kernel_initializer="glorot_normal",

bias_initializer="glorot_normal",activation="relu"))

#outputsoftmaxlayer

ann_model.add(Dense(7,kernel_initializer="glorot_normal",

bias_initializer="glorot_normal",activation="softmax"))



ann_model.compile(loss="categorical_crossentropy",

optimizer="adam",

metrics=["accuracy"])



ann_history=ann_model.fit(x_train,y_train,epochs=25,batch_size=20,validation_data = (x_validate,y_validate),verbose = 1)



loss_ann, accuracy_ann = ann_model.evaluate(x_test, y_test, verbose=1)

loss_ann_v, accuracy_ann_v = ann_model.evaluate(x_validate, y_validate, verbose=1)

print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_ann_v, loss_ann_v))

print("Test: accuracy = %f  ;  loss = %f" % (accuracy_ann, loss_ann))

model.save("model_ann.h5")
plot_model_history(cnn_history)

plot_model_history(ann_history)
# Predict the values from the validation dataset

Y_pred = model.predict(x_validate)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(y_validate,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(7)) 

# Predict the values from the validation dataset

Y_ann_pred = ann_model.predict(x_validate)

# Convert predictions classes to one hot vectors 

Y_ann_pred_classes = np.argmax(Y_ann_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_ann_true = np.argmax(y_validate,axis = 1) 

# compute the confusion matrix

confusion_mtx_ann = confusion_matrix(Y_ann_true, Y_ann_pred_classes)



# plot the confusion matrix

plot_confusion_matrix(confusion_mtx_ann, classes = range(7)) 
