# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import os
print(os.listdir('/kaggle'))
train_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/'
test_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test/'
val_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/val/'
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from PIL import Image
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                             rescale=1. / 255,
                             featurewise_std_normalization=True,
                             rotation_range=20,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             dtype='float64')

datagen_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input, 
                                    rescale=1. / 255,
                                    featurewise_std_normalization=True, 
                                    dtype='float64')
train_generator = datagen.flow_from_directory(
    directory=train_dir,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=16,
    class_mode='categorical',
    shuffle=True,
    seed=42,
    interpolation='nearest'
)
test_generator = datagen_no_aug.flow_from_directory(
    directory=test_dir,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=1,
    class_mode=None
)
valid_generator = datagen_no_aug.flow_from_directory(
    directory=val_dir,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=16,
    class_mode='categorical',
    shuffle=True,
    seed=42,
    interpolation='nearest'
)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
weights_path='/kaggle/input/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

weights_url='https://www.kaggle.com/xanjay/resnet50#resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    
num_classes = 2

model = Sequential()

model.add(ResNet50(include_top=False, pooling='avg',weights= 'imagenet' ))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

model.layers[0].trainable = False


from tensorflow.python.keras import optimizers

sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)


model.compile(optimizer='sgd', 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
model.summary()
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = 2)
cb_checkpointer = ModelCheckpoint(filepath = '../working/best.hdf5', monitor = 'val_loss', save_best_only = True, mode = 'auto')
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
fit_history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=6,
                    callbacks=[cb_checkpointer, cb_early_stopper]
)
model.load_weights("../working/best.hdf5")
print(fit_history.history.keys())
plt.figure(1, figsize = (15,8)) 
    
plt.subplot(221)  
plt.plot(fit_history.history['accuracy'])  
plt.plot(fit_history.history['val_accuracy'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 
    
plt.subplot(222)  
plt.plot(fit_history.history['loss'])  
plt.plot(fit_history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 

plt.show()
model.evaluate_generator(generator=valid_generator,
steps=STEP_SIZE_VALID)

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
pred=model.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)
print(predicted_class_indices)
print(predicted_class_indices.size)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})

#results[results['Predictions']=='PNEUMONIA'].tail(10)
results.tail(10)
results.head(10)
results.to_csv("/kaggle/working/PNEUMONIA_results.csv",index=False)
import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
#imports the mobilenet model and discards top layer
#base_model=MobileNet(weights='imagenet',include_top=False) 
baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
#model= base_model.output
#model=base_model.output
#model=GlobalAveragePooling2D()(model)
#model=Dense(1024,activation='relu')(model) #we add dense layers so that the model can learn more complex functions and classify for better results.
#model=Dense(1024,activation='relu')(model) #dense layer 2
#model=Dense(512,activation='relu')(model) #dense layer 3
#preds=Dense(1,activation='softmax')(model) #final layer with softmax activation
#model=Model(inputs=base_model.input,outputs=preds)
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)
for layer in baseModel.layers:
    layer.trainable = False
model.summary()

train_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/'
test_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test/'
val_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/val/'

train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input, 
                                 rescale=1. / 255,
                                 featurewise_center = True,
                                 zca_whitening=True,
                                 fill_mode="nearest",
                                 validation_split=0.2,
                                 vertical_flip = True) 

train_generator=train_datagen.flow_from_directory(train_dir, # this is where you specify the path to the main data folder
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode= 'categorical', #'binary', #'categorical',
                                                 shuffle=True,
                                                 subset='training')
valid_generator=train_datagen.flow_from_directory(train_dir, # this is where you specify the path to the main data folder
                                                 target_size=(224,224),
                                                  color_mode='rgb',
                                                 batch_size=4,
                                                 class_mode= 'categorical', #'binary', #'categorical',
                                                 shuffle=True,
                                                 subset='validation')
# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-3
EPOCHS = 25
BS = 8
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
step_size_train=train_generator.n//train_generator.batch_size
step_size_val=valid_generator.n//valid_generator.batch_size
fit_history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=step_size_train, 
                    validation_data=valid_generator,
                    validation_steps=step_size_val,
                    epochs=3)
print(fit_history.history.keys())
#model.load_weights("../working/best.hdf5")
plt.figure(1, figsize = (15,8)) 
    
plt.subplot(221)  
plt.plot(fit_history.history['accuracy'])  
plt.plot(fit_history.history['val_accuracy'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 
    
plt.subplot(222)  
plt.plot(fit_history.history['loss'])  
plt.plot(fit_history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 

plt.show()
test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input, 
                                 rescale=1. / 255,
                                 featurewise_center = True,
                                 zca_whitening=True,
                                 fill_mode="nearest") 

test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=1,
    class_mode=None
)


STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
pred=model.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
pred=model.predict(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)
print(predicted_class_indices)
print(predicted_class_indices.size)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
#results[results['Predictions']=='PNEUMONIA'].tail(10)
results.tail(10)
results.head(10)
!pip install imutils
# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
train_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/'
test_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test/'
val_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/val/'
imgPaths = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train'
imagePaths = list(paths.list_images(imgPaths))
imagePaths
data = []
labels = []
for imagePath in imagePaths:

    # extract the class label from the filename

    label = imagePath.split(os.path.sep)[-2]

    # load the image, swap color channels, and resize it to be a fixed

    # 224x224 pixels while ignoring aspect ratio

    image = cv2.imread(imagePath)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (224, 224))

    # update the data and labels lists, respectively

    data.append(image)

    labels.append(label)
# convert the data and labels to NumPy arrays while scaling the pixel
# intensities to the range [0, 1]
data = np.array(data) / 255.0
labels = np.array(labels)
labels
INIT_LR = 1e-3
EPOCHS = 25
BS = 8
# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.20, stratify=labels, random_state=42)
# initialize the training data augmentation object
trainAug = ImageDataGenerator(rotation_range=15,fill_mode="nearest")
# load the VGG16 network, ensuring the head FC layer sets are left
# off
baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)
# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
# train the head of the network
print("[INFO] training head...")
H = model.fit_generator( trainAug.flow(trainX, trainY, batch_size=BS), 
                        steps_per_epoch=len(trainX) // BS,
                        validation_data=(testX, testY),
                        validation_steps=len(testX) // BS,
                        epochs=EPOCHS)
# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)
# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))
# compute the confusion matrix and and use it to derive the raw
# accuracy, sensitivity, and specificity
cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
# show the confusion matrix, accuracy, sensitivity, and specificity
print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))
# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])