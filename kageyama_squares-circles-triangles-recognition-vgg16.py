# Ignore  the warnings

import warnings

warnings.filterwarnings('always')

warnings.filterwarnings('ignore')



# data visualisation and manipulation

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

 

#configure

# sets matplotlib to inline and displays graphs below the corressponding cell.

%matplotlib inline  

style.use('fivethirtyeight')

sns.set(style='whitegrid',color_codes=True)



#model selection

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder



#preprocess.

from keras.preprocessing.image import ImageDataGenerator



#dl libraraies

from keras import backend as K

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop

from keras.utils import to_categorical

from keras.utils.vis_utils import model_to_dot

from keras.utils.vis_utils import plot_model



# specifically for cnn

from keras.layers import Dropout, Flatten,Activation

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,GlobalAveragePooling2D

from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,CSVLogger,ReduceLROnPlateau,LearningRateScheduler

    

import tensorflow as tf

import random as rn



# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.

import cv2                  

import numpy as np  

from tqdm import tqdm

import os                   

from random import shuffle  

from zipfile import ZipFile

from PIL import Image



print(os.listdir("../input"))
X=[]

Z=[]

IMG_SIZE=150

CIRCLES_DIR='../input/shapes/shapes/circles'

SQUARES_DIR='../input/shapes/shapes/squares'

TRIANGLES_DIR='../input/shapes/shapes/triangles'
def assign_label(img,label):

    return label
def make_train_data(label,DIR):

    for img in tqdm(os.listdir(DIR)):

        label=assign_label(img,label)

        path = os.path.join(DIR,img)

        img = cv2.imread(path,cv2.IMREAD_COLOR)

        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))

        

        X.append(np.array(img))

        Z.append(str(label))
# make 'circle' data

make_train_data('circle',CIRCLES_DIR)

print(len(X))
# make 'square' data

make_train_data('square',SQUARES_DIR)

print(len(X))
# make 'triangle' data

make_train_data('triangle',TRIANGLES_DIR)

print(len(X))
# check some image

fig,ax=plt.subplots(5,2)

fig.set_size_inches(15,15)

for i in range(5):

    for j in range (2):

        l=rn.randint(0,len(Z))

        ax[i,j].imshow(X[l])

        ax[i,j].set_title('Shape: '+Z[l])

        

plt.tight_layout()
le=LabelEncoder()

Y=le.fit_transform(Z)

Y=to_categorical(Y,3)

X=np.array(X)

X=X/255
# separate data

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.35,random_state=42)
# fix random seed

np.random.seed(42)

rn.seed(42)

tf.set_random_seed(42)
from keras.applications.vgg16 import VGG16

from IPython.display import SVG



base_model = VGG16(include_top=False,

                  input_shape = (IMG_SIZE,IMG_SIZE,3),

                  weights = 'imagenet')



for layer in base_model.layers:

    layer.trainable = False

    

for layer in base_model.layers:

    print(layer,layer.trainable)



model = Sequential()

model.add(base_model)

model.add(GlobalAveragePooling2D())

model.add(Dropout(0.5))

model.add(Dense(3,activation='softmax'))

model.summary()



SVG(model_to_dot(model).create(prog='dot', format='svg'))

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
# set callbacks

checkpoint = ModelCheckpoint(

    './base.model',

    monitor='val_loss',

    verbose=1,

    save_best_only=True,

    mode='min',

    save_weights_only=False,

    period=1

)

earlystop = EarlyStopping(

    monitor='val_loss',

    min_delta=0.001,

    patience=30,

    verbose=1,

    mode='auto'

)

tensorboard = TensorBoard(

    log_dir = './logs',

    histogram_freq=0,

    batch_size=16,

    write_graph=True,

    write_grads=True,

    write_images=False,

)



csvlogger = CSVLogger(

    filename= "training_csv.log",

    separator = ",",

    append = False

)



reduce = ReduceLROnPlateau(

    monitor='val_loss',

    factor=0.1,

    patience=3,

    verbose=1, 

    mode='auto'

)



callbacks = [checkpoint,tensorboard,csvlogger,reduce]
batch_size=10

epochs=20
# data augmentation to prevent overfitting

datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=True,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(x_train)
model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
History = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_test,y_test),

                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)
plt.plot(History.history['loss'])

plt.plot(History.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(['train', 'test'])

plt.show()
plt.plot(History.history['acc'])

plt.plot(History.history['val_acc'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epochs')

plt.legend(['train', 'test'])

plt.show()
# getting predictions on val set.

pred=model.predict(x_test)

pred_digits=np.argmax(pred,axis=1)
# now storing some properly as well as misclassified indexes'.

i=0

prop_class=[]

mis_class=[]



for i in range(len(y_test)):

    if(np.argmax(y_test[i])==pred_digits[i]):

        prop_class.append(i)

    if(len(prop_class)==8):

        break



i=0

for i in range(len(y_test)):

    if(not np.argmax(y_test[i])==pred_digits[i]):

        mis_class.append(i)

    if(len(mis_class)==8):

        break
warnings.filterwarnings('always')

warnings.filterwarnings('ignore')



count=0

fig,ax=plt.subplots(4,2)

fig.set_size_inches(15,15)

for i in range (4):

    for j in range (2):

        ax[i,j].imshow(x_test[prop_class[count]])

        ax[i,j].set_title("Predicted : "+str(le.inverse_transform([pred_digits[prop_class[count]]]))+"\n"+"Actual : "+str(le.inverse_transform([np.argmax([y_test[prop_class[count]]])])))

        plt.tight_layout()

        count+=1