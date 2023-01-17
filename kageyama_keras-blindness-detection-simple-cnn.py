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

from keras.applications.inception_v3 import InceptionV3, preprocess_input

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

TRAIN_DIR='../input/aptos2019-blindness-detection/train_images'
def make_train_data(label,path):

    img = cv2.imread(path,cv2.IMREAD_COLOR)

    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))



    X.append(np.array(img))

    Z.append(str(label))
df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

df.head()
x = df['id_code']

y = df['diagnosis']
for id_code,diagnosis in tqdm(zip(x,y)):

    path = os.path.join(TRAIN_DIR,'{}.png'.format(id_code))

    make_train_data(diagnosis,path)
# check some image

fig,ax=plt.subplots(5,2)

fig.set_size_inches(15,15)

for i in range(5):

    for j in range (2):

        l=rn.randint(0,len(Z))

        ax[i,j].imshow(X[l])

        ax[i,j].set_title(Z[l])

        

plt.tight_layout()
Y=to_categorical(Z)

X=np.array(X)

X=X/255
x_train,x_valid,y_train,y_valid = train_test_split(X,Y,test_size=0.2,random_state=42)

del X

del Y

del Z
augs_gen = ImageDataGenerator(

        featurewise_center=False,  

        samplewise_center=False, 

        featurewise_std_normalization=False,  

        samplewise_std_normalization=False,  

        zca_whitening=False,  

        rotation_range=10,  

        zoom_range = 0.1, 

        width_shift_range=0.2,  

        height_shift_range=0.2, 

        horizontal_flip=True,  

        vertical_flip=False) 



augs_gen.fit(x_train)
# # modelling starts using a CNN.



model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (150,150,3)))

model.add(MaxPooling2D(pool_size=(2,2)))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

 



model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



model.add(Flatten())

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dense(5, activation = "softmax"))
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

    patience=5,

    min_lr=1e-6,

    verbose=1, 

    mode='auto'

)



callbacks = [checkpoint,tensorboard,csvlogger,reduce]
batch_size=64

epochs=10
model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
History = model.fit_generator(augs_gen.flow(x_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_valid,y_valid),

                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size,

                              callbacks=callbacks)
test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

test_df.head()
x = test_df['id_code']
TEST_X = []

def make_test_data(path):

    img = cv2.imread(path,cv2.IMREAD_COLOR)

    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))



    TEST_X.append(np.array(img))
TEST_DIR='../input/aptos2019-blindness-detection/test_images'

for id_code in tqdm(x):

    path = os.path.join(TEST_DIR,'{}.png'.format(id_code))

    make_test_data(path)
TEST_X=np.array(TEST_X)

TEST_X=TEST_X/255

pred=model.predict(TEST_X)
pred=np.argmax(pred,axis=1)

pred
sub_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

sub_df.head()
sub_df.diagnosis = pred

sub_df.head()
sub_df.to_csv("submission.csv",index=False)