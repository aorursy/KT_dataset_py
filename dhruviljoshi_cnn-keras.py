import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pathlib import Path

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

import matplotlib.pyplot as plt

print(os.listdir("/input"))

# print(os.listdir(train_path))



# train_path

# Any results you write to the current directory are saved as output.



from time import time



from sklearn.model_selection import train_test_split



from keras.preprocessing.image import ImageDataGenerator

from keras.applications.vgg16 import VGG16

from keras.callbacks import TensorBoard,EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,Callback

# import the necessary packages

from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation

from keras.layers.core import Flatten

from keras.layers.core import Dropout

from keras.layers.core import Dense

from keras import backend as K
# datapath='../input/'+os.listdir("../input")[1]+'/'

# if not Path(datapath).is_dir():

datapath='/input'

df = pd.read_csv(datapath+'train_labels.csv')

print(df.head())
print('Number of image : ', len(df))

print('Ratio labels : ', sum(df['label'].values)/len(df))

img = plt.imread(datapath+"train/"+df.iloc[0]['id']+'.tif')

print('Images shape', img.shape)
for i in range(5):

    img = plt.imread(datapath+"train/"+df.iloc[i]['id']+'.tif')

    print(df.iloc[i]['label'])

    plt.imshow(img)

    plt.show()
model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = 3, padding = 'same', activation = 'relu', input_shape = (96, 96, 3)))

#96

model.add(Conv2D(filters = 16, kernel_size = 3, padding = 'same', activation = 'relu'))

#96

model.add(Conv2D(filters = 16, kernel_size = 3, padding = 'same', activation = 'relu'))

#96

model.add(Dropout(0.3))

model.add(MaxPooling2D(pool_size = 3))

#32

model.add(Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu'))

model.add(Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu'))

model.add(Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu'))

model.add(Dropout(0.3))

model.add(MaxPooling2D(pool_size = 3))

#11

model.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))

model.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))

model.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))

model.add(Dropout(0.3))

model.add(MaxPooling2D(pool_size = 3))

#4

model.add(Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = 'elu'))

model.add(Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = 'elu'))

model.add(Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = 'elu'))

model.add(Dropout(0.3))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(1, activation = 'sigmoid'))

model.summary()
df = pd.read_csv(datapath+'train_labels.csv',dtype='str')

def append_ext(fn):

    return fn+".tif"



df["id"]=df["id"].apply(append_ext)

train_datagen = ImageDataGenerator(

       # horizontal_flip=True,

       #vertical_flip=True,

       #brightness_range=[0.5, 1.5],

       #fill_mode='reflect',                               

        #rotation_range=15,

        rescale=1./255,

        #shear_range=0.2,

        #zoom_range=0.2

        validation_split=0.15

    

)



test_datagen = ImageDataGenerator(rescale=1./255)



train_path = datapath+'train'

valid_path =  datapath+'train'

test_path=datapath+'test'

print(train_path)

print(valid_path)



print(df.head())

print(df.dtypes)

print(df.loc[0])

print(df.loc[1])
train_generator = train_datagen.flow_from_dataframe(

                dataframe=df,

                directory=train_path,

                x_col = 'id',

                y_col = 'label',

                has_ext=False,

                subset='training',

                target_size=(96, 96),

                batch_size=256,

                class_mode='binary'

                )





validation_generator = train_datagen.flow_from_dataframe(

                dataframe=df,

                directory=valid_path,

                x_col = 'id',

                y_col = 'label',

                has_ext=False,

                subset='validation', # This is the trick to properly separate train and validation dataset

                target_size=(96, 96),

                batch_size=64,

                shuffle=False,

                class_mode='binary'

                )

# df_test = pd.read_csv(datapath+'train_labels.csv')

# test_generator = test_datagen.flow_from_directory(test_path,

#                                         target_size=(96,96),

#                                         batch_size=64,

#                                         class_mode='binary',

#                                         shuffle=False)
class LossHistory(Callback):

    def on_train_begin(self, logs={}):

        self.losses = {'batch': [], 'epoch': []}

        self.accuracy = {'batch': [], 'epoch': []}

        self.val_loss = {'batch': [], 'epoch': []}

        self.val_acc = {'batch': [], 'epoch': []}



    def on_batch_end(self, batch, logs={}):

        self.losses['batch'].append(logs.get('loss'))

        self.accuracy['batch'].append(logs.get('acc'))

        self.val_loss['batch'].append(logs.get('val_loss'))

        self.val_acc['batch'].append(logs.get('val_acc'))



    def on_epoch_end(self, batch, logs={}):

        self.losses['epoch'].append(logs.get('loss'))

        self.accuracy['epoch'].append(logs.get('acc'))

        self.val_loss['epoch'].append(logs.get('val_loss'))

        self.val_acc['epoch'].append(logs.get('val_acc'))



    def plot(self, loss_type):

        iters = range(len(self.losses[loss_type]))

    

        plt.figure(figsize=(16,10))

        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')

        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')

        plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')

        plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')

        plt.grid(True)

        plt.xlabel(loss_type)

        plt.ylabel('acc-loss')

        plt.legend(loc="upper right")

        plt.show()

        

    def save(self,name):

        arr=np.vstack((self.accuracy["epoch"],self.losses["epoch"],self.val_acc["epoch"],self.val_loss["epoch"]))

        np.save(name,arr)

        
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#---

history = LossHistory()

                              

                              

#---

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size

#-----

filepath = "model.h5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 

                             save_best_only=True, mode='max')



reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, 

                                   verbose=1, mode='max', min_lr=0.00001)

                              

                              

callbacks_list = [checkpoint, reduce_lr,history]



#----

his=model.fit_generator(

                train_generator,

                steps_per_epoch=STEP_SIZE_TRAIN,

                epochs=15,

                callbacks=callbacks_list,

                validation_data=validation_generator,

                validation_steps=STEP_SIZE_VALID)
history.plot("epoch")

history.save("A.npy")
import matplotlib.pyplot as plt

train_acc = his.history['acc']

val_acc = his.history['val_acc']



epochs = range(len(train_acc))

plt.plot(epochs,train_acc,'b',label='Training accuracy')

plt.plot(epochs,val_acc,'r',label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()



plt.show()
test_df = pd.read_csv(datapath+'sample_submission.csv')



from matplotlib.pyplot import imread

# Kaggle testing

from glob import glob

TESTING_BATCH_SIZE = 64



# datapath='../input/'+os.listdir("../input")[1]+'/'

# if not Path(datapath).is_dir():

#     datapath='../intput/'

tmp_path=datapath+'test/'

testing_files = glob(os.path.join(tmp_path,'*.tif'))

submission = pd.DataFrame()

print(len(testing_files))

for index in range(0, len(testing_files), TESTING_BATCH_SIZE):

    data_frame = pd.DataFrame({'path': testing_files[index:index+TESTING_BATCH_SIZE]})

    data_frame['id'] = data_frame.path.map(lambda x: x.split('/')[3].split(".")[0])

    data_frame['image'] = data_frame['path'].map(imread)

    images = np.stack(data_frame.image, axis=0)

    predicted_labels = [model.predict(np.expand_dims(image/255.0, axis=0))[0][0] for image in images]

    predictions = np.array(predicted_labels)

    data_frame['label'] = predictions

    submission = pd.concat([submission, data_frame[["id", "label"]]])

    if index % 1000 == 0 :

        print(index/len(testing_files) * 100)

submission.to_csv('submission_new_model.csv', index=False, header=True)

model.save('model.h5')

print(submission.head())