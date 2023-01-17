# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

import matplotlib.pyplot as plt

%matplotlib inline



from keras.models import Sequential

from keras.layers import Dense , Dropout , Lambda, Flatten

from keras.optimizers import Adam ,RMSprop

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

print('keras v' + keras.__version__)

# Any results you write to the current directory are saved as output.
# create the training & test sets, skipping the header row with [1:]

train = pd.read_csv("../input/train.csv")

print(train.shape)

train.head()
test= pd.read_csv("../input/test.csv")

print(test.shape)

test.head()
X_train = (train.ix[:,1:].values).astype('float32') # all pixel values

y_train = train.ix[:,0].values.astype('int32') # only labels i.e targets digits

X_test = test.values.astype('float32')
X_train.shape
X_train
print(y_train.shape)

print(y_train[:2])
#Convert train datset to (num_images, img_rows, img_cols) format 

X_train = X_train.reshape(X_train.shape[0], 28, 28)



for i in range(6, 9):

    plt.subplot(330 + (i+1))

    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))

    plt.title(y_train[i]);
#expand 1 more dimention as 1 for colour channel gray

X_train = X_train.reshape(X_train.shape[0], 28, 28,1)

X_train.shape
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

X_test.shape
mean_px = X_train.mean().astype(np.float32)

std_px = X_train.std().astype(np.float32)



def norm_input(x): 

    return (x-mean_px)/std_px
from keras.utils.np_utils import to_categorical

y_train= to_categorical(y_train)

num_classes = y_train.shape[1]

num_classes
# fix random seed for reproducibility

seed = 43

np.random.seed(seed)
from keras.models import  Sequential

from keras.layers.core import  Lambda , Dense, Flatten, Dropout

from keras.callbacks import EarlyStopping

from keras.layers import BatchNormalization, Conv2D , MaxPooling2D
def get_model():

    model = Sequential([

        Lambda(norm_input, input_shape=(28,28,1)),

        Conv2D(32,(3,3), activation='relu'),

        BatchNormalization(axis=1),

        Conv2D(32,(3,3), activation='relu'),

        MaxPooling2D(),

        BatchNormalization(axis=1),

        Conv2D(64,(3,3), activation='relu'),

        BatchNormalization(axis=1),

        Conv2D(64,(3,3), activation='relu'),

        MaxPooling2D(),

        Flatten(),

        BatchNormalization(),

        Dense(512, activation='relu'),

        BatchNormalization(),

        Dropout(0.5),

        Dense(10, activation='softmax')

        ])

    return model



model = get_model()



for layer in model.layers:

    if 'batch_normalization' in layer.name:

        #print('set momentum')

        layer.momentum = 0.5

print("input shape ",model.input_shape)

print("output shape ",model.output_shape)



model.summary()
from keras.optimizers import RMSprop

model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
from keras.preprocessing import image



val_gen = image.ImageDataGenerator()

gen = image.ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,height_shift_range=0.08, zoom_range=0.08)
from keras.preprocessing import image

from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.callbacks import LearningRateScheduler

import os.path



run_id = 6

auto_weights_path = 'auto_weights.h5'

best_weights_path = 'best_weights' + str(run_id) + '.h5'



if os.path.isfile(best_weights_path) :

    print ("load best_weights_path:" + best_weights_path)

    model.load_weights(best_weights_path)

    

def step_decay(epoch):

    if epoch >= 0 and epoch < 5:

        lrate = 0.001

    elif epoch >= 5 and epoch < 10:

        lrate = 0.1

    elif epoch >= 10 and epoch < 15:

        lrate = 0.001

    elif epoch >= 15 and epoch < 20:

        lrate = 0.0001

    else:

        lrate = 0.00001

    

    print (str(epoch) + " learning rate:%.7f" % lrate)

    return lrate



reduce_lr = LearningRateScheduler(step_decay)



callbacks_list = [

    ModelCheckpoint(auto_weights_path, monitor='val_acc', verbose=1, save_best_only=True),

    EarlyStopping(monitor='val_acc', patience=12, verbose=1),reduce_lr]
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)

batches = gen.flow(X_train, y_train, batch_size=1)

val_batches=val_gen.flow(X_val, y_val, batch_size=1)



epochs = 30

batch_size = 4

steps_per_epoch=int(np.ceil(batches.n/batch_size))

validation_steps=int(np.ceil(val_batches.n/batch_size))



print ("batch_size:" + str(batch_size))

print ("trn_classes:" + str(batches.n))

print ("val_classes:" + str(val_batches.n))

print ("steps_per_epoch:" + str(steps_per_epoch))

print ("validation_steps:" + str(validation_steps))
history = model.fit_generator(batches, 

                    steps_per_epoch=steps_per_epoch, 

                    epochs=epochs, 

                    validation_data=val_batches, 

                    validation_steps=validation_steps,

                    callbacks=callbacks_list,

                    verbose=1)
print("Training accuracy: %.2f%% / Validation accuracy: %.2f%%" % (100*history.history['acc'][-1], 100*history.history['val_acc'][-1]))
best_weights_path = 'best_weights' + str(run_id + 1 ) + '.h5'

model.save_weights(best_weights_path)

print('save_weights:' + best_weights_path)
import matplotlib.pyplot as plt



# list all data in history

print(history.history.keys())



plt.plot(history.history['val_acc'])

plt.plot(history.history['acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
predictions = model.predict_classes(X_test, verbose=0)

submissions_path = "submission3.csv"



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),"Label": predictions})

submissions.to_csv(submissions_path, index=False, header=True)
from IPython.display import FileLink, FileLinks



FileLink(submissions_path)
print(submissions_path)