import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

# Importing the Keras libraries and packages

from keras.preprocessing.image import ImageDataGenerator

import keras

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
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
train_df['diagnosis'].unique()
train_df["id_code"]=train_df["id_code"].apply(lambda x:x+".png")

train_df['diagnosis'] = train_df['diagnosis'].astype(str)

test_df["id_code"]=test_df["id_code"].apply(lambda x:x+".png")
train_df['diagnosis'].value_counts().plot(kind='bar')
nb_classes = 5

lbls = list(map(str, range(nb_classes)))

batch_size = 32

img_size = 150

nb_epochs = 10
train_datagen = ImageDataGenerator(rescale = 1./255,

                                   validation_split=0.3

                                  )
training_set = train_datagen.flow_from_dataframe(

    dataframe=train_df,

    directory="../input/train_images",

    x_col="id_code",

    y_col="diagnosis",

    batch_size=batch_size,

    shuffle=True,

    class_mode="categorical",

    classes=lbls,

    target_size=(img_size,img_size),

    subset='training')



test_set = train_datagen.flow_from_dataframe(

    dataframe=train_df,

    directory="../input/train_images",

    x_col="id_code",

    y_col="diagnosis",

    batch_size=batch_size,

    shuffle=True,

    class_mode="categorical",

    classes=lbls,

    target_size=(img_size,img_size),

    subset='validation'

)
classifier = Sequential()
classifier.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', 

                      input_shape = (img_size,img_size,3)))

classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

 



classifier.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



classifier.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

classifier.add(Flatten())
classifier.add(Dense(units = 512, activation = 'relu'))

classifier.add(Dense(units = 5, activation = 'softmax'))
classifier.compile(optimizer = Adam(lr=0.001),loss='categorical_crossentropy', metrics = ['accuracy'])
classifier.fit_generator(training_set,

                         steps_per_epoch = 10,

                         epochs = nb_epochs,

                         validation_data = test_set,

                         validation_steps = 10)
history = classifier.fit_generator(training_set,

                         steps_per_epoch = 10,

                         epochs = nb_epochs,

                         validation_data = test_set,

                         validation_steps = 10)
# list all data in history

print(history.history.keys())

# summarize history for accuracy

plt.plot(history.history['acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.show()
classifier.summary()
import cv2

import matplotlib.pyplot as plt

test_image = cv2.imread('../input/test_images/3d4d693f7983.png', cv2.IMREAD_COLOR)

test_image = cv2.resize(test_image, (150,150))





plt.imshow(test_image)
test_ids = test_df['id_code']
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(  

        dataframe=test_df,

        directory = "../input/test_images",    

        x_col="id_code",

        target_size = (img_size,img_size),

        batch_size = 1,

        shuffle = False,

        class_mode = None

        )
test_generator.reset()

predict=classifier.predict_generator(test_generator, steps = len(test_generator.filenames))
filenames=test_generator.filenames

results=pd.DataFrame({"id_code":filenames,

                      "diagnosis":np.argmax(predict,axis=1)})

results['id_code'] = results['id_code'].map(lambda x: str(x)[:-4])

results.to_csv("submission.csv",index=False)
results.head()