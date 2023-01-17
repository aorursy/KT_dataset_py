from tensorflow.keras.callbacks import CSVLogger



csv_logger = CSVLogger('/kaggle/working/log.csv', append=True, separator = ',')



import numpy as np 

import pandas as pd



from sklearn.model_selection import StratifiedShuffleSplit

# importing the keras libraries and packages

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, Flatten, MaxPool2D, Dense, Dropout

from tensorflow.keras import optimizers

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow.keras.backend as K



def f1_sc(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    recall = true_positives / (possible_positives + K.epsilon())

    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())

    return f1_val



METRICS = [tf.keras.metrics.CategoricalAccuracy(name='accuracy'),

           tf.keras.metrics.AUC(name='auc'),f1_sc]



# initialising the CNN



model = Sequential()



model.add(Conv2D(input_shape=(224,224,3),filters=32,kernel_size=(3,3),padding="same", activation="relu",data_format='channels_last'))

model.add(Conv2D(filters=32,kernel_size=(3,3),padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Dropout(0.2))



model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Dropout(0.2))



model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(units=512,activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(units=256,activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(units=64,activation="relu"))



model.add(Dense(units=4, activation="softmax"))



model.compile(optimizer=optimizers.Adam(lr = 0.0001),loss=tf.keras.losses.CategoricalCrossentropy(),metrics=METRICS)



# fitting the CNN to the images

from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale=1./255,

                                   horizontal_flip=True,

                                   vertical_flip=True)



test_datagen = ImageDataGenerator(rescale=1./255,

                                  horizontal_flip=True,

                                  vertical_flip=True)



# splitting the dataset into training set and test set



df = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')

X = df['image_id'].values

y = df[['healthy', 'multiple_diseases', 'rust', 'scab']].values



sss = StratifiedShuffleSplit(n_splits=2, test_size=0.25, random_state=101)

sss.get_n_splits(X, y)



print(sss)



for train_index, test_index in sss.split(X, y):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]



y_test = pd.DataFrame(y_test, columns = ['healthy', 'multiple_diseases', 'rust', 'scab'])

y_train = pd.DataFrame(y_train, columns = ['healthy', 'multiple_diseases', 'rust', 'scab'])

X_test = pd.DataFrame(X_test+'.jpg', columns = ['image_id'])

X_train = pd.DataFrame(X_train+'.jpg', columns = ['image_id'])



testing_set = pd.concat([X_test,y_test], axis =1)

print(testing_set.sum())

training_set = pd.concat([X_train,y_train], axis =1)

print(training_set.sum())



train_generator = train_datagen.flow_from_dataframe(dataframe=training_set,

                                                    directory='/kaggle/input/plant-pathology-2020-fgvc7/images',

                                                    x_col="image_id",

                                                    y_col=['healthy', 'multiple_diseases', 'rust', 'scab'],

                                                    target_size=(224, 224),

                                                    batch_size=32,

                                                    class_mode='raw')



validation_generator = test_datagen.flow_from_dataframe(dataframe=testing_set,

                                                        directory="/kaggle/input/plant-pathology-2020-fgvc7/images",

                                                        x_col="image_id",

                                                        y_col=['healthy', 'multiple_diseases', 'rust', 'scab'],

                                                        target_size=(224, 224),

                                                        batch_size=32,

                                                        class_mode='raw')



mc = ModelCheckpoint(filepath="/kaggle/working/best_model.h5", monitor= 'val_f1_sc', verbose=1, save_best_only=True, save_weights_only=False, mode='max', validation_freq=1)

es = EarlyStopping(monitor='val_f1_sc', min_delta = 0, patience=20, verbose = 1, mode='max', baseline=None)



class_weight = {0 : 0.881783, 1 : 5.01838, 2 : 0.732296, 3 : 0.768581}



model.fit_generator(train_generator,shuffle = True, max_queue_size=32,

                         epochs=200, validation_data=validation_generator,

                    class_weight=class_weight, callbacks=[es,mc,csv_logger])