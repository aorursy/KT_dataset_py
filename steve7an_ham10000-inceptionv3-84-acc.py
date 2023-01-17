import pandas as pd

import numpy as np

from keras.models import Sequential

from keras import optimizers

from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

from keras.applications import InceptionV3
base_model= InceptionV3(include_top=False, weights="imagenet", input_shape=(299,299,3))


model= Sequential()

model.add(base_model)

model.add(Conv2D(64, (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.40))

model.add(Flatten())

model.add(Dense(128,activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(7, activation='softmax'))

data=pd.read_csv("../input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv")

data['image_full_name']=data['image_id']+'.jpg'

X=data[['image_full_name','dx','lesion_id']]
from sklearn.model_selection import train_test_split

Y=X.pop('dx').to_frame()

X_train, X_test, y_train, y_test   = train_test_split(X,Y, test_size=0.2, random_state=42)

X_train,X_val,y_train,y_val        =train_test_split(X_train, y_train, test_size=0.25, random_state=42)
train=pd.concat([X_train,y_train],axis=1)

val=pd.concat([X_val,y_val],axis=1)

test=pd.concat([X_test,y_test],axis=1)


from sklearn.preprocessing import LabelEncoder

encoder= LabelEncoder()

encoder.fit(val['dx'])

name_as_indexes_train=encoder.transform(val['dx']) 

val['label']=name_as_indexes_train

encoder=LabelEncoder()

encoder.fit(test['dx'])

name_as_indexes_test=encoder.transform(test['dx']) 

test['label']=name_as_indexes_test

from keras_preprocessing.image import ImageDataGenerator

train_generator = ImageDataGenerator(rescale = 1./255,

                                     rotation_range=10,  

                                     zoom_range = 0.1, 

                                     width_shift_range=0.0,  height_shift_range=0.0) 
train_data= train_generator.flow_from_dataframe(dataframe=train,x_col="image_full_name",y_col="dx",

                                                batch_size=64,directory="../input/mnist1000-with-one-image-folder/ham1000_images/HAM1000_images",

                                                shuffle=True,class_mode="categorical",target_size=(299,299))


test_generator=ImageDataGenerator(rescale = 1./255)
test_data= test_generator.flow_from_dataframe(dataframe=test,x_col="image_full_name",y_col="dx",

                                              directory="../input/mnist1000-with-one-image-folder/ham1000_images/HAM1000_images",

                                              shuffle=False,batch_size=1,class_mode=None,target_size=(299,299))
val_data=test_generator.flow_from_dataframe(dataframe=val,x_col="image_full_name",y_col="dx",

                                            directory="../input/mnist1000-with-one-image-folder/ham1000_images/HAM1000_images",

                                            batch_size=64,shuffle=False,class_mode="categorical",target_size=(299,299))
from keras.callbacks import ReduceLROnPlateau

learning_control = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=.5, min_lr=0.00001)
from keras.callbacks import ModelCheckpoint

# Save the model with best weights

checkpointer = ModelCheckpoint('../input/best.hdf5', verbose=1,save_best_only=True)
model.compile(optimizer=optimizers.adam(lr=0.0001),loss="categorical_crossentropy",metrics=["accuracy"])

model.fit_generator(generator=train_data,

                            steps_per_epoch=train_data.samples//train_data.batch_size,

                            validation_data=val_data,

                            verbose=1,

                            validation_steps=val_data.samples//val_data.batch_size,

                            epochs=30,callbacks=[learning_control, checkpointer])
test_data.reset()

predictions = model.predict_generator(test_data, steps=test_data.samples/test_data.batch_size,verbose=1)

y_pred= np.argmax(predictions, axis=1)
from sklearn.metrics import confusion_matrix 

cm= confusion_matrix(name_as_indexes_test,y_pred)

print(cm)