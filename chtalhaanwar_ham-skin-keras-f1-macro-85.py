%%capture

!pip install efficientnet
import pandas as pd
train=pd.read_csv('../input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv')

train.head()
#count the diseases

train['dx'].value_counts()
from glob import glob

import os

base_skin_dir = '../input/skin-cancer-mnist-ham10000'

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x

                     for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}
train['path'] = train['image_id'].map(imageid_path_dict.get)

train['path']=train['path'].astype('str')

train.head()
train.shape
labels=pd.get_dummies(train['dx'])

train=pd.concat([train['path'],labels],axis=1)
train=train.sample(frac=1)

train.head()
col=list(train.columns)[1::]

col
from sklearn.model_selection import train_test_split

train,val=train_test_split(train,test_size=0.2)
train.shape
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_gen= ImageDataGenerator(

    horizontal_flip=True,

    vertical_flip=True,

    rotation_range=360,

    width_shift_range=0.1,

    height_shift_range=0.1,

    zoom_range=.1,

    rescale=1/255,

    fill_mode='nearest')

img_shape=300

batch_size=32

n_epochs=20
train_generator=data_gen.flow_from_dataframe(train,directory='',

                                                  target_size=(img_shape,img_shape),

                                                  x_col='path',

                                                  y_col=col,

                                                  class_mode='raw',

                                                  shuffle=True,

                                                  batch_size=batch_size)
val_generator=data_gen.flow_from_dataframe(val,directory='',

                                                        target_size=(img_shape,img_shape),

                                                        x_col="path",

                                                        y_col=col,

                                                        class_mode='raw',

                                                        shuffle=False,

                                                        batch_size=batch_size)
train_generator.next()[0].shape,train_generator.next()[1].shape
from tensorflow.keras.layers import GlobalAveragePooling2D,Dropout,Dense

from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau

from tensorflow.keras.optimizers import Adam

import tensorflow.keras.backend as K

from tensorflow.keras.models import Sequential,Model

import efficientnet.tfkeras as efn

model =efn.EfficientNetB3(weights ='noisy-student', include_top=False, input_shape = (img_shape,img_shape,3))
x = model.output

x = GlobalAveragePooling2D()(x)

x = Dropout(0.3)(x)

x = Dense(128, activation="relu")(x)

x = Dropout(0.3)(x)

x = Dense(64, activation="relu")(x)

predictions = Dense(len(col), activation="softmax")(x)

model = Model(inputs=model.input, outputs=predictions)

model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
results = model.fit(train_generator,epochs=20,

    steps_per_epoch=train_generator.n/batch_size,

    validation_data=val_generator,

    validation_steps=val_generator.n/batch_size,

    callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=3, min_lr=0.000001)])
import matplotlib.pyplot as plt

plt.plot(results.history['loss'])

plt.plot(results.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
plt.plot(results.history['accuracy'])

plt.plot(results.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
y_pred = model.predict(val_generator,steps=val_generator.n/batch_size)

y_pred=y_pred.round().astype(int)

y_true=val.iloc[:,1::].values

y_true.shape
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,classification_report

print('Accuracy',accuracy_score(y_true,y_pred))

print('ROC score',roc_auc_score(y_true,y_pred))

print('F1 score',f1_score(y_true,y_pred,average='macro'))
print(classification_report(y_true,y_pred,target_names=col))
model.save_weights('ham_eff3_weights.hdf5')
