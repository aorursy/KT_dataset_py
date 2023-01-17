import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

from os import listdir, makedirs, remove
from os.path import join, exists, expanduser

from keras.preprocessing import image
from keras.applications import xception
from keras.applications import inception_v3
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
cache_dir = expanduser(join('~', '.keras'))
if not exists(cache_dir):
    makedirs(cache_dir)
models_dir = join(cache_dir, 'models')
if not exists(models_dir):
    makedirs(models_dir)
!cp ../input/keras-pretrained-models/xception* ~/.keras/models/
!cp ../input/keras-pretrained-models/inception_v3* ~/.keras/models/
batch_size=32
im_size = 299

image_generator = image.ImageDataGenerator(horizontal_flip=True,
                                           zoom_range=.2,
                                           rotation_range=30,
                                           width_shift_range=.2,
                                           height_shift_range=.2,
                                           rescale=1./255,
                                           shear_range=.2,
                                           validation_split=0.1,
                                          brightness_range=(0.4, 1.4))

train_generator = image_generator.flow_from_directory('../input/etipgdla2020c2/train',
                                                    target_size=(im_size, im_size),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    subset='training')

validation_generator = image_generator.flow_from_directory('../input/etipgdla2020c2/train',
                                                         target_size=(im_size, im_size),
                                                         batch_size=batch_size,
                                                         class_mode='categorical',
                                                         subset='validation')
xception_net = xception.Xception(weights='imagenet', include_top=False, pooling='avg')

x = xception_net.output
# x = Dense(1024, activation='relu')(x)
predictions = Dense(20, activation='softmax')(x)

model = Model(inputs=xception_net.input, outputs=predictions)

for layer in xception_net.layers:
    layer.trainable = False

135

checkpoint = ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)
    
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
len(model.layers)
# result = xception_net.predict_generator(train_generator, steps=len(train_generator), verbose=1)
model.fit_generator(train_generator, 
                    steps_per_epoch=len(train_generator) // batch_size,
                    callbacks=[checkpoint],
                    epochs=50,
                    validation_data=validation_generator,
                    validation_steps=len(validation_generator))

for i, layer in enumerate(model.layers):
   print(i, layer.name)
for layer in model.layers[:126]:
   layer.trainable = False


for layer in model.layers[126:]:
    layer.trainable = True
    
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_generator, 
                    steps_per_epoch=len(train_generator) // batch_size,
                    callbacks=[checkpoint],
                    epochs=50,
                    validation_data=validation_generator,
                    validation_steps=len(validation_generator))
test_df = pd.read_csv('../input/etipgdla2020c2/test.csv')
test_df['filenames']=test_df['id']+'.jpg'
test_datagen = image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(test_df,
                                                    '../input/etipgdla2020c2/test',
                                                    x_col='filenames',
                                                    target_size=(im_size, im_size),
                                                    class_mode=None,
                                                    shuffle = False,
                                                    batch_size=1)

filenames = test_generator.filenames
remove("submission.csv")
predict = model.predict_generator(test_generator,steps = len(filenames))
id_to_breed = {y:x for x,y in train_generator.class_indices.items()}
result = np.argmax(predict,axis=1)
result = list(map(lambda x: id_to_breed[x],result))
test_df['breed']=''
for i in range(len(filenames)):
    test_df.at[i,'breed']= result[filenames.index(test_df.at[i,'filenames'])]

predictions=np.column_stack((np.array(test_df.id),np.array(result)))
test_df.drop('filenames',axis=1).to_csv("submission.csv",header=True, index=False)