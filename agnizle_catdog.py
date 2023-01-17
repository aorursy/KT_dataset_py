!unzip ../input/dogs-vs-cats-redux-kernels-edition/train.zip -d train
!unzip ../input/dogs-vs-cats-redux-kernels-edition/test.zip -d test
import numpy as np
import re
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
import pandas as pd

import os
print(os.getcwd())
for thisdir, dirnames, filenames in os.walk('.',topdown=True):
    for dirname in dirnames:
        print(os.path.join(thisdir, dirname))

train_dir = 'train/train/'
test_dir = 'test/test/'
weights_dir = '../input/keras-pretrain-model-weights/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5'

## Input data ##
target_size = (224,224,3)
classes = ['cat','dog']
batch_size = 100

pdata = pd.Series(os.listdir(train_dir))
pdata = pd.concat([pdata,pdata.map(lambda x: x[:3])],axis=1)
pdata.columns = ['filename','class']
pdata = pdata.sample(frac=1).reset_index(drop=True) # shuffle rows
pdata.to_csv('catdog_trainfiles.csv', index=False)
pdata.head()

val_split = round(0.8*len(pdata))
print(val_split)
## Transfer learning ##
generator = ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=True, width_shift_range=5, height_shift_range=5)#, rotation_range=5
train_generator = generator.flow_from_dataframe(pdata, class_mode='categorical', classes=classes, shuffle=False,
                                                directory=train_dir, batch_size=batch_size, target_size=target_size[:2])
valid_generator = generator.flow_from_dataframe(pdata.iloc[val_split:], class_mode='categorical', classes=classes, shuffle=False,
                                                directory=train_dir, batch_size=batch_size, target_size=target_size[:2])
test_generator  = generator.flow_from_directory(test_dir+'../', class_mode=None, batch_size=batch_size,
                                                shuffle=False, target_size=target_size[:2])

model = Sequential()
model.add(MobileNetV2(include_top=False, pooling="avg", input_shape=target_size, weights=weights_dir))
#model.add(Dense(10, activation="softmax"))
model.add(Dense(2, activation="softmax"))
model.layers[0].trainable = False

model.compile('rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_generator, steps_per_epoch=200, epochs=1,
                    validation_data=valid_generator, validation_steps=50)

## Fine tuning ##
model.layers[0].trainable = True
finetune_split = int(0.6*len(model.layers[0].layers))
for layer in model.layers[0].layers[:finetune_split]:
    layer.trainable = False
model.compile(RMSprop(learning_rate=0.0005), # default 0.001
              loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_generator, steps_per_epoch=200, epochs=1,
                    validation_data=valid_generator, validation_steps=50)

## Predicting ##
predictions = pd.Series()
test_generator.reset()
for i in range(len(test_generator)):
    if i%5==4: print('iteration '+str(i+1))
    batch = test_generator.next()
    batch_pred = model.predict_on_batch(batch)
    pred_avg = pd.concat([pd.Series(batch_pred[:,1]),1-pd.Series(batch_pred[:,0])], axis=1).mean(axis=1)
    predictions = predictions.append(pred_avg, ignore_index=True)

#predictions = predictions.round()
ids = pd.Series(test_generator.filenames).map(lambda x: int(re.search('\d+',x)[0]))
out = pd.concat([ids,predictions], axis=1)

out.columns = ['id','label']
out = out.sort_values(by=['id'])
out = out.reset_index(drop=True)
print(out.head())
out.to_csv('catdog_predictions.csv', index=False)