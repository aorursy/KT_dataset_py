import gc, cv2

import numpy as np

import pandas as pd

import tensorflow as tf



from keras import applications

from keras import optimizers

from keras.models import Sequential, Model

from keras.layers import Input, Dense, BatchNormalization, MaxPooling2D, Dropout, Flatten

from keras.preprocessing.image import ImageDataGenerator
traindf = pd.read_csv('/kaggle/input/flower-recognition-he/he_challenge_data/data/train.csv', dtype=str)

testdf = pd.read_csv('/kaggle/input/flower-recognition-he/he_challenge_data/data/test.csv', dtype=str)
traindf['image_id'] = traindf['image_id'] + ".jpg"

testdf['image_id'] = testdf['image_id'] + ".jpg"
datagen = ImageDataGenerator(rescale=1./255.,

                             rotation_range=40,

                             width_shift_range=0.2, 

                             height_shift_range=0.2,

                             shear_range=0.2,

                             zoom_range=0.2,

                             horizontal_flip=True,

                             vertical_flip=True,

                             fill_mode='nearest')



train_generator = datagen.flow_from_dataframe(

dataframe=traindf,

directory="/kaggle/input/flower-recognition-he/he_challenge_data/data/train/",

x_col="image_id",

y_col="category",

subset="training",

batch_size=32,

seed=42,

shuffle=True,

class_mode="categorical",

target_size=(224, 224))
test_datagen=ImageDataGenerator(rescale=1./255.)



test_generator=test_datagen.flow_from_dataframe(

dataframe=testdf,

directory="/kaggle/input/flower-recognition-he/he_challenge_data/data/test/",

x_col="image_id",

y_col=None,

batch_size=1,

seed=42,

shuffle=False,

class_mode=None,

target_size=(224, 224))
vgg_model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (224, 224, 3))



for layer in vgg_model.layers[:-5]:

    layer.trainable=False

    

for layer in vgg_model.layers[1:4]:

    layer.trainable=True

    

input = Input(shape=(224, 224, 3),name = 'image_input')

output_vgg16_conv = vgg_model(input)



x = BatchNormalization()(output_vgg16_conv)

x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

x = Dropout(0.2)(x)



x = Flatten()(x)

x = Dense(4096, activation='relu')(x)

x = BatchNormalization()(x)

x = Dropout(0.3)(x)



x = Dense(1024, activation='relu')(x)

x = BatchNormalization()(x)

x = Dropout(0.3)(x)



x = Dense(512, activation='relu')(x)

x = BatchNormalization()(x)

x = Dropout(0.3)(x)



x = Dense(102, activation='softmax')(x)



model = Model(input=input, output=x)



model.summary()
model.compile(loss="categorical_crossentropy",

              optimizer=optimizers.Adadelta(lr=0.1, rho=0.95, epsilon=1e-08, decay=0.0),

              metrics=['accuracy'])
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

model.fit_generator(generator=train_generator,

                    steps_per_epoch=STEP_SIZE_TRAIN,

                    epochs=30,

                    verbose=1)
test_generator.reset()

pred = model.predict_generator(test_generator,

                               steps=STEP_SIZE_TEST,

                               verbose=1)
predicted_class_indices=np.argmax(pred, axis=1)
labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())

predictions = [labels[k] for k in predicted_class_indices]
filenames=test_generator.filenames

filenames=[f.split('.')[0] for f in filenames]



results=pd.DataFrame({"image_id":filenames,

                      "category":predictions})

results = results.sort_values(by = ['image_id'], ascending = [True])

results.to_csv("results.csv", index=False)