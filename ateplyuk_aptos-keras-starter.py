import os

import cv2

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import json

from keras.models import Sequential, Model

from keras.layers import Dense, Flatten, Activation, Dropout, GlobalAveragePooling2D

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers, applications

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

from keras import backend as K 
train_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")

train_df["id_code"]=train_df["id_code"].apply(lambda x:x+".png")

train_df['diagnosis'] = train_df['diagnosis'].astype(str)

train_df.head()
# Example of images 

img_names = train_df['id_code'][:10]



plt.figure(figsize=[15,15])

i = 1

for img_name in img_names:

    img = cv2.imread("../input/aptos2019-blindness-detection/train_images/%s" % img_name)[...,[2, 1, 0]]

    plt.subplot(6, 5, i)

    plt.imshow(img)

    i += 1

plt.show()
nb_classes = 5

lbls = list(map(str, range(nb_classes)))

batch_size = 32

img_size = 64

nb_epochs = 5
%%time



train_datagen=ImageDataGenerator(

    rescale=1./255, 

    validation_split=0.25,

#     horizontal_flip = True,    

#     zoom_range = 0.3,

#     width_shift_range = 0.3,

#     height_shift_range=0.3

    )



train_generator=train_datagen.flow_from_dataframe(

    dataframe=train_df,

    directory="../input/aptos2019-blindness-detection/train_images",

    x_col="id_code",

    y_col="diagnosis",

    batch_size=batch_size,

    shuffle=True,

    class_mode="categorical",

    classes=lbls,

    target_size=(img_size,img_size),

    subset='training')



valid_generator=train_datagen.flow_from_dataframe(

    dataframe=train_df,

    directory="../input/aptos2019-blindness-detection/train_images",

    x_col="id_code",

    y_col="diagnosis",

    batch_size=batch_size,

    shuffle=True,

    class_mode="categorical", 

    classes=lbls,

    target_size=(img_size,img_size),

    subset='validation')
model = applications.ResNet50(weights=None, 

                          include_top=False, 

                          input_shape=(img_size, img_size, 3))

model.load_weights('../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
model.trainable = False
#Adding custom layers 

x = model.output

x = Flatten()(x)

x = Dense(1024, activation="relu")(x)

x = Dropout(0.5)(x)

predictions = Dense(nb_classes, activation="softmax")(x)

model_final = Model(input = model.input, output = predictions)



model_final.compile(optimizers.rmsprop(lr=0.001, decay=1e-6),loss='categorical_crossentropy',metrics=['accuracy'])
# Callbacks



checkpoint = ModelCheckpoint("model_1.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
%%time

history = model_final.fit_generator(generator=train_generator,                   

                                    steps_per_epoch=100,

                                    validation_data=valid_generator,                    

                                    validation_steps=30,

                                    epochs=nb_epochs,

                                    callbacks = [checkpoint, early],

                                    max_queue_size=16,

                                    workers=2,

                                    use_multiprocessing=True,

                                    verbose=0)
with open('history.json', 'w') as f:

    json.dump(history.history, f)



history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot()

history_df[['acc', 'val_acc']].plot()
sam_sub_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

sam_sub_df["id_code"]=sam_sub_df["id_code"].apply(lambda x:x+".png")

print(sam_sub_df.shape)

sam_sub_df.head()
%%time

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(  

        dataframe=sam_sub_df,

        directory = "../input/aptos2019-blindness-detection/test_images",    

        x_col="id_code",

        target_size = (img_size,img_size),

        batch_size = 1,

        shuffle = False,

        class_mode = None

        )
%%time

test_generator.reset()

predict=model_final.predict_generator(test_generator, steps = len(test_generator.filenames))
predict.shape
filenames=test_generator.filenames

results=pd.DataFrame({"id_code":filenames,

                      "diagnosis":np.argmax(predict,axis=1)})

results['id_code'] = results['id_code'].map(lambda x: str(x)[:-4])

results.to_csv("submission.csv",index=False)
results.head()