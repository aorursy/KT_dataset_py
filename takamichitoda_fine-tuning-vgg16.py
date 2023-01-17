import numpy as np

import pandas as pd

import os
train = pd.read_csv("../input/imet-2019-fgvc6/train.csv")

labels = pd.read_csv("../input/imet-2019-fgvc6/labels.csv")

sub = pd.read_csv("../input/imet-2019-fgvc6/sample_submission.csv")



train["id"] = train.id.map(lambda x: "{}.png".format(x))

train["attribute_ids"] = train.attribute_ids.map(lambda x: x.split())

sub["id"] = sub.id.map(lambda x: "{}.png".format(x))



display(sub.head())

display(train.head())
batch_size = 32

img_size = 64

nb_epochs = 150

nb_classes = labels.shape[0]

lbls = list(map(str, range(nb_classes)))
%%time



from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.25)



train_generator=train_datagen.flow_from_dataframe(

    dataframe=train,

    directory="../input/imet-2019-fgvc6/train",

    x_col="id",

    y_col="attribute_ids",

    batch_size=batch_size,

    shuffle=True,

    class_mode="categorical",

    classes=lbls,

    target_size=(img_size,img_size),

    subset='training')



valid_generator=train_datagen.flow_from_dataframe(

    dataframe=train,

    directory="../input/imet-2019-fgvc6/train",

    x_col="id",

    y_col="attribute_ids",

    batch_size=batch_size,

    shuffle=True,

    class_mode="categorical",    

    classes=lbls,

    target_size=(img_size,img_size),

    subset='validation')





test_datagen = ImageDataGenerator(rescale=1./255)



test_generator = test_datagen.flow_from_dataframe(  

        dataframe=sub,

        directory = "../input/imet-2019-fgvc6/test",    

        x_col="id",

        target_size = (img_size,img_size),

        batch_size = 1,

        shuffle = False,

        class_mode = None

        )
from keras.applications.vgg16 import VGG16

from keras.layers import Dropout

from keras.models import Sequential

from keras.layers import Dense, Flatten



vgg_conv = VGG16(weights=None, include_top=False, input_shape=(img_size, img_size, 3))

vgg_conv.load_weights('../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')



for layer in vgg_conv.layers[:-4]:

    layer.trainable = False



model = Sequential()

model.add(vgg_conv)

 

model.add(Flatten())

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(nb_classes, activation='softmax'))

 

model.summary()
from keras import optimizers



model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])
history = model.fit_generator(

                    generator=train_generator,

                    steps_per_epoch=100,

                    validation_data=valid_generator,

                    validation_steps=50,

                    epochs=nb_epochs,

                    verbose=1)
import json



with open('history.json', 'w') as f:

    json.dump(history.history, f)



history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot()

history_df[['acc', 'val_acc']].plot()
%%time



test_generator.reset()

predict=model.predict_generator(test_generator, steps = len(test_generator.filenames))
predicted_class_indices = np.argmax(predict,axis=1)



labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())

predictions = [labels[k] for k in predicted_class_indices]



sub["attribute_ids"] = predictions

sub['id'] = sub['id'].map(lambda x: str(x)[:-4])



sub.to_csv("submission.csv",index=False)

sub.head()