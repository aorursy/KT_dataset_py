import os

import numpy as np

import pandas as pd

import keras

from keras.models import Sequential

from scipy.ndimage import convolve



from random import randrange 



from keras.applications import DenseNet121

from keras.preprocessing import image

from keras.applications.vgg16 import preprocess_input

from keras.layers import Input, Flatten, Dense, Dropout, Activation

from keras.layers import MaxPooling2D, AveragePooling2D, Conv2D

from keras.models import Model,Sequential

from keras.utils import np_utils



from keras.preprocessing.image import ImageDataGenerator

# some code from https://marcotcr.github.io/lime/tutorials/Tutorial%20-%20images.html tute



from keras.optimizers import SGD 

img_x = 224

img_y = 224

bat_siz = 32

num_epok = 32

# In[2]:





data_generator = ImageDataGenerator(

        zoom_range = 0.4,

        vertical_flip  = True,

        horizontal_flip = True,

        rescale=1.0/255.0

        )
train_data_labels = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")

train_data_labels['id_code'] = train_data_labels['id_code'] + '.png'

#train_data_labels['diagnosis'] = train_data_labels['diagnosis'].to_string()

train_data_labels['diagnosis'] = train_data_labels['diagnosis'].apply(str)



train_generator = data_generator.flow_from_dataframe(dataframe = train_data_labels,

                                                     directory = os.path.join('..', 'input','aptos2019-blindness-detection', 'train_images'),

        target_size = (img_x, img_y), 

        y_col = 'diagnosis',

        x_col = 'id_code',

        class_mode = 'categorical',

        batch_size = bat_siz

        )
densenet = DenseNet121(

    weights='../input/densenet-keras/DenseNet-BC-121-32-no-top.h5',

    include_top=False,

    input_shape=(224,224,3)

)

## Now, define the model

model = Sequential()

model.add(densenet)

model.add(Dropout(0.5)) # unlearn 

model.add(Dense(256, activation = "softplus"))

model.add(Dense(256, activation = "softplus"))

## Need to figure out ways to use convolution layers w/o eating all resources

model.add(Conv2D(64, (3, 3)))

model.add(Activation('softplus'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation = "softplus"))

model.add(Dropout(0.5))

model.add(Dense(5, activation = 'softmax'))



sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',

              optimizer=sgd,

              metrics=['accuracy'])
model.fit_generator(

        train_generator,

        steps_per_epoch = train_data_labels.shape[0]/bat_siz,

        epochs = num_epok)
test_data_labels = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")

test_data_labels['id_code'] = test_data_labels['id_code'] + '.png'





test_generator = data_generator.flow_from_dataframe(dataframe = test_data_labels,

                                                     directory = os.path.join('..', 'input','aptos2019-blindness-detection','test_images'),

        target_size = (img_x, img_y), 

    

        x_col = 'id_code',

        class_mode = None,

        batch_size = bat_siz

        )



predictions = model.predict_generator(test_generator,

                                      steps = test_data_labels.shape[0]/bat_siz)



pred_holder = []

for x in predictions:

    pred_holder.append(np.argmax(x))
for i in range(10):

    print(predictions[i])

    print(pred_holder[i])


output_df = pd.DataFrame({'diagnosis':pred_holder, 

                          'id_code':test_data_labels.id_code.str.replace(pat = "\.png", repl = "")})





output_df.to_csv("submission.csv", mode = "w")

output_df.head()



model.save(os.path.join(".","densenet_plus_five"))
