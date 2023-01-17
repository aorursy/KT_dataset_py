import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
!cp -r "/kaggle/input/week-2-shopee-299/" "/kaggle/working"
!ls "/kaggle/working/week-2-shopee-299/test"
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

import os
from tqdm import tqdm_notebook as tqdm
digits = ['eighty' , 'fifteen' , 'five']



for M in tqdm(digits):

    for dirname, _, filenames in os.walk('/kaggle/working/week-2-shopee-299/%s/%s/' % (M,M)):

        for filename in filenames:

            try:

                img=Image.open(os.path.join(dirname, filename))

                img.verify()

            except:

                os.remove(os.path.join(dirname, filename))
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

from tensorflow.keras.preprocessing import image

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D , Dropout
datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, brightness_range=(0.9,1.0),

                                  preprocessing_function = preprocess_input)



train_it = datagen.flow_from_directory('/kaggle/working/week-2-shopee-299/eighty/eighty/', 

                                       target_size=(299, 299) , shuffle=True, 

                                       class_mode='categorical', batch_size=256)



valid_it = datagen.flow_from_directory('/kaggle/working/week-2-shopee-299/fifteen/fifteen/',

                                       target_size=(299, 299) , shuffle=False, 

                                     class_mode='categorical', batch_size=256)





test_it = datagen.flow_from_directory('/kaggle/working/week-2-shopee-299/five/five/',

                                       target_size=(299, 299) , shuffle=False,

                                     class_mode='categorical', batch_size=256)



kaggle_it = datagen.flow_from_directory('/kaggle/working/week-2-shopee-299/test/',

                                       target_size=(299, 299) , shuffle=False,

                                     class_mode='categorical', batch_size=256)
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299,299,3))

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)

x = Dropout(0.2)(x)

predictions = Dense(42, activation='softmax')(x)



model = Model(inputs=base_model.input, outputs=predictions)



for layer in base_model.layers:

    layer.trainable = False



model.compile(optimizer='rmsprop', loss="categorical_crossentropy" , metrics = ["accuracy"])



# train the model on the new data for a few epochs

history = model.fit_generator(train_it, validation_data=valid_it, epochs=6)
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.show()
fifteen_fn = valid_it.filenames

fifteen_labels = []



for K in range(4):

    fifteen = model.predict_generator(valid_it)

    fifteen_labels.append(np.argmax(fifteen, axis=1))

    

fifteen_dictionary = {'filename' : fifteen_fn , 

                      'category1' : fifteen_labels[0] , 'category2' : fifteen_labels[1] ,

                     'category3' : fifteen_labels[2] , 'category4' : fifteen_labels[3]}



fifteen_df = pd.DataFrame(fifteen_dictionary)

fifteen_df.head()
five_fn = test_it.filenames

five_labels = []



for K in range(4):

    five = model.predict_generator(test_it)

    five_labels.append(np.argmax(five, axis=1))

    

five_dictionary = {'filename' : five_fn , 

                      'category1' : five_labels[0] , 'category2' : five_labels[1] ,

                     'category3' : five_labels[2] , 'category4' : five_labels[3]}



five_df = pd.DataFrame(five_dictionary)

five_df.head()
kaggle_fn = [I.split("/")[1] for I in kaggle_it.filenames]

kaggle_labels = []



for K in range(4):

    kaggle = model.predict_generator(kaggle_it)

    kaggle_labels.append(np.argmax(kaggle, axis=1))

    

kaggle_dictionary = {'filename' : kaggle_fn , 

                      'category1' : kaggle_labels[0] , 'category2' : kaggle_labels[1] ,

                     'category3' : kaggle_labels[2] , 'category4' : kaggle_labels[3]}



kaggle_df = pd.DataFrame(kaggle_dictionary)

kaggle_df.head()
!rm -r "/kaggle/working/week-2-shopee-299/"
fifteen_df.to_csv("fifteen.csv")

five_df.to_csv("five.csv")

kaggle_df.to_csv("kaggle.csv")