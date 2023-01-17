import os

import pandas as pd

import numpy as np

from keras.models import Sequential, Model

from keras.layers import Dense, Conv2D, Flatten, Activation, Dropout

from keras.preprocessing.image import ImageDataGenerator

from keras import applications, optimizers
img_size = 224

batch_size = 32
%%time

train_datagen=ImageDataGenerator(

#         rotation_range=40,

#         width_shift_range=0.2,

#         height_shift_range=0.2,

#         shear_range=0.2,

#         zoom_range=0.2,

#         horizontal_flip=True,

#         vertical_flip = True

        )



train_generator = train_datagen.flow_from_directory(

        '../input/myplates/plates/plates/train',

        target_size=(img_size, img_size),

        batch_size=batch_size,

        class_mode='binary')



test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(  

        '../input/myplates/plates/plates',

        classes=['test'],

        target_size = (img_size, img_size),

        batch_size = 1,

        shuffle = False,        

        class_mode = None)    
base_model = applications.InceptionResNetV2(weights='imagenet', 

                          include_top=False, 

                          input_shape=(img_size, img_size, 3))
base_model.trainable = False
x = base_model.output

x = Flatten()(x)

x = Dense(1024, activation="relu")(x)

x = Dropout(0.5)(x)

predictions = Dense(1, activation="sigmoid")(x)

model = Model(input = base_model.input, output = predictions)



model.compile(loss='binary_crossentropy', optimizer = optimizers.rmsprop(lr=0.0001, decay=1e-5), metrics=['accuracy'])
%%time

model.fit_generator(

        train_generator,

        steps_per_epoch=100,

        epochs=50,

        verbose=1)
%%time

test_generator.reset()

predict = model.predict_generator(test_generator, steps = len(test_generator.filenames))

len(predict)
sub_df = pd.read_csv('../input/platesv2/sample_submission.csv')

sub_df.head()
sub_df['label'] = predict

sub_df['label'] = sub_df['label'].apply(lambda x: 'dirty' if x > 0.5 else 'cleaned')

sub_df.head()
sub_df['label'].value_counts()
sub_df.to_csv('sub.csv', index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)





# create a link to download the dataframe

create_download_link(sub_df)