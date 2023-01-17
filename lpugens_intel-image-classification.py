# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print(tf.__version__)

print(tf.config.list_physical_devices())

tf.random.set_seed(25081994)
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(

    samplewise_center=True,

    samplewise_std_normalization=True,

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True

)
training_flow = image_generator.flow_from_directory(

    '/kaggle/input/intel-image-classification/seg_train/seg_train',

    target_size=(224, 224),

    batch_size=64

)

test_flow = image_generator.flow_from_directory(

    '/kaggle/input/intel-image-classification/seg_test/seg_test',

    target_size=(224, 224),

    shuffle=False,

    batch_size=64

)
CLASSES = os.walk('/kaggle/input/intel-image-classification/seg_train/seg_train')

CLASSES = next(CLASSES)[1]
base_model = tf.keras.applications.MobileNetV2(

    include_top=False

)

base_model.trainable = False



x = base_model.output

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dense(128, activation='relu')(x)

preds = tf.keras.layers.Dense(6, activation='softmax')(x)



model = tf.keras.Model(inputs=base_model.input, outputs=preds)



model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['loss'])



callback = tf.keras.callbacks.EarlyStopping(

    monitor='accuracy',

    patience=3,

    restore_best_weights=True

)
model.fit(

    training_flow,

    validation_data=test_flow,

    epochs=100,

    callbacks=[callback]

)
base_model.trainable = True

opt = tf.keras.optimizers.Adam(learning_rate=1e-5)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(

    training_flow,

    validation_data=test_flow,

    epochs=100,

    callbacks=[callback]

)
model.save('model')
image_pred_generator = tf.keras.preprocessing.image.ImageDataGenerator(

    samplewise_center=True,

    samplewise_std_normalization=True

)

prediction_flow = image_generator.flow_from_directory(

    '../input/intel-image-classification/seg_pred',

    shuffle=False,

    target_size=(224, 224)

)

csv_logger = tf.keras.callbacks.CSVLogger('result.csv')
results = model.predict_generator(

    prediction_flow

)
index = next(os.walk('/kaggle/input/intel-image-classification/seg_pred/seg_pred/'))[2]

prediction_flow.reset()

df = pd.DataFrame(data=results, columns=list(training_flow.class_indices.keys()), index=prediction_flow.filenames)
df.to_csv('result_df.csv')