import pandas as pd



df = pd.read_csv(

    "/kaggle/input/count-the-paperclips/train.csv", 

    na_values=['NA', '?'])



df['filename']="clips-"+df["id"].astype(str)+".png"
TRAIN_PCT = 0.9

TRAIN_CUT = int(len(df) * TRAIN_PCT)



df_train = df[0:TRAIN_CUT]

df_validate = df[TRAIN_CUT:]



print(f"Training size: {len(df_train)}")

print(f"Validate size: {len(df_validate)}")
df_train
import tensorflow as tf

import keras_preprocessing

from keras_preprocessing import image

from keras_preprocessing.image import ImageDataGenerator



IMAGES_DIR = "/kaggle/input/count-the-paperclips/clips-data-2020/clips"



training_datagen = ImageDataGenerator(

  rescale = 1./255,

  horizontal_flip=True,

  vertical_flip=True,

  fill_mode='nearest')



train_generator = training_datagen.flow_from_dataframe(

        dataframe=df_train,

        directory=IMAGES_DIR,

        x_col="filename",

        y_col="clip_count",

        target_size=(256, 256),

        batch_size=32,

        class_mode='other')



validation_datagen = ImageDataGenerator(rescale = 1./255)



val_generator = validation_datagen.flow_from_dataframe(

        dataframe=df_validate,

        directory=IMAGES_DIR,

        x_col="filename",

        y_col="clip_count",

        target_size=(256, 256),

        class_mode='other')
from tensorflow.keras.callbacks import EarlyStopping



model = tf.keras.models.Sequential([

    # Note the input shape is the desired size of the image 150x150 with 3 bytes color

    # This is the first convolution

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(256, 256, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),

    # The second convolution

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    # 512 neuron hidden layer

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(1, activation='linear')

])





model.summary()

epoch_steps = 250 # needed for 2.2

validation_steps = len(df_validate)

model.compile(loss = 'mean_squared_error', optimizer='adam')

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto',

        restore_best_weights=True)

history = model.fit(train_generator,  

  verbose = 1, 

  validation_data=val_generator, callbacks=[monitor], epochs=25)

#  steps_per_epoch=epoch_steps, validation_steps=validation_steps, # needed for 2.2
df_test = pd.read_csv(

    "/kaggle/input/count-the-paperclips/test.csv", 

    na_values=['NA', '?'])



df_test['filename']="clips-"+df_test["id"].astype(str)+".png"



test_datagen = ImageDataGenerator(rescale = 1./255)



test_generator = validation_datagen.flow_from_dataframe(

        dataframe=df_test,

        directory=IMAGES_DIR,

        x_col="filename",

        batch_size=1,

        shuffle=False,

        target_size=(256, 256),

        class_mode=None)
test_generator.reset()

pred = model.predict(test_generator,steps=len(df_test))
df_submit = pd.DataFrame({'id':df_test['id'],'clip_count':pred.flatten()})
df_submit.to_csv("/kaggle/working/submit.csv",index=False)
!ls /kaggle/input/count-the-paperclips/clips-data-2020