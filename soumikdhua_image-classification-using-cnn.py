import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import random



from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.utils import to_categorical

from keras.utils.vis_utils import plot_model



from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

%matplotlib inline

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras import layers

from keras.utils.vis_utils import plot_model

import keras.backend as K

from tensorflow.keras.layers import Dense, ZeroPadding2D, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, GlobalMaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Dropout, Add, Concatenate

from tensorflow.keras import Input

from tensorflow.keras.initializers import glorot_uniform

from tensorflow.keras.optimizers import Adam, RMSprop
TRAIN_DIR = "/kaggle/input/nnfl-lab-1/training/training/"

TEST_DIR = "/kaggle/input/nnfl-lab-1/testing/testing/"
IMAGE_WIDTH=400

IMAGE_HEIGHT=400

IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

IMAGE_CHANNELS=3



batch_size=32
filenames = os.listdir(TRAIN_DIR)

categories = []

for filename in filenames:

    category = filename.split('_')[0]

    if category == 'chair':

        categories.append(0)

    elif category == 'kitchen':

        categories.append(1)

    elif category == 'knife':

        categories.append(2)

    elif category == 'saucepan':

        categories.append(3)



df = pd.DataFrame({

    'filename': filenames,

    'category': categories

})



df.head()
df['category'].value_counts().plot.bar()
sample = random.choice(filenames)

image = load_img(TRAIN_DIR+sample)

plt.imshow(image)
df["category"] = df["category"].replace({0: 'chair', 1: 'kitchen', 2: 'knife', 3: 'saucepan'}) 
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)

train_df = train_df.reset_index(drop=True)

validate_df = validate_df.reset_index(drop=True)
train_df['category'].value_counts().plot.bar()
validate_df['category'].value_counts().plot.bar()
total_train = train_df.shape[0]

total_validate = validate_df.shape[0]



print("Total iamges in train set - {}".format(total_train))

print("Total iamges in validation set - {}".format(total_validate))
train_datagen = ImageDataGenerator(

    rotation_range=15,

    rescale=1./255,

    shear_range=0.1,

    zoom_range=0.2,

    horizontal_flip=True,

    width_shift_range=0.1,

    height_shift_range=0.1

)



train_generator = train_datagen.flow_from_dataframe(

    train_df, 

    TRAIN_DIR, 

    x_col='filename',

    y_col='category',

    target_size=IMAGE_SIZE,

    class_mode='categorical',

    batch_size=batch_size

)



validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(

    validate_df, 

    TRAIN_DIR, 

    x_col='filename',

    y_col='category',

    target_size=IMAGE_SIZE,

    class_mode='categorical',

    batch_size=batch_size

)
input = Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))



x1 = Conv2D(32, (3,3), activation='relu', padding='same')(input)

x1 = BatchNormalization()(x1)

x1 = MaxPooling2D(pool_size=(2, 2))(x1)

x1 = Dropout(0.25)(x1)



x1 = Conv2D(64, (3,3), activation='relu', padding='same')(x1)

x1 = BatchNormalization()(x1)

x1 = MaxPooling2D(pool_size=(2, 2))(x1)

x1 = Dropout(0.25)(x1)



x1 = Conv2D(128, (3,3), activation='relu', padding='same')(x1)

x1 = BatchNormalization()(x1)

x1 = MaxPooling2D(pool_size=(2, 2))(x1)

x1 = Dropout(0.25)(x1)



x1 = Conv2D(256, (3,3), activation='relu', padding='same')(x1)

x1 = BatchNormalization()(x1)

x1 = MaxPooling2D(pool_size=(2, 2))(x1)

x1 = Dropout(0.25)(x1)



x1 = Conv2D(512, (3,3), activation='relu', padding='same')(x1)

x1 = BatchNormalization()(x1)

x1 = MaxPooling2D(pool_size=(2, 2))(x1)

x1 = Dropout(0.25)(x1)



x1 = Conv2D(1024, (3,3), activation='relu', padding='same')(x1)

x1 = BatchNormalization()(x1)

x1 = MaxPooling2D(pool_size=(2, 2))(x1)

x1 = Dropout(0.25)(x1)

################################################################################



x2 = Conv2D(32, (5,5), activation='relu', padding='same')(input)

x2 = BatchNormalization()(x2)

x2 = MaxPooling2D(pool_size=(2, 2))(x2)

x2 = Dropout(0.25)(x2)



x2 = Conv2D(64, (5,5), activation='relu', padding='same')(x2)

x2 = BatchNormalization()(x2)

x2 = MaxPooling2D(pool_size=(2, 2))(x2)

x2 = Dropout(0.25)(x2)



x2 = Conv2D(128, (5,5), activation='relu', padding='same')(x2)

x2 = BatchNormalization()(x2)

x2 = MaxPooling2D(pool_size=(2, 2))(x2)

x2 = Dropout(0.25)(x2)



x2 = Conv2D(256, (5,5), activation='relu', padding='same')(x2)

x2 = BatchNormalization()(x2)

x2 = MaxPooling2D(pool_size=(2, 2))(x2)

x2 = Dropout(0.25)(x2)



x2 = Conv2D(512, (5,5), activation='relu', padding='same')(x2)

x2 = BatchNormalization()(x2)

x2 = MaxPooling2D(pool_size=(2, 2))(x2)

x2 = Dropout(0.25)(x2)



x2 = Conv2D(1024, (5,5), activation='relu', padding='same')(x2)

x2 = BatchNormalization()(x2)

x2 = MaxPooling2D(pool_size=(2, 2))(x2)

x2 = Dropout(0.25)(x2)

################################################################################



x3 = Conv2D(32, (7,7), activation='relu', padding='same')(input)

x3 = BatchNormalization()(x3)

x3 = MaxPooling2D(pool_size=(2, 2))(x3)

x3 = Dropout(0.25)(x3)



x3 = Conv2D(64, (7,7), activation='relu', padding='same')(x3)

x3 = BatchNormalization()(x3)

x3 = MaxPooling2D(pool_size=(2, 2))(x3)

x3 = Dropout(0.25)(x3)



x3 = Conv2D(128, (7,7), activation='relu', padding='same')(x3)

x3 = BatchNormalization()(x3)

x3 = MaxPooling2D(pool_size=(2, 2))(x3)

x3 = Dropout(0.25)(x3)



x3 = Conv2D(256, (7,7), activation='relu', padding='same')(x3)

x3 = BatchNormalization()(x3)

x3 = MaxPooling2D(pool_size=(2, 2))(x3)

x3 = Dropout(0.25)(x3)



x3 = Conv2D(512, (7,7), activation='relu', padding='same')(x3)

x3 = BatchNormalization()(x3)

x3 = MaxPooling2D(pool_size=(2, 2))(x3)

x3 = Dropout(0.25)(x3)



x3 = Conv2D(1024, (7,7), activation='relu', padding='same')(x3)

x3 = BatchNormalization()(x3)

x3 = MaxPooling2D(pool_size=(2, 2))(x3)

x3 = Dropout(0.25)(x3)

################################################################################



y = Concatenate()([x1, x2, x3])



y = Conv2D(1024, (3,3), activation='relu')(y)

y = BatchNormalization()(y)

y = MaxPooling2D(pool_size=(2, 2))(y)

y = Dropout(0.25)(y)



y = Flatten()(y)



y = Dense(2048, activation='relu')(y)

# y = BatchNormalization()(y)

# y = Dropout(0.25)(y)



y = Dense(96, activation='relu')(y)

# y = BatchNormalization()(y)

# y = Dropout(0.25)(y)



output = Dense(4, activation='softmax')(y)



model = keras.Model(inputs=input, outputs=output, name="custom_model")
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

earlystop = EarlyStopping(patience=10)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=2, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00000000001)

callbacks = [earlystop, learning_rate_reduction]



model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.001), metrics=['accuracy'])
epochs=60





history = model.fit_generator(

    train_generator, 

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=total_validate//batch_size,

    steps_per_epoch=total_train//batch_size,

    callbacks=callbacks

)
model.save_weights("model.h5")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

ax1.plot(history.history['loss'], color='b', label="Training loss")

ax1.plot(history.history['val_loss'], color='r', label="Validation loss")

ax1.set_xticks(np.arange(1, epochs, 1))

ax1.set_yticks(np.arange(0, 1, 0.1))



ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")

ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

ax2.set_xticks(np.arange(1, epochs, 1))



legend = plt.legend(loc='best', shadow=True)

plt.tight_layout()

plt.show()


test_filenames = os.listdir(TEST_DIR)

test_df = pd.DataFrame({

    'filename': test_filenames

})

nb_samples = test_df.shape[0]
test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(

    test_df, 

    TEST_DIR, 

    x_col='filename',

    y_col=None,

    class_mode=None,

    target_size=IMAGE_SIZE,

    batch_size=batch_size,

    shuffle=False

)
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
test_df['category'] = np.argmax(predict, axis=-1)

label_map = dict((v,k) for k,v in train_generator.class_indices.items())

test_df['category'] = test_df['category'].replace(label_map)
test_df.head()
test_df['category'] = test_df['category'].replace({ 'chair': 0, 'kitchen': 1, 'knife': 2, 'saucepan': 3})
test_df['category'].value_counts().plot.bar()
sample_test = test_df.head(10)

sample_test.head()

plt.figure(figsize=(12, 24))

for index, row in sample_test.iterrows():

    filename = row['filename']

    category = row['category']

    img = load_img(TEST_DIR+filename, target_size=IMAGE_SIZE)

    plt.subplot(6, 3, index+1)

    plt.imshow(img)

    plt.xlabel(filename + '(' + "{}".format(category) + ')' )

plt.tight_layout()

plt.show()
submission_df = test_df.copy()

submission_df['id'] = submission_df['filename']

submission_df['label'] = submission_df['category']

submission_df.drop(['filename', 'category'], axis=1, inplace=True)

submission_df.to_csv('submission.csv', index=False)
submission_df.head()
from IPython.display import HTML 

import pandas as pd 

import numpy as np

import base64 





def create_download_link(df, title = "Download CSV file",filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode()) 

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(submission_df)