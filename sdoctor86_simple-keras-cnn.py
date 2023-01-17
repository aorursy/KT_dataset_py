import tensorflow as tf

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPooling2D

from tensorflow.keras.optimizers import Adam

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import pathlib



DEV_MODE = False

data_root_path = pathlib.Path("../input")
# will feed through data set map

def load_and_preprocess_image(path):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize(image, [32, 32])

    

    image /= 255.0  # normalize to [0,1] range

    return image
# augmentations will feed through training data set

def random_bright(image):

    return tf.image.random_brightness(image, 0.12)

            

def random_contrast(image):

    return tf.image.random_contrast(image, 0.9, 1.1)



def augment_image(image,label):

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_flip_up_down(image)

    image = random_contrast(image)

    image = random_bright(image)

    return image,label
# read training csv

train_df = pd.read_csv(data_root_path/'train.csv', dtype={'id': 'str', 'has_cactus': np.int32})

DATASET_SIZE = len(train_df)

print("n =", DATASET_SIZE)
# fix distribution in training data since there are more 1's than 0's

# if there are 75% 1's then by guessing 1 we are already at 75% accuracy

no_cactus, yes_cactus = train_df.has_cactus.value_counts().sort_values().values

no_multiplier = int(yes_cactus/no_cactus)-1

no_cactus_rows = train_df[train_df.has_cactus == 0]

for i in range(no_multiplier):

    train_df = train_df.append(no_cactus_rows)
ROOT_TRAIN_PATH = '../input/train/train/'

train_paths = ROOT_TRAIN_PATH + train_df['id']
# Dataset

path_ds = tf.data.Dataset.from_tensor_slices(train_paths)

image_ds = path_ds.map(load_and_preprocess_image)

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_df['has_cactus'].values, tf.int32))
IMAGE_SHAPE = (32, 32, 3)



# batching

image_label_ds = tf.data.Dataset.zip((image_ds, label_ds)).shuffle(DATASET_SIZE)



BATCH_SIZE = 16



if DEV_MODE:

    train_size, val_size = int(0.7 * DATASET_SIZE), int(0.3 * DATASET_SIZE)

else:

    train_size, val_size = DATASET_SIZE, 0



train_ds = image_label_ds.map(augment_image).take(train_size).batch(BATCH_SIZE).repeat()

val_ds = image_label_ds.skip(train_size).batch(BATCH_SIZE).repeat()
def get_model():

    model = Sequential()

    

    model.add(Conv2D(32, (2, 2),  input_shape=IMAGE_SHAPE,  activation='relu'))

    model.add(Conv2D(64, (3, 3),  activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    

    model.add(Conv2D(128, (4, 4),  activation='relu'))

    model.add(Conv2D(64, (5, 5), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dense(16, activation='relu'))

    model.add(Dense(1, activation="sigmoid"))

    

    return model



model = get_model()



model.compile(optimizer='adam', 

              loss='binary_crossentropy',

              metrics=['acc'])



model.summary()
# lets training re-augment images multiple times per epoch

train_steps = 2*(train_size//BATCH_SIZE)

val_steps = val_size//BATCH_SIZE



if val_steps == 0:

    val_steps = None

    val_ds = None



history = model.fit(train_ds,

                    steps_per_epoch=train_steps, 

                    epochs=50,

                    validation_data=val_ds, 

                    validation_steps=val_steps,

                    callbacks=None)
#plot accuracy

plt.plot(history.history['acc'], label="train acc")

if DEV_MODE:

    plt.plot(history.history['val_acc'], label="val acc")

plt.legend()

plt.show()
#evaluate on non augmented images, should not be worse

training_eval = tf.data.Dataset.zip((image_ds, label_ds)).shuffle(DATASET_SIZE)

training_eval = training_eval.take(DATASET_SIZE).batch(DATASET_SIZE)

model.evaluate(training_eval, steps=1)
#test dataset setup

test_paths = [path for path in sorted(pathlib.Path('../input/test/test/').glob('*.jpg'))]

TEST_SIZE = len(test_paths)

test_path_ds = tf.data.Dataset.from_tensor_slices([str(path)for path in test_paths])

test_image_ds = test_path_ds.map(load_and_preprocess_image)

test_image_ds = test_image_ds.take(TEST_SIZE).batch(TEST_SIZE)
#predictions

preds=model.predict(test_image_ds, steps = 1)
#write submission csv

test_df=pd.DataFrame({'id': [path.name for path in test_paths] })

test_df['has_cactus']=preds

test_df.to_csv("submission.csv",index=False)