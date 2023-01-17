# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
img_dir = '../input/image-classification/images/images'

val_dir = '../input/image-classification/validation/validation'
! pip install -q tf-nightly-gpu
import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt

from tensorflow.keras import layers

from tensorflow.keras.models import Sequential

tf.__version__
# change as you want

img_width, img_height = 224, 224

batch_size = 50
train_ds = tf.keras.preprocessing.image_dataset_from_directory(

    directory=img_dir,

    validation_split=0.2,

    subset="training",

    seed=1007,

    image_size=(img_width, img_height),

    batch_size=batch_size,

)
class_names = train_ds.class_names

print(class_names)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(

    directory=img_dir,

    validation_split=0.2,

    subset="validation",

    seed=1007,

    image_size=(img_width, img_height),

    batch_size=batch_size,

)
import matplotlib.pyplot as plt



plt.figure(figsize=(10, 10))

for images, labels in train_ds.take(1):

    for i in range(9):

        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))

        plt.title(class_names[labels[i]])

        plt.axis("off")
resize_and_rescale = tf.keras.Sequential([

  layers.experimental.preprocessing.Rescaling(1./255)

])
data_augmentation = tf.keras.Sequential([

  layers.experimental.preprocessing.RandomFlip("horizontal"),

  layers.experimental.preprocessing.RandomRotation(0.1),

])
plt.figure(figsize=(10, 10))

for images, _ in train_ds.take(1):

    for i in range(6):

        augmented_images = data_augmentation(images)

        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(augmented_images[0].numpy().astype("uint8"))

        plt.axis("off")
AUTOTUNE = tf.data.experimental.AUTOTUNE



def prepare(ds,augment=False):

    # Resize and rescale all datasets 

    ds = ds.map(lambda x, y: (resize_and_rescale(x), y),num_parallel_calls=AUTOTUNE)



    # Use data augmentation only on the training set

    if augment:

        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),num_parallel_calls=AUTOTUNE)



    # Use buffered prefecting on all datasets

    return ds.prefetch(buffer_size=AUTOTUNE)
train_ds = prepare(train_ds,augment=True)

val_ds = prepare(val_ds)
base_model = tf.keras.applications.Xception(input_shape = (img_width, img_height, 3), include_top = False, weights = 'imagenet')
base_model.trainable = False
base_model.summary()
len(base_model.layers)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

prediction_layer = tf.keras.layers.Dense(4)
model = tf.keras.Sequential([

  base_model,

  global_average_layer,

  prediction_layer

])
model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])

model.summary()

len(model.trainable_variables)
epochs=8

callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(train_ds,validation_data=val_ds,epochs=epochs,callbacks=[callbacks])
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss=history.history['loss']

val_loss=history.history['val_loss']



plt.figure(figsize=(20, 8))

plt.subplot(1, 2, 1)

plt.plot( acc, label='Training Accuracy')

plt.plot( val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot( loss, label='Training Loss')

plt.plot( val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()
saved_model_path = "/tmp/saved_google_image_model"
model.save(saved_model_path)
model=tf.keras.models.load_model(saved_model_path)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(

    directory=val_dir,

    validation_split=0.9999,

    subset="validation",

    seed=1007,

    image_size=(img_width, img_height),

    batch_size=batch_size,

)

test_ds = prepare(test_ds)
# Evaluating model on testing data

loss, acc = model.evaluate(test_ds)

print("Accuracy", acc)

for image_batch, labels_batch in test_ds:

    predictions = model.predict_on_batch(image_batch)

    predictions = tf.nn.softmax(predictions)

    predictions=predictions.numpy()

    pred =[]

    for i in predictions:

        pred.append(np.argmax(i))



    print('Predictions:\n',pred)

    print('Labels:\n', labels_batch.numpy())
test_dir = '../input/image-classification/test/test/classify'

test_img=os.listdir(test_dir)
def read_image(img_path):

    img = keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))

    img_array = keras.preprocessing.image.img_to_array(img)

    img_array = tf.expand_dims(img_array, 0) # Create a batch

    img_array /= 255.

    return img_array,img
def test_single_image(img_path):

    img_array,img=read_image(img_path)

    predictions = model.predict(img_array)

    score = tf.nn.softmax(predictions[0])

    print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)],

                                                                                          100 * np.max(score)))

    return img,score
plt.figure(figsize=(15, 15))

for i in range(len(test_img)):

    img_path=test_dir+'/'+test_img[i]

    img,score=test_single_image(img_path)



    ax = plt.subplot(4, 3,i+1)

    plt.imshow(img)

    plt.title(class_names[np.argmax(score)])

    plt.axis("off")
def test_external_image(name,image_url):

    plt.figure(figsize=(5, 5))

    img_path = tf.keras.utils.get_file(name, origin=image_url)

    img,score=test_single_image(img_path)

    

    plt.imshow(img)

    plt.title(class_names[np.argmax(score)])

    plt.axis("off")
image_url1 = "https://images2.minutemediacdn.com/image/upload/c_crop,h_1126,w_2000,x_0,y_181/f_auto,q_auto,w_1100/v1554932288/shape/mentalfloss/12531-istock-637790866.jpg"

test_external_image("image_url_food",image_url1)
image_url2="https://www.rnz.co.nz/assets/news_crops/60885/eight_col_32917696_l.jpg"

test_external_image("image_url2",image_url2)

image_url3 = "https://cdn.britannica.com/02/210202-050-D644C84B/Horyu-ji-Temple-Ikaruga-Nara-Japan-Buddhism.jpg"

test_external_image("image_url3",image_url3)

image_url4="https://img.etimg.com/thumb/msid-66129697,width-640,resizemode-4,imgsize-342241/how-to-get-your-trips-sponsored.jpg"

test_external_image("travel_image_url",image_url4)
image_url5="https://www.discoversouthafrica.net/wp-content/uploads/2018/04/01artandculture.jpg"

test_external_image("artandculture_image_url",image_url5)
