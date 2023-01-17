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
! pip install  tf-nightly
import tensorflow as tf

from tensorflow import keras

tf.__version__
# change as you want

image_size = (180, 180)

batch_size = 50
train_ds = tf.keras.preprocessing.image_dataset_from_directory(

    directory=img_dir,

    validation_split=0.2,

    subset="training",

    seed=1007,

    image_size=image_size,

    batch_size=batch_size,

)
train_ds.class_names
val_ds = tf.keras.preprocessing.image_dataset_from_directory(

    directory=img_dir,

    validation_split=0.2,

    subset="validation",

    seed=1007,

    image_size=image_size,

    batch_size=batch_size,

)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(

    directory=val_dir,

    validation_split=0.9999,

    subset="validation",

    seed=1007,

    image_size=image_size,

    batch_size=batch_size,

)
# put your code here 
import numpy as np

import os

import PIL

import PIL.Image
class_names = train_ds.class_names

import matplotlib.pyplot as plt



plt.figure(figsize=(16, 10))

for images, labels in train_ds.take(1):

    for i in range(9):

        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))

        plt.title(class_names[int(labels[i])])

        plt.axis("off")

# put your code here 
from tensorflow.keras import datasets, layers, models,Sequential

img_height = 180

img_width = 180

num_classes = 4


model = Sequential([ layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

 

    layers.Conv2D(16, 3, padding='same', activation='relu'),

    layers.MaxPooling2D(),

    layers.Dropout(0.2),

    

    layers.Conv2D(32, 3, padding='same', activation='relu'),

    layers.MaxPooling2D(),

    layers.Dropout(0.2),

    

    layers.Conv2D(32, 3, padding='same', activation='relu'),

    layers.MaxPooling2D(),

    layers.Dropout(0.2),

    

    

    layers.Conv2D(64, 3, padding='same', activation='relu'),

    layers.MaxPooling2D(),

    

    layers.Flatten(),

    layers.Dense(128, activation='relu'),

    layers.Dense(num_classes)

])
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])



model.summary()
from keras.callbacks import ModelCheckpoint, EarlyStopping
# simple early stopping



epochs=30

verbose=3



es = EarlyStopping(monitor='val_loss', mode='min', verbose=verbose, patience=50)

mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=verbose, save_best_only=True)
# fit model

history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,verbose=verbose,callbacks=[es, mc])
# put your code here 
plt.figure(figsize=(20, 10))



plt.subplot(1, 2, 1)

plt.plot(history.history['accuracy'], label='Training accuracy')

plt.plot(history.history['val_accuracy'], label = 'Validation accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)



plt.plot(history.history['loss'], label='Training Loss')

plt.plot(history.history['val_loss'], label = 'Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()
# put your code here 
# load the saved model

from keras.models import load_model

model = load_model('best_model.h5')
model.save_weights('best_model.h5')


#model=load_model('best_model.h5')
test_loss, test_acc = model.evaluate(test_ds , verbose=verbose) 

print( 'Test Accuracy: %.3f' % (test_acc))
test_dir='../input/image-classification/test/test/classify'

os.listdir(test_dir)
#from keras.models import load_model





#model = load_model('class_img.h5')



#saved_model.compile(optimizer='adam',

#              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

#              metrics=['accuracy'])
from keras.preprocessing import image



plt.figure(figsize=(16, 10))

for i in range(10):

    img = tf.keras.preprocessing.image.load_img(test_dir+'/'+os.listdir(test_dir)[i],target_size=image_size)

    ax = plt.subplot(3, 4, i + 1)

    plt.imshow(img)

    

    img = image.img_to_array(img)

    img = np.expand_dims(img, axis = 0)

    predction= np.argmax(model.predict(img))

    #print('image class is :',i,class_names[predction])

    plt.title(class_names[predction])

    plt.axis("off")

   
# put your code here 
url = 'https://beforetravelling.com/wp-content/uploads/2020/01/Adventure-Travel-1.jpg'

path = tf.keras.utils.get_file('Adventure-Travel-1', origin=url)

plt.figure(figsize=(7, 7))

img = keras.preprocessing.image.load_img(

    path, target_size=(img_height, img_width)

)

plt.imshow(img)

img_array = keras.preprocessing.image.img_to_array(img)

img_array = tf.expand_dims(img_array, 0) # Create a batch



predictions = model.predict(img_array)

score = tf.nn.softmax(predictions[0])



print(

    "This image most likely belongs to {} with a {:.2f} percent confidence."

    .format(class_names[np.argmax(score)], 100 * np.max(score))

)


url = 'https://upload.wikimedia.org/wikipedia/commons/d/d8/MoscowMuseum_of_Architecture.JPG'

path = tf.keras.utils.get_file('MoscowMuseum_of_Architecture', origin=url)



img = keras.preprocessing.image.load_img(

    path, target_size=(img_height, img_width)

)

plt.figure(figsize=(7, 7))

plt.imshow(img)

img_array = keras.preprocessing.image.img_to_array(img)

img_array = tf.expand_dims(img_array, 0) # Create a batch



predictions = model.predict(img_array)

score = tf.nn.softmax(predictions[0])



print(

    "This image most likely belongs to {} with a {:.2f} percent confidence."

    .format(class_names[np.argmax(score)], 100 * np.max(score))

)



url = 'https://images2.minutemediacdn.com/image/upload/c_crop,h_1126,w_2000,x_0,y_181/f_auto,q_auto,w_1100/v1554932288/shape/mentalfloss/12531-istock-637790866.jpg'

path = tf.keras.utils.get_file('12531-istock-637790866', origin=url)



img = keras.preprocessing.image.load_img(

    path, target_size=(img_height, img_width)

)

plt.figure(figsize=(7, 7))

plt.imshow(img)

img_array = keras.preprocessing.image.img_to_array(img)

img_array = tf.expand_dims(img_array, 0) # Create a batch



predictions = model.predict(img_array)

score = tf.nn.softmax(predictions[0])



print(

    "This image most likely belongs to {} with a {:.2f} percent confidence."

    .format(class_names[np.argmax(score)], 100 * np.max(score))

)

url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Taj_Mahal_%28Edited%29.jpeg/1024px-Taj_Mahal_%28Edited%29.jpeg'

path = tf.keras.utils.get_file('Taj_Mahal_%28Edited%29.jpeg/1024px-Taj_Mahal_%28Edited%29.jpeg', origin=url)



img = keras.preprocessing.image.load_img(

    path, target_size=(img_height, img_width)

)

plt.figure(figsize=(7, 7))

plt.imshow(img)

img_array = keras.preprocessing.image.img_to_array(img)

img_array = tf.expand_dims(img_array, 0) # Create a batch



predictions = model.predict(img_array)

score = tf.nn.softmax(predictions[0])



print(

    "This image most likely belongs to {} with a {:.2f} percent confidence."

    .format(class_names[np.argmax(score)], 100 * np.max(score))

)
url = 'https://www.kitchensanctuary.com/wp-content/uploads/2020/04/Chicken-Fried-Rice-square-FS-.jpg'

path = tf.keras.utils.get_file('Chicken-Fried-Rice-square-FS-.jpg', origin=url)



img = keras.preprocessing.image.load_img(

    path, target_size=(img_height, img_width)

)

plt.figure(figsize=(7, 7))

plt.imshow(img)

img_array = keras.preprocessing.image.img_to_array(img)

img_array = tf.expand_dims(img_array, 0) # Create a batch



predictions = model.predict(img_array)

score = tf.nn.softmax(predictions[0])



print(

    "This image most likely belongs to {} with a {:.2f} percent confidence."

    .format(class_names[np.argmax(score)], 100 * np.max(score))

)
