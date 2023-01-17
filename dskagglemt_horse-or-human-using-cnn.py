import keras

from keras.preprocessing.image import ImageDataGenerator # for data augmentation 



import matplotlib.pyplot as plt
keras.__version__
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)
for dirname, _, filenames in os.walk('/kaggle/input'):

    Image_Count = 0

#     print(dirname)

    

    for file in filenames:

        Image_Count += 1

    

    if Image_Count > 0:

        print('Total Files in directory {} is {}'.format(dirname, Image_Count))
train_path = '/kaggle/input/horses-or-humans-dataset/horse-or-human/train'

val_path = '/kaggle/input/horses-or-humans-dataset/horse-or-human/validation'
training_datagen = ImageDataGenerator(

    rescale = 1./255,

    rotation_range = 40,

    width_shift_range = 0.2,

    height_shift_range = 0.2,

    zoom_range = 0.2,

    horizontal_flip = True,

    fill_mode = 'nearest'



)
training_data = training_datagen.flow_from_directory(

    directory = train_path,

    target_size = (150, 150),   # Setting the size of output images to have in same size.

    batch_size = 32,

    class_mode = 'binary',

    shuffle=True,

    seed=42

)
training_data.class_indices
training_data
# Doing the same for validation dataset.

# But here we do not need to generate the images for Validation, so we just use rescale.

valid_datagen = ImageDataGenerator(rescale = 1./255)



valid_data = valid_datagen.flow_from_directory(

    directory = val_path,

    target_size = (150, 150),   # Setting the size of output images to have in same size.

    batch_size = 32,

    class_mode = 'binary'

)
def plotImages(images_arr):

    fig, axes = plt.subplots(1,5,figsize = (20, 20))

    axes = axes.flatten()

    

    for img, ax in zip(images_arr, axes):

        ax.imshow(img)

        

    plt.tight_layout()

    plt.show()
images_to_show = [training_data[0][0][0] for i in range(5)] # taking 5 images.

plotImages(images_to_show)
cnn_model = keras.models.Sequential(

    [

        keras.layers.Conv2D(filters = 32, kernel_size = 3, input_shape = [150, 150, 3]),

        keras.layers.MaxPooling2D(pool_size = (2,2)),

        keras.layers.Conv2D(filters = 64, kernel_size = 3),

        keras.layers.MaxPooling2D(pool_size = (2,2)),

        keras.layers.Conv2D(filters = 128, kernel_size = 3),

        keras.layers.MaxPooling2D(pool_size = (2,2)),

        keras.layers.Conv2D(filters = 256, kernel_size = 3),

        keras.layers.MaxPooling2D(pool_size = (2,2)),

        

        keras.layers.Dropout(0.5),

        

        # Neural Network Building

        keras.layers.Flatten(),

        keras.layers.Dense(units = 128, activation = 'relu'), # Input Layer

        keras.layers.Dropout(0.1),

        keras.layers.Dense(units = 256, activation = 'relu'), # Hidden Layer

        keras.layers.Dropout(0.25),

        keras.layers.Dense(units = 2, activation = 'softmax'), # Output Layer

    ]

)
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint
cnn_model.compile(

    optimizer = Adam(lr=0.0001),

    loss = 'sparse_categorical_crossentropy',

    metrics = ['accuracy']

)
# model_path = '/kaggle/input/output/horse_human_model.h5'  

model_path = '/kaggle/working/horse_human_model.h5'

# model_path = '/kaggle/input/horse_human_model.h5'

checkpoint = ModelCheckpoint(model_path, monitor = 'val_accuracy', verbose = 1, save_best_only = True, mode = 'max')

callbacks_list = [checkpoint]
history = cnn_model.fit(

    training_data,

    epochs = 100,

    verbose = 1,

    validation_data = valid_data,

    callbacks = callbacks_list

)
# Summarize the accuracy.



plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])



plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('epoch or iteration')

plt.legend(["train", 'valid'], loc = 'upper left')

plt.show()
# Summarize the Loss



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])



plt.title('Model Loss')

plt.ylabel('Accuracy')

plt.xlabel('epoch or iteration')

plt.legend(["train", 'valid'], loc = 'upper left')

plt.show()
import numpy as np

import keras

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing import image
import os

for dirname, _, filenames in os.walk('/kaggle'):

    print(dirname)
model_path = '/kaggle/working/horse_human_model.h5'
trained_model = keras.models.load_model(model_path)
# Define path for Horse Image

horse1 = '/kaggle/input/hh-test-images/horse1.jpg'

horse2 = '/kaggle/input/hh-test-images/horse2.jpg'

horse3 = '/kaggle/input/hh-test-images/horse3.jpg'

horse4 = '/kaggle/input/hh-test-images/horse4.jpg'



# Define path for Human Images

human1 = '/kaggle/input/hh-test-images/human.jpg'

human2 = '/kaggle/input/hh-test-images/human2.jpg'

human3 = '/kaggle/input/hh-test-images/human3.jpg'

human4 = '/kaggle/input/hh-test-images/human4.jpg'

human5 = '/kaggle/input/hh-test-images/human5.jpg'

def pred_horse_human(model, horse_human):

    test_image = image.load_img(horse_human, target_size = (150, 150))

    test_image = image.img_to_array(test_image)/255

    test_image = np.expand_dims(test_image, axis = 0)

    

    result = model.predict(test_image).round(3)

    

    pred = np.argmax(result)

    print(result, "-->", pred)

    

    if pred == 0:

        print("Predicted as a HORSE")

    else:

        print("Predicted as a HUMAN")

        
for horse_human in [horse1, horse2, horse3, horse4, human1, human2, human3, human4, human5]:

    pred_horse_human(trained_model, horse_human)
import cv2
def pred_horse_human_image(model, horse_human):

    test_image = image.load_img(horse_human, target_size = (150, 150))

    test_image = image.img_to_array(test_image)/255

    test_image = np.expand_dims(test_image, axis = 0)

    

    result = model.predict(test_image).round(3)

    

    pred = np.argmax(result)

#     print(result, "-->", pred)

    

    if pred == 0:

        prediction = "HORSE"

    else:

        prediction = "HUMAN"



    img_array = cv2.imread(horse_human, cv2.IMREAD_GRAYSCALE)

    plt.imshow(img_array, cmap = 'gray')

    

    plt.axis('off')

    plt.title("PREDICTED AS : " + prediction)

    

    plt.show()
pred_horse_human_image(trained_model, horse1)
for horse_human in [horse1, horse2, horse3, horse4, human1, human2, human3, human4, human5]:

    pred_horse_human_image(trained_model, horse_human)