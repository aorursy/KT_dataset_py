# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
import tensorflow as tf

tf.__version__ 
# Import The Libraries 



from tensorflow.keras.layers import Input, Lambda, Dense, Flatten

from tensorflow.keras.models import Model

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

from tensorflow.keras.preprocessing import image

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

from tensorflow.keras.models import Sequential



import numpy as np

from glob import glob

import matplotlib.pyplot as plt
# Paths

train_Path = '../input/car-brand-images-dataset/Train'

test_Path = '../input/car-brand-images-dataset/Test'
# Set Resize variable

IMAGE_SIZE = [224, 224] # This is my desired image size... and also ResNet50 accepts image of 224*224.
resnet = ResNet50(

    input_shape = IMAGE_SIZE + [3], # Making the image into 3 Channel, so concating 3.

    weights = 'imagenet', # Default weights.

    include_top = False   # 

)
resnet.summary()
for layer in resnet.layers:

    layer.trainable = False

    

# This will let us use the default weights used by the imagenet.    
# Usefule for getting number of output classes.

# folders = glob('../input/car-brand-images-dataset/Train/*')

folders = glob(train_Path + '/*')

folders
car_label = ['mercedes', 'audi', 'lamborghini']
# Set the flatten layer.

x = Flatten() (resnet.output)
prediction = Dense(len(folders), activation = 'softmax')(x)
# Create a model Object

model = Model(inputs = resnet.input, outputs = prediction)
model.summary()
model.compile (

    loss = 'categorical_crossentropy',

    optimizer = 'adam',

    metrics = ['accuracy']

)
# Use the Image Data Generator



# from tensorflow.keras.proprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(

    rescale = 1./255,

    shear_range = 0.2,

    zoom_range = 0.2,

    horizontal_flip = True

)



test_datagen = ImageDataGenerator(

    rescale = 1./255

)
training_set = train_datagen.flow_from_directory(

    train_Path,

    target_size = IMAGE_SIZE,

    batch_size = 32,

    class_mode = 'categorical' # As we have more than 2 so using categorical.. for 2 we might have used binary.

)
test_set = train_datagen.flow_from_directory(

    test_Path,

    target_size = IMAGE_SIZE,

    batch_size = 32,

    class_mode = 'categorical'

)
# Fir the model.



history = model.fit_generator(

    training_set,

    validation_data = test_set,

    epochs = 50,

    steps_per_epoch = len(training_set),

    validation_steps = len(test_set)

)
# Plot the Loss



plt.plot(history.history['loss'], label = 'train_loss')

plt.plot(history.history['val_loss'], label ='val loss')

plt.legend()

plt.show()

# plt.savefig('LossVal_loss')
# Plot the Accuracy

plt.plot(history.history['accuracy'], label = 'train accuracy')

plt.plot(history.history['val_accuracy'], label ='val accuracy')

plt.legend()

plt.show()

# plt.savefig('valAccuracy')
# Save it as a h5 file

from tensorflow.keras.models import load_model



model.save('car_brand_clf_resnet50.h5')
prediction = model.predict(test_set)
prediction
prediction = np.argmax(prediction, axis = 1)

prediction
unseen_data_path = '../input/unseen-data/'
# img = image.load_img('../input/unseen-data/audi_1.jpg', target_size = IMAGE_SIZE)



img = image.load_img(unseen_data_path + 'audi_1.jpg', target_size = IMAGE_SIZE)

img
x = image.img_to_array(img)

x
x.shape
x = x / 255
x = np.expand_dims(x, axis = 0)

img_data = preprocess_input(x)

x.shape, img_data.shape
model.predict(img_data)
a = np.argmax(model.predict(img_data), axis = 1)

a
if a == 0:

    print("Its Audi")

elif a == 1:

    print("Its Lamorghini")

else:

    print("Its Mercedes")
car_label
def unseen_data_test(path, image_name, model):

    img = image.load_img(path + image_name, target_size = IMAGE_SIZE)

    print('Original Image')

#     print(img)

    plt.imshow(img)

    x = image.img_to_array(img)

    x = x / 255

    x = np.expand_dims(x, axis = 0)

    img_data = preprocess_input(x)

    a = np.argmax(model.predict(img_data), axis = 1)

    

    if a == 0:

        print("Its Audi")

    elif a == 1:

        print("Its Mercedes")

    else:

        print("Its Lamorghini ")
unseen_data_test(unseen_data_path, 'lamborghini_1.jpg', model)
unseen_data_test(unseen_data_path, 'lamborghini_2.jpg', model)
unseen_data_test(unseen_data_path, 'lamborghini_4.jpg', model)
unseen_data_test(unseen_data_path, 'audi_2.jpg', model)
unseen_data_test(unseen_data_path, 'audi_3.jpg', model)
# for 

# img = image.load_img(unseen_data_path + 'audi_1.jpg', target_size = IMAGE_SIZE)
