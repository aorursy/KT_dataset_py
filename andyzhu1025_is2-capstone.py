# Importing all relevant packages.

# Numpy is for mathematical computation,

# PIL is for accessing images,

# TensorFlow/Keras is for training our model.

# IMPORTANT



import numpy as np

from PIL import Image

from tensorflow.keras.applications import ResNet50

from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D

from tensorflow.keras.preprocessing import image

from tensorflow.keras.preprocessing.image import ImageDataGenerator



# The model will predict 7 different types of animals:

# Buffalo, Cheetahs, Elephants, Giraffes, Male Lions, Rhinos, Zebras.

# I was only able to allocate enough resources to support these animals.

num_classes = 7



# We load the convolutional neural network ResNet50. This will help us extract features from images.

resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'



# ResNet50 has thousands of classes; we are only interested in the aforementioned 7.

# Thus, we take off the final layer.

model = Sequential()

model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))

model.add(Dense(num_classes, activation='softmax'))

model.layers[0].trainable = False

model.compile(optimizer='sgd', 

                     loss='categorical_crossentropy', 

                     metrics=['accuracy'])



# Our data will be from the "capstonedata" dataset I created.

# Here, we load this data into a format our model can read.

# One dataset is for training, the other is for validation;

# We use training to fit the model, validation for analyzing.

data_generator = ImageDataGenerator(preprocess_input)



train = data_generator.flow_from_directory(

                                        directory="../input/capstonedata/CapstoneData/Train",

                                        target_size=(224, 224),

                                        batch_size=10,

                                        class_mode='categorical')



validation = data_generator.flow_from_directory(

                                        directory="../input/capstonedata/CapstoneData/Validation",

                                        target_size=(224, 224),

                                        class_mode='categorical')



# Now we train the model itself over each of our images.

# The "fit_generator" function gives us helfpul statistics about accuracy.

# I set "epochs" to 10. Increasing that number usually increases

# accuracy, but it takes more time to train as a result.

fit = model.fit_generator(train,

                                       epochs = 10,

                                       steps_per_epoch=133,

                                       validation_data=validation,

                                       validation_steps=1)



# The user may input desired images and see 

# what the model predicts. This function returns

# the model's guess and displays the image.

def predict_img(img_path):

    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    preds = model.predict(x)

    d = dict(zip(preds.tolist()[0],(["Buffalo","Cheetah","Elephant","Giraffe","Male Lion","Rhino","Zebra"])))

    im = Image.open(img_path)

    display(im)

    return "Guess: " + d[max(d.keys())]

predict_img("../input/capstonedata/CapstoneData/Validation/Zebra/375.jpg")
predict_img("../input/capstonedata/CapstoneData/Validation/Cheetah/3224140.jpg")
predict_img("../input/capstonedata/CapstoneData/Validation/Buffalo/373.jpg")
predict_img("../input/capstonedata/CapstoneData/Validation/Elephant/326.jpg")