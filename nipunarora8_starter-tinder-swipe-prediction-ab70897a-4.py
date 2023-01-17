from keras.preprocessing.image import ImageDataGenerator #for data generators

import matplotlib.pyplot as plt

import os # accessing directory structure
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#since we have less data we can create more data using image data generators



train_datagen = ImageDataGenerator(

    rotation_range=25, 

    width_shift_range=0.1,

    height_shift_range=0.1, 

    shear_range=0.2, 

    zoom_range=0.2,

    horizontal_flip=True, 

    fill_mode="nearest",

    rescale=1./255)



test_datagen = ImageDataGenerator(rescale=1./255)
batch_size=32 #add your own batch size
train = train_datagen.flow_from_directory(

    '/kaggle/input/Dataset/Train',

    target_size=(224,224),

    batch_size=batch_size,

    class_mode='binary')



val = test_datagen.flow_from_directory(

    '/kaggle/input/Dataset/Val',

    target_size=(224,224),

    batch_size=batch_size,

    class_mode='binary')



test = test_datagen.flow_from_directory(

    '/kaggle/input/Dataset/Test',

    target_size=(224,224),

    batch_size=batch_size,

    class_mode='binary')
for i in range(10):

    if train[0][1][i]==1.0:

        plt.title("Like")

    else:

        plt.title("Dislike")

    plt.imshow(train[0][0][i])

    plt.show()