import numpy as np

import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

from keras.layers import Input, Conv2D, Flatten, MaxPooling2D, Dense, BatchNormalization, DepthwiseConv2D, LeakyReLU, Add, GlobalMaxPooling2D

from keras.models import Model

from skimage.transform import resize

from PIL import Image, ImageOps



import glob, os, random
base_path = '../input/cropped images/'



img_list = glob.glob(os.path.join(base_path, '*/*.jpg'))



print(len(img_list))
for i, img_path in enumerate(random.sample(img_list, 6)):

    img = load_img(img_path)

    

    img = img_to_array(img, dtype=np.uint8)

    

    old_size = img.shape[:2]

    ratio = 200. / max(old_size)

    new_size = tuple([int(x*ratio) for x in old_size])

    

    img = resize(img, output_shape=new_size, mode='edge', preserve_range=True).astype(np.uint8)

    

    delta_w = 200 - new_size[1]

    delta_h = 200 - new_size[0]

    padding = ((delta_h//2, delta_h-(delta_h//2)), (delta_w//2, delta_w-(delta_w//2)), (0, 0))

    

    img = np.pad(img, padding, 'edge')



    plt.subplot(2, 3, i+1)

    plt.imshow(img.squeeze())
def resize_pad(img):

    old_size = img.shape[:2]

    ratio = 200. / max(old_size)

    new_size = tuple([int(x*ratio) for x in old_size])

    

    img = resize(img, output_shape=new_size, mode='edge', preserve_range=True)

    

    delta_w = 200 - new_size[1]

    delta_h = 200 - new_size[0]

    padding = ((delta_h//2, delta_h-(delta_h//2)), (delta_w//2, delta_w-(delta_w//2)), (0, 0))

    

    img = np.pad(img, padding, 'edge')

    

    return img



def preprocessing_train(x):

    x = resize_pad(x)

    return x



def preprocessing_val(x):

    x = resize_pad(x)

    return x



train_datagen = ImageDataGenerator(

    preprocessing_function=preprocessing_train,

    rescale=1./255,

    zoom_range=0.05,

    width_shift_range=0.05,

    height_shift_range=0.05,

    horizontal_flip=True,

    vertical_flip=True,

    rotation_range=90,

    validation_split=0.1

)



test_datagen = ImageDataGenerator(

    preprocessing_function=preprocessing_val,

    rescale=1./255,

    validation_split=0.1

)



train_generator = train_datagen.flow_from_directory(

    base_path,

    color_mode='grayscale',

    target_size=(200, 200),

    batch_size=32,

    class_mode='categorical',

    subset='training',

    seed=0

)



validation_generator = test_datagen.flow_from_directory(

    base_path,

    color_mode='grayscale',

    target_size=(200, 200),

    batch_size=32,

    class_mode='categorical',

    subset='validation',

    seed=0

)



labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())



print(labels)
inputs = Input(shape=(200, 200, 1))



net = Conv2D(filters=64, kernel_size=3, padding='same')(inputs)

net = LeakyReLU()(net)

net = MaxPooling2D()(net)



net = Conv2D(filters=64, kernel_size=3, padding='same')(net)

net = LeakyReLU()(net)

net = MaxPooling2D()(net)



net = Conv2D(filters=64, kernel_size=3, padding='same')(net)

net = LeakyReLU()(net)

net = MaxPooling2D()(net)



shortcut = net



net = DepthwiseConv2D(kernel_size=3, padding='same')(net)

net = LeakyReLU()(net)



net = Conv2D(filters=64, kernel_size=1, padding='same')(net)

net = LeakyReLU()(net)



net = DepthwiseConv2D(kernel_size=3, padding='same')(net)

net = LeakyReLU()(net)



net = Conv2D(filters=64, kernel_size=1, padding='same')(net)

net = LeakyReLU()(net)



net = Add()([shortcut, net])



net = Conv2D(filters=64, kernel_size=3, padding='same')(net)

net = LeakyReLU()(net)

net = MaxPooling2D()(net)



net = Conv2D(filters=64, kernel_size=3, padding='same')(net)

net = LeakyReLU()(net)

net = MaxPooling2D()(net)



net = DepthwiseConv2D(kernel_size=3, padding='same')(net)

net = LeakyReLU()(net)



net = Conv2D(filters=128, kernel_size=1, padding='same')(net)

net = LeakyReLU()(net)



net = Flatten()(net)



net = Dense(128, activation='relu')(net)



net = Dense(64, activation='relu')(net)



outputs = Dense(20, activation='softmax')(net)



model = Model(inputs=inputs, outputs=outputs)



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])



model.summary()
model.fit_generator(train_generator, epochs=50, validation_data=validation_generator)