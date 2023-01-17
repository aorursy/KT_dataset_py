from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
img_transform  = ImageDataGenerator(
                rotation_range=80,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                rescale=1./255,
                horizontal_flip=True,
                fill_mode='nearest'
                )
img = load_img('../input/cat-and-dog/test_set/test_set/cats/cat.4040.jpg')

img
# this is a Numpy array with shape (3, 150, 150)

x = img_to_array(img)

# this is a Numpy array with shape (1, 3, 150, 150)

x = x.reshape((1,) + x.shape)
# make new idrectory

import os

os.mkdir('../working/image_output')
i = 0
for batch in img_transform.flow(x, batch_size=1, save_prefix='cats', save_format='png', save_to_dir='../working/image_output'):
    i += 1
    
    if i > 20:
        break
load_img('../working/image_output/cats_0_1935.png')
load_img('../working/image_output/cats_0_4042.png')
