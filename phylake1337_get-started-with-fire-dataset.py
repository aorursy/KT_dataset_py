import os

import numpy as np # linear algebra

import matplotlib.pyplot as plt

from tqdm import tqdm

from tensorflow.keras.preprocessing.image import load_img
postive_fire_path = '/kaggle/input/fire-dataset/fire_dataset/fire_images'

negative_fire_path = '/kaggle/input/fire-dataset/fire_dataset/non_fire_images'
def images_to_array(data_dir, img_size=(480,640, 3)):

    '''

    1- Read image samples from certain directory.

    2- Resize and Stack them into one big numpy array.

    3- shuffle the data.

    '''

    image_names = os.listdir(data_dir)

    data_size = len(image_names)

    #initailize data arrays.

    X = np.zeros([data_size, img_size[0], img_size[1], img_size[2]], dtype=np.uint8)

    #read data.

    for i in tqdm(range(data_size)):

        image_name = image_names[i]

        img_dir = os.path.join(data_dir, image_name)

        img_pixels = load_img(img_dir, color_mode='rgb', target_size=img_size)

        X[i] = img_pixels

        

    #shuffle    

    ind = np.random.permutation(data_size)

    X = X[ind]

    

    print('Ouptut Data Size: ', X.shape)

    return X
postive_fire_imgs = images_to_array(postive_fire_path)

negative_fire_imgs = images_to_array(negative_fire_path)
samples = np.concatenate((postive_fire_imgs[:3], negative_fire_imgs[:3]), axis=0) 



f, ax = plt.subplots(2, 3, figsize=(20,11))

f.subplots_adjust(hspace = .05, wspace=.05)

for i, img in enumerate(samples):

    ax[i//3, i%3].imshow(img)

    ax[i//3, i%3].axis('off')

    if i<3:

        ax[i//3, i%3].title.set_text('Postive')

    else:

        ax[i//3, i%3].title.set_text('Negative')

plt.show() 
postive_labels = np.ones((len(postive_fire_imgs), 1))

negative_labels = np.zeros((len(negative_fire_imgs), 1))

X = np.concatenate((postive_fire_imgs, negative_fire_imgs))

y = np.concatenate((postive_labels, negative_labels))

#shuffle

ind = np.random.permutation(999)

X = X[ind]

y = y[ind]

print('X Shape: ', X.shape)

print('y Shape: ', y.shape)
samples = X[:6]

f, ax = plt.subplots(2, 3, figsize=(20,11))

f.subplots_adjust(hspace = .05, wspace=.05)

for i, img in enumerate(samples):

    ax[i//3, i%3].imshow(img)

    ax[i//3, i%3].axis('off')

    if y[i]:

        ax[i//3, i%3].title.set_text('Postive')

    else:

        ax[i//3, i%3].title.set_text('Negative')

plt.show() 