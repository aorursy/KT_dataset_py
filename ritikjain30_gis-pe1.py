# importing ImageDataGenerator



from keras.preprocessing.image import ImageDataGenerator



# creating generator



datagen = ImageDataGenerator(rescale=1. / 255)
# preparing iterators for each dataset



data_it = datagen.flow_from_directory('../input/giseurosat/2750', class_mode='categorical')

data_it.batch_size

data_it.target_size

data_it.color_mode


batchX, batchY = data_it.next()





batchX.shape
import os

import cv2



def load_gray_images_from_folder(folder):

    images = []

    for filename in os.listdir(folder):

        if filename.endswith(".jpg"):

            img = cv2.imread(os.path.join(folder, filename))

            if img is not None:

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                images.append(gray)

    return images



root_folder = '../input/giseurosat/2750/AnnualCrop'

all_canny_gray_images = [img for img in load_gray_images_from_folder(root_folder)]
def load_images_from_folder(folder):

    images = []

    for filename in os.listdir(folder):

        if filename.endswith(".jpg"):

            img = cv2.imread(os.path.join(folder, filename))

            if img is not None:

                images.append(img)

    return images



root_folder = '../input/giseurosat/2750/AnnualCrop'

all_RGB_images = [img for img in load_images_from_folder(root_folder)]
import os

os.mkdir('/kaggle/working/AnnualCrop/')
number = 0



for image in all_RGB_images:

    b_channel, g_channel, r_channel = cv2.split(image)

    four_channel = cv2.merge((b_channel, g_channel, r_channel, all_canny_gray_images[number]))

    cv2.imwrite('./AnnualCrop/four_channel_{}.jpg'.format(number), four_channel)

    number += 1
import numpy as np



np.asarray(four_channel).shape
