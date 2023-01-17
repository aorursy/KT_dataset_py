# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

%matplotlib inline



from IPython.display import clear_output

from time import sleep



from math import sin, cos, pi

import cv2

from tqdm.notebook import tqdm
horizontal_flip = False

rotation_augmentation = True

brightness_augmentation = True

shift_augmentation = True

random_noise_augmentation = True



include_unclean_data = True    # Whether to include samples with missing keypoint values. Note that the missing values would however be filled using Pandas' 'ffill' later.

sample_image_index = 20    # Index of sample train image used for visualizing various augmentations



rotation_angles = [12]    # Rotation angle in degrees (includes both clockwise & anti-clockwise rotations)

pixel_shifts = [12] 
Train_Dir = '../input/fpoints/training.csv'

Test_Dir = '../input/fpoints/test.csv'

lookid_dir = '../input/fpoints/IdLookupTable.csv'

train_data = pd.read_csv(Train_Dir)  

test_data = pd.read_csv(Test_Dir)

lookid_data = pd.read_csv(lookid_dir)

os.listdir('../input')
train_data.info()
'''

there are total 31 columns, of with 30 are cordinates of 15 points and 31st column have 

the pixel in 1d array of size 96*96=9216 .

28 columns have null values which is filled by its previous value using ffill command



After that we reshape the image into 96*96 matrix

and then we split the training set using train test split method with split ratio 1:10



after that 2d image is converted to image like rgb image so as to applt resnet model which is CNN model

'''
print("Length of train data: {}".format(len(train_data)))

print("Number of Images with missing pixel values: {}".format(len(train_data) - int(train_data.Image.apply(lambda x: len(x.split())).value_counts().values)))
train_data.isnull().sum()
clean_train_data = train_data.dropna()

print("clean_train_data shape: {}".format(np.shape(clean_train_data)))



unclean_train_data = train_data.fillna(method = 'ffill')

print("unclean_train_data shape: {}\n".format(np.shape(unclean_train_data)))
def plot_sample(image, keypoint, axis, title):

    image = image.reshape(96,96)

    axis.imshow(image, cmap='gray')

    axis.scatter(keypoint[0::2], keypoint[1::2],c='red', marker='x', s=20)

    plt.title(title)
#Separate data into clean & unclean subsets

def load_images(image_data):

    images = []

    for idx, sample in image_data.iterrows():

        image = np.array(sample['Image'].split(' '), dtype=int)

        image = np.reshape(image, (96,96,1))

        images.append(image)

    images = np.array(images)/255.

    return images



def load_keypoints(keypoint_data):

    keypoint_data = keypoint_data.drop('Image',axis = 1)

    keypoint_features = []

    for idx, sample_keypoints in keypoint_data.iterrows():

        keypoint_features.append(sample_keypoints)

    keypoint_features = np.array(keypoint_features, dtype = 'float')

    return keypoint_features



clean_train_images = load_images(clean_train_data)

print("Shape of clean_train_images: {}".format(np.shape(clean_train_images)))

clean_train_keypoints = load_keypoints(clean_train_data)

print("Shape of clean_train_keypoints: {}".format(np.shape(clean_train_keypoints)))

test_images = load_images(test_data)

print("Shape of test_images: {}".format(np.shape(test_images)))



train_images = clean_train_images

train_keypoints = clean_train_keypoints

fig, axis = plt.subplots()

plot_sample(clean_train_images[sample_image_index], clean_train_keypoints[sample_image_index], axis, "Sample image & keypoints")



if include_unclean_data:

    unclean_train_images = load_images(unclean_train_data)

    print("Shape of unclean_train_images: {}".format(np.shape(unclean_train_images)))

    unclean_train_keypoints = load_keypoints(unclean_train_data)

    print("Shape of unclean_train_keypoints: {}\n".format(np.shape(unclean_train_keypoints)))

    train_images = np.concatenate((train_images, unclean_train_images))

    train_keypoints = np.concatenate((train_keypoints, unclean_train_keypoints))
def left_right_flip(images, keypoints):

    flipped_keypoints = []

    flipped_images = np.flip(images, axis=2)   # Flip column-wise (axis=2)

    for idx, sample_keypoints in enumerate(keypoints):

        flipped_keypoints.append([96.-coor if idx%2==0 else coor for idx,coor in enumerate(sample_keypoints)])    # Subtract only X co-ordinates of keypoints from 96 for horizontal flipping

    return flipped_images, flipped_keypoints



if horizontal_flip:

    flipped_train_images, flipped_train_keypoints = left_right_flip(clean_train_images, clean_train_keypoints)

    print("Shape of flipped_train_images: {}".format(np.shape(flipped_train_images)))

    print("Shape of flipped_train_keypoints: {}".format(np.shape(flipped_train_keypoints)))

    train_images = np.concatenate((train_images, flipped_train_images))

    train_keypoints = np.concatenate((train_keypoints, flipped_train_keypoints))

    fig, axis = plt.subplots()

    plot_sample(flipped_train_images[sample_image_index], flipped_train_keypoints[sample_image_index], axis, "Horizontally Flipped") 
def rotate_augmentation(images, keypoints):

    rotated_images = []

    rotated_keypoints = []

    print("Augmenting for angles (in degrees): ")

    for angle in rotation_angles:    # Rotation augmentation for a list of angle values

        for angle in [angle,-angle]:

            print(f'{angle}', end='  ')

            M = cv2.getRotationMatrix2D((48,48), angle, 1.0)

            angle_rad = -angle*pi/180.     # Obtain angle in radians from angle in degrees (notice negative sign for change in clockwise vs anti-clockwise directions from conventional rotation to cv2's image rotation)

            # For train_images

            for image in images:

                rotated_image = cv2.warpAffine(image, M, (96,96), flags=cv2.INTER_CUBIC)

                rotated_images.append(rotated_image)

            # For train_keypoints

            for keypoint in keypoints:

                rotated_keypoint = keypoint - 48.    # Subtract the middle value of the image dimension

                for idx in range(0,len(rotated_keypoint),2):

                    # https://in.mathworks.com/matlabcentral/answers/93554-how-can-i-rotate-a-set-of-points-in-a-plane-by-a-certain-angle-about-an-arbitrary-point

                    rotated_keypoint[idx] = rotated_keypoint[idx]*cos(angle_rad)-rotated_keypoint[idx+1]*sin(angle_rad)

                    rotated_keypoint[idx+1] = rotated_keypoint[idx]*sin(angle_rad)+rotated_keypoint[idx+1]*cos(angle_rad)

                rotated_keypoint += 48.   # Add the earlier subtracted value

                rotated_keypoints.append(rotated_keypoint)

            

    return np.reshape(rotated_images,(-1,96,96,1)), rotated_keypoints



if rotation_augmentation:

    rotated_train_images, rotated_train_keypoints = rotate_augmentation(clean_train_images, clean_train_keypoints)

    print("\nShape of rotated_train_images: {}".format(np.shape(rotated_train_images)))

    print("Shape of rotated_train_keypoints: {}\n".format(np.shape(rotated_train_keypoints)))

    train_images = np.concatenate((train_images, rotated_train_images))

    train_keypoints = np.concatenate((train_keypoints, rotated_train_keypoints))

    fig, axis = plt.subplots()

    plot_sample(rotated_train_images[sample_image_index], rotated_train_keypoints[sample_image_index], axis, "Rotation Augmentation")
def alter_brightness(images, keypoints):

    altered_brightness_images = []

    inc_brightness_images = np.clip(images*1.2, 0.0, 1.0)    # Increased brightness by a factor of 1.2 & clip any values outside the range of [-1,1]

    dec_brightness_images = np.clip(images*0.6, 0.0, 1.0)    # Decreased brightness by a factor of 0.6 & clip any values outside the range of [-1,1]

    altered_brightness_images.extend(inc_brightness_images)

    altered_brightness_images.extend(dec_brightness_images)

    return altered_brightness_images, np.concatenate((keypoints, keypoints))



if brightness_augmentation:

    altered_brightness_train_images, altered_brightness_train_keypoints = alter_brightness(clean_train_images, clean_train_keypoints)

    print(f"Shape of altered_brightness_train_images: {np.shape(altered_brightness_train_images)}")

    print(f"Shape of altered_brightness_train_keypoints: {np.shape(altered_brightness_train_keypoints)}")

    train_images = np.concatenate((train_images, altered_brightness_train_images))

    train_keypoints = np.concatenate((train_keypoints, altered_brightness_train_keypoints))

    fig, axis = plt.subplots()

    plot_sample(altered_brightness_train_images[sample_image_index], altered_brightness_train_keypoints[sample_image_index], axis, "Increased Brightness") 

    fig, axis = plt.subplots()

    plot_sample(altered_brightness_train_images[len(altered_brightness_train_images)//2+sample_image_index], altered_brightness_train_keypoints[len(altered_brightness_train_images)//2+sample_image_index], axis, "Decreased Brightness") 
def shift_images(images, keypoints):

    shifted_images = []

    shifted_keypoints = []

    for shift in pixel_shifts:    # Augmenting over several pixel shift values

        for (shift_x,shift_y) in [(-shift,-shift),(-shift,shift),(shift,-shift),(shift,shift)]:

            M = np.float32([[1,0,shift_x],[0,1,shift_y]])

            for image, keypoint in zip(images, keypoints):

                shifted_image = cv2.warpAffine(image, M, (96,96), flags=cv2.INTER_CUBIC)

                shifted_keypoint = np.array([(point+shift_x) if idx%2==0 else (point+shift_y) for idx, point in enumerate(keypoint)])

                if np.all(0.0<shifted_keypoint) and np.all(shifted_keypoint<96.0):

                    shifted_images.append(shifted_image.reshape(96,96,1))

                    shifted_keypoints.append(shifted_keypoint)

    shifted_keypoints = np.clip(shifted_keypoints,0.0,96.0)

    return shifted_images, shifted_keypoints



if shift_augmentation:

    shifted_train_images, shifted_train_keypoints = shift_images(clean_train_images, clean_train_keypoints)

    print(f"Shape of shifted_train_images: {np.shape(shifted_train_images)}")

    print(f"Shape of shifted_train_keypoints: {np.shape(shifted_train_keypoints)}")

    train_images = np.concatenate((train_images, shifted_train_images))

    train_keypoints = np.concatenate((train_keypoints, shifted_train_keypoints))

    fig, axis = plt.subplots()

    plot_sample(shifted_train_images[sample_image_index], shifted_train_keypoints[sample_image_index], axis, "Shift Augmentation")
def add_noise(images):

    noisy_images = []

    for image in images:

        noisy_image = cv2.add(image, 0.008*np.random.randn(96,96,1))    # Adding random normal noise to the input image & clip the resulting noisy image between [-1,1]

        noisy_images.append(noisy_image.reshape(96,96,1))

    return noisy_images



if random_noise_augmentation:

    noisy_train_images = add_noise(clean_train_images)

    print(f"Shape of noisy_train_images: {np.shape(noisy_train_images)}")

    train_images = np.concatenate((train_images, noisy_train_images))

    train_keypoints = np.concatenate((train_keypoints, clean_train_keypoints))

    fig, axis = plt.subplots()

    plot_sample(noisy_train_images[sample_image_index], clean_train_keypoints[sample_image_index], axis, "Random Noise Augmentation")
print("Shape of final train_images: {}".format(np.shape(train_images)))

print("Shape of final train_keypoints: {}".format(np.shape(train_keypoints)))



print("\n Clean Train Data: ")

fig = plt.figure(figsize=(20,8))

for i in range(10):

    axis = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])

    plot_sample(clean_train_images[i], clean_train_keypoints[i], axis, "")

plt.show()



if include_unclean_data:

    print("Unclean Train Data: ")

    fig = plt.figure(figsize=(20,8))

    for i in range(10):

        axis = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])

        plot_sample(unclean_train_images[i], unclean_train_keypoints[i], axis, "")

    plt.show()



if horizontal_flip:

    print("Horizontal Flip Augmentation: ")

    fig = plt.figure(figsize=(20,8))

    for i in range(10):

        axis = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])

        plot_sample(flipped_train_images[i], flipped_train_keypoints[i], axis, "")

    plt.show()



if rotation_augmentation:

    print("Rotation Augmentation: ")

    fig = plt.figure(figsize=(20,8))

    for i in range(10):

        axis = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])

        plot_sample(rotated_train_images[i], rotated_train_keypoints[i], axis, "")

    plt.show()

    

if brightness_augmentation:

    print("Brightness Augmentation: ")

    fig = plt.figure(figsize=(20,8))

    for i in range(10):

        axis = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])

        plot_sample(altered_brightness_train_images[i], altered_brightness_train_keypoints[i], axis, "")

    plt.show()



if shift_augmentation:

    print("Shift Augmentation: ")

    fig = plt.figure(figsize=(20,8))

    for i in range(10):

        axis = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])

        plot_sample(shifted_train_images[i], shifted_train_keypoints[i], axis, "")

    plt.show()

    

if random_noise_augmentation:

    print("Random Noise Augmentation: ")

    fig = plt.figure(figsize=(20,8))

    for i in range(10):

        axis = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])

        plot_sample(noisy_train_images[i], clean_train_keypoints[i], axis, "")

    plt.show()
train_keypoints.shape
np.save('train_images', train_images)

np.save('train_keypoints', train_keypoints)

np.save('test_images', test_images)
