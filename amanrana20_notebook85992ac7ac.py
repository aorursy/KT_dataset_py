import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import dicom

import pandas as pd

import matplotlib.pyplot as plt

import keras.backend as K

K.set_image_dim_ordering('th')

from keras.models import Model

from keras.layers import Input, Dense, Flatten, Convolution3D, MaxPooling3D, BatchNormalization







PATH_BASE = '../input'

print('{} {}\n'.format('Files:', os.listdir(PATH_BASE)))

EXT_CSV = 'stage1_labels.csv'

EXT_SAMPLE_IMAGES = 'sample_images'

folders = os.listdir(os.path.join(PATH_BASE, EXT_SAMPLE_IMAGES))

print('# Folders in sample_images: {}'.format(len(folders))) 
patients = []

counter = 0

for i, folder in enumerate(folders):

    files = os.listdir(os.path.join(PATH_BASE, EXT_SAMPLE_IMAGES, folder))

    print('Folder {}: {} has {} files'.format(i+1, folder, len(files)))

    images = []

    for j, file in enumerate(files):

        images.append(dicom.read_file(os.path.join(PATH_BASE, EXT_SAMPLE_IMAGES, folder, file)))

    counter += len(images)

    images.sort(key = lambda z: int(z.InstanceNumber))

    patients.append({folder: images})
csv = pd.read_csv(os.path.join(PATH_BASE, EXT_CSV))
def get_cancer_ground_truth(patient_id):

    label = (csv.loc[csv['id'] == patient_id]['cancer']).any()

    if label:

        return np.array([1, 0])

    else:

        return np.array([0, 1])
def create_dataset():

    images_3d_arr = []

    for i, patient in enumerate(patients):

        patient_id, patient_images = list(patient.keys()), list(patient.values())

        images_3d = np.zeros([len(patient_images[0]), 512, 512])

        for j, image_slice in enumerate(patient_images[0]):

            image = image_slice.pixel_array

            image[image == -2000] = 0

            image = np.add(image * image_slice.RescaleSlope, image_slice.RescaleIntercept)

            images_3d[j, :, :] = image

        has_cancer = get_cancer_ground_truth(patient_id[0])

        images_3d_arr.append([np.array(images_3d).astype(np.float16), has_cancer])

        

    return np.array(images_3d_arr)
import matplotlib.animation as animation

from IPython.display import HTML

%matplotlib nbagg



fig = plt.figure()



all_images = []



for i, patient in enumerate(patients):

    patient_id, patient_images = list(patient.keys()), list(patient.values())

    for j, image_slice in enumerate(patient_images[0]):

        image = image_slice.pixel_array

        image[image == -2000] = 0

        image = np.add(image * image_slice.RescaleSlope, image_slice.RescaleIntercept)

        all_images.append(image)

    if i == 1:

        break



im = plt.imshow(all_images[1], cmap=plt.cm.bone , animated=True)



def update_fig(pos):

    im.set_array(all_images[pos])

    return im



ani = animation.FuncAnimation(fig, update_fig, frames=range(len(all_images)), interval=50, blit=True)

ani.save('lung.gif', writer='imagemagick')

plt.show()
def cnn_model(input_shape, output_shape):

    inp = Input(shape=(input_shape[0], input_shape[1], input_shape[2], input_shape[3]))

    #Layer 1

    l1_conv1 = Convolution3D(32, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(inp)

    l1_conv2 = Convolution3D(32, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l1_conv1)

    l1_maxpool1 = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2))(l1_conv1)

    

    #Layer 2

    l2_conv1 = Convolution3D(32, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l1_maxpool1)

    l2_conv2 = Convolution3D(32, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l2_conv1)

    l2_maxpool1 = MaxPooling3D(pool_size=(2, 2, 2))(l2_conv2)

    

    #Layer 3

    l3_conv1 = Convolution3D(64, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l2_maxpool1)

    l3_conv2 = Convolution3D(64, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l3_conv1)

    l3_maxpool1 = MaxPooling3D(pool_size=(2, 2, 2))(l3_conv2)

    

    #Layer 4

    l4_conv1 = Convolution3D(128, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l3_maxpool1)

    l4_conv2 = Convolution3D(128, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l4_conv1)

    l4_maxpool1 = MaxPooling3D(pool_size=(2, 2, 2))(l4_conv2)

    

    #Layer 5

    l5_conv1 = Convolution3D(128, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l4_maxpool1)

    l5_conv2 = Convolution3D(128, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l5_conv1)

    l5_maxpool1 = MaxPooling3D(pool_size=(2, 2, 2))(l5_conv2)

    

    #Layer 6

    l6_conv1 = Convolution3D(256, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l5_maxpool1)

    l6_conv2 = Convolution3D(256, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l6_conv1)

    l6_maxpool1 = MaxPooling3D(pool_size=(2, 2, 2))(l6_conv2)

    

    # Flatten

    flat = Flatten()(l6_maxpool1)

    dense1 = Dense(512, init='glorot_uniform', activation='relu')(flat)

    dense2 = Dense(64, init='glorot_uniform', activation='relu')(dense1)

    out = Dense(output_shape, activation='softmax')(dense2)

    

    model = Model(input=inp, output=out)

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    return model
'''

dataset = create_dataset()

sh = np.array(dataset[0, 0]).shape

l = len(dataset)

X = np.zeros([l, 432, sh[1], sh[2]])



for i, x in enumerate(np.array(dataset[:, 0])):

    X[i, :, :, :] = x



Y = np.array([y for y in dataset[:, 1]])

print(X.shape, Y.shape)

'''
model = cnn_model((1, 177, 512, 512), 2)

print(model.summary())