from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import os

import pydicom

import cv2

import math



import scipy.ndimage

from skimage import measure, morphology

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from sklearn.preprocessing import LabelEncoder 



import tensorflow as tf

from tensorflow.keras import datasets, layers, models, regularizers



PATH = '/kaggle/input/spieaapm-lung-ct/'

os.listdir(PATH)
labels_df = pd.read_csv(PATH+'Labels.csv', index_col = 0)

labels_df = labels_df[:70]

labels_df.head()
labels_df['Diagnosis'] = labels_df['Diagnosis'].str.strip()



le = LabelEncoder() 

le.fit(labels_df['Diagnosis'])

print(list(le.classes_))

labels_df['Diagnosis'] = le.transform(labels_df['Diagnosis'])

labels_df.head()
patients_dir = os.listdir(PATH+'Dataset/Dataset')



def load_data(patient):

    path = PATH+'Dataset/Dataset/'+patient

    #print(patient)

    

    label = labels_df['Diagnosis'][patient]

    #print(label)

    

    slices = [pydicom.dcmread(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        

    #print(slice_thickness)

    for s in slices:

        s.SliceThickness = slice_thickness

    

    #print(len(slices), slices[0].pixel_array.shape)

    return label, slices
label, slices = load_data(patients_dir[0])
def get_pixels_hu(slices):

    image = np.stack([s.pixel_array for s in slices])

    image = image.astype(np.int16)



    # Set outside-of-scan pixels to 0 i.e. air

    image[image == -2000] = 0

    

    # Convert to Hounsfield units (HU)

    for slice_number in range(len(slices)):

        

        intercept = slices[slice_number].RescaleIntercept

        slope = slices[slice_number].RescaleSlope

        

        if slope != 1:

            image[slice_number] = slope * image[slice_number].astype(np.float64)

            image[slice_number] = image[slice_number].astype(np.int16)

            

        image[slice_number] += np.int16(intercept)

        

    image = [cv2.resize(np.array(frame),(128,128)) for frame in image]

            

    return np.array(image, dtype=np.int16)
def chunks(l,n, hm_slices):

    count=0

    for i in range(0, len(l), n):

        if(count < hm_slices):

            yield l[i:i + n]

            count=count+1



def mean(a):

    return sum(a) / len(a)



def reshape_data(slices, hm_slices=128):

    new_slices = []

    

    chunk_sizes = math.floor(len(slices) / hm_slices)

    for slice_chunk in chunks(slices, chunk_sizes, hm_slices):

        slice_chunk = list(map(mean, zip(*slice_chunk)))

        new_slices.append(slice_chunk)

        

    #print(len(new_slices))

    return np.array(new_slices)
images = get_pixels_hu(slices)

images = reshape_data(images)

plt.hist(images.flatten(), bins=80, color='c')

plt.xlabel("Hounsfield Units (HU)")

plt.ylabel("Frequency")

plt.show()



# Show some slice in the middle

plt.imshow(images[20], cmap=plt.cm.gray)

plt.show()
def plot_3d(image, threshold=-300):

    

    # Position the scan upright, so the head of the patient would be at the top facing the camera

    p = image.transpose(2,1,0)

    

    verts, faces, normals, values = measure.marching_cubes_lewiner(p, threshold)



    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111, projection='3d')



    # Fancy indexing: `verts[faces]` to generate a collection of triangles

    mesh = Poly3DCollection(verts[faces], alpha=0.70)

    face_color = [0.45, 0.45, 0.75]

    mesh.set_facecolor(face_color)

    ax.add_collection3d(mesh)



    ax.set_xlim(0, p.shape[0])

    ax.set_ylim(0, p.shape[1])

    ax.set_zlim(0, p.shape[2])



    plt.show()
plot_3d(images, 400)
def largest_label_volume(im, bg=-1):

    vals, counts = np.unique(im, return_counts=True)



    counts = counts[vals != bg]

    vals = vals[vals != bg]



    if len(counts) > 0:

        return vals[np.argmax(counts)]

    else:

        return None



def segment_lung_mask(image, fill_lung_structures=True):

    binary_image = np.array(image > -320, dtype=np.int8)+1

    labels = measure.label(binary_image)

    

    background_label = labels[0,0,0]

    binary_image[background_label == labels] = 2

    

    if fill_lung_structures:

        # For every slice we determine the largest solid structure

        for i, axial_slice in enumerate(binary_image):

            axial_slice = axial_slice - 1

            labeling = measure.label(axial_slice)

            l_max = largest_label_volume(labeling, bg=0)

            

            if l_max is not None: #This slice contains some lung

                binary_image[i][labeling != l_max] = 1



    

    binary_image -= 1 #Make the image actual binary

    binary_image = 1-binary_image # Invert it, lungs are now 1

    

    # Remove other air pockets insided body

    labels = measure.label(binary_image, background=0)

    l_max = largest_label_volume(labels, bg=0)

    if l_max is not None: # There are air pockets

        binary_image[labels != l_max] = 0

 

    return binary_image
segmented_lungs = segment_lung_mask(images, False)

segmented_lungs_fill = segment_lung_mask(images, True)
plot_3d(segmented_lungs, 0)
plot_3d(segmented_lungs_fill - segmented_lungs, 0)
MIN_BOUND = -1000.0

MAX_BOUND = 400.0

    

def normalize(image):

    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)

    image[image>1] = 1.

    image[image<0] = 0.

    return image



PIXEL_MEAN = 0.25



def zero_center(image):

    image = image - PIXEL_MEAN

    return image
def process_data(patient):

    label, slices = load_data(patient)

        

    images = get_pixels_hu(slices)

    images = [cv2.resize(np.array(frame),(64,64)) for frame in images]

    images = reshape_data(images, hm_slices=64)

    

    segmented_lungs = segment_lung_mask(images, False)

    segmented_lungs_fill = segment_lung_mask(images, True)

    lungs = segmented_lungs-segmented_lungs_fill

    

    lungs = normalize(lungs)

    lungs = zero_center(lungs)

    

    return lungs, label
for i in patients_dir[:2]:

    lungs, label = process_data(i)

    print(lungs.shape, label)
train_data = []

train_labels = []

test_data = []

test_labels = []



for i, patient in enumerate(patients_dir):    

    lungs, label = process_data(patient)

    if i<60:

        train_data.append(lungs)

        train_labels.append(label)

    else:

        test_data.append(lungs)

        test_labels.append(label)

     

train_data = np.array(train_data)

train_labels = np.array(train_labels)

test_data = np.array(test_data)

test_labels = np.array(test_labels)
print(test_data.shape,train_data.shape)

print(test_labels.shape, train_labels.shape)
train_data = train_data.reshape((60, 64, 64, 64,1))

test_data = test_data.reshape((10, 64, 64, 64,1))

print(test_data.shape,train_data.shape)



tf.compat.v2.random.set_seed(1)

np.random.seed(1)
model = models.Sequential()

model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=(64, 64, 64, 1)))

model.add(layers.MaxPooling3D((2, 2, 2)))

model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))

model.add(layers.MaxPooling3D((2, 2, 2)))

model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))

model.add(layers.MaxPooling3D((2, 2, 2)))

model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))



model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(2, activation='relu'))



model.summary()
model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



history = model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))