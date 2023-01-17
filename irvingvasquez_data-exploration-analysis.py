import numpy as np

from random import randint

import os

import matplotlib.pyplot as plt



import classification_utils as cnbv

# configure the dataset address



dataset_folder = '../input/nbv-classification/classification/classification/training/'

print(os.listdir('../input/nbv-classification/'))



file_lbl = 'dataset_pose.npy'
# address

file_vol = 'dataset_vol_classification_training.npy'



# load the inputs

path_input_vol = os.path.join(dataset_folder, file_vol)

dataset_vol = np.load(path_input_vol)



print("Input data size: \n",dataset_vol.shape)
#lets draw some grids



for i in range(3):

    cnbv.showGrid(dataset_vol[randint(0, len(dataset_vol))])
# The Labels



file_lbl = 'dataset_lbl_classification_training.npy'



path_input_lbl = os.path.join(dataset_folder, file_lbl)

dataset_lbl = np.load(path_input_lbl)

print("Labels data size: \n",dataset_lbl.shape)

classes = np.unique(dataset_lbl)

print("Available clases: n", classes)
# Read the pose that corresponds to a class.

# such poses are the vertices of a icosahedron



# This function converts a class to its corresponding pose

def getPositions(nbv_class, positions):

    return np.array(positions[nbv_class])
# Read the pose that corresponds to a class.

nbv_positions = np.genfromtxt('../input/nbv-classification/points_in_sphere.txt')



# This function converts a class to its corresponding pose

def getPositions(nbv_class, positions):

    return np.array(positions[nbv_class])
positions = getPositions(classes, nbv_positions)



fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')



ax.scatter(positions[:,1], positions[:,2], positions[:,2], color='red')



ax.set_xlabel('X Label')

ax.set_ylabel('Y Label')

ax.set_zlabel('Z Label')



plt.show()
# draw the some grids with nbv examples

for i in range(3):

    idx = randint(0, len(dataset_vol))

    print(getPositions(dataset_lbl[idx], nbv_positions)[0])

    cnbv.showGrid(dataset_vol[idx], getPositions(dataset_lbl[idx], nbv_positions)[0])