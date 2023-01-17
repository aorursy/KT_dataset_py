# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)=

import tensorflow as tf

import cv2

import scipy.spatial.distance as sp

import pathlib

import imageio



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
face_data = []

training_paths = pathlib.Path('../input/faceset3/new_train_images').glob('*/*.jpg')

training_sorted = sorted([x for x in training_paths])

print(len(training_sorted))

for x in training_sorted:

    im_path = x

    temp = cv2.imread(str(im_path),0)

#     print(temp.shape)

    temp = temp.flatten()

#     print(temp.shape)

    face_data.append(temp)

# print(face_data.shape)

# print(im_path)

# im = imageio.imread(str(im_path))

# print(im.flatten())
train_faces = np.asarray(face_data)

mean = np.mean(train_faces, axis = 0)

print(train_faces.shape)

print(mean.shape)
# for i in range (train_faces.shape[1]):

#     for j in range(train_faces.shape[0]):

#         train_faces[j][i] = train_faces[j][i] - mean[i]

train_faces = train_faces - mean
cov_mat = np.matmul(train_faces,np.transpose(train_faces))

eigval, eigvec = np.linalg.eig(cov_mat)

print(eigval)

print(eigvec.shape)
imp_eig_vec = eigvec[::,0:1]

eig_faces = np.matmul(np.transpose(imp_eig_vec),train_faces)

print(eig_faces.shape)

weights = np.matmul(eig_faces,np.transpose(train_faces))

print(weights.shape)
w = np.reshape(weights,weights.shape[1])

st_dev = np.std(w, axis = 0)

var = st_dev*st_dev

var = np.reshape(var,[1,1])
test_data = []

test_paths = pathlib.Path('../input/testfaceset3/new_test_images').glob('*/*.jpg')

test_sorted = sorted([x for x in test_paths])
for x in test_sorted:

    im_path = x

    print(x)

    temp = cv2.imread(str(im_path),0)

#     print(temp.shape)

    temp = temp.flatten()

#     print(temp.shape)

    test_data.append(temp)

test_faces = np.asarray(test_data)

test_mean = np.mean(test_faces, axis = 0)
# for i in range (test_faces.shape[1]):

#     for j in range(test_faces.shape[0]):

#         test_faces[j][i] = test_faces[j][i] - mean[i]

test_faces = test_faces - mean

print(test_faces.shape)


correct_count = 0

for j in range(7):

    min_dist = (10)**21

    test_obj = np.matmul(eig_faces,test_faces[j])

#     print(test_obj[0])

    for i in range (weights.shape[1]):    

        dist = sp.mahalanobis(weights[0][i], test_obj[0], np.linalg.inv(var))

#         dist = sp.euclidean(weights[0][i], test_obj[0])

        if ( dist < min_dist ):

            min_dist = dist

            min_i = i

    print(min_dist,min_i,j)

#     if((min_i//5)==j):

#         correct_count += 1

    threshold = 0.005

    if (min_dist < threshold):

        if ((min_i//5))==j:

            correct_count += 1

    else:

        if j>=5:

            correct_count += 1

print(correct_count/7)
cov_mat = np.matmul(train_faces, np.transpose(train_faces))

eigval, eigvec = np.linalg.eig(cov_mat)

imp_eig_vec = eigvec[::,0:2]

eig_faces = np.matmul(np.transpose(imp_eig_vec),train_faces)

weights = np.matmul(eig_faces,np.transpose(train_faces))

final_eig_faces = weights

print(weights.shape)
import copy

class_means = np.empty([5,2])

all_mean = np.mean(final_eig_faces, axis = 1)

print(all_mean.shape)



for j in range(5):

    for i in range (int(j*final_eig_faces.shape[1]/5), int((j+1)*final_eig_faces.shape[1]/5)):

        class_means[j][0] = class_means[j][0] + final_eig_faces[0][i]

        class_means[j][1] = class_means[j][1] + final_eig_faces[1][i]



    class_means[j][0] = class_means[j][0]/5

    class_means[j][1] = class_means[j][1]/5



dataSw = copy.deepcopy(final_eig_faces)

dataSb = copy.deepcopy(final_eig_faces)

for i in range ( dataSb.shape[1] ):

    dataSb[0][i] = dataSb[0][i] - all_mean[0]

    dataSb[1][i] = dataSb[1][i] - all_mean[1]



for i in range ( dataSw.shape[1] ):

    a = int(i/5)

    dataSw[0][i] = dataSw[0][i] - class_means[a][0]

    dataSw[1][i] = dataSw[1][i] - class_means[a][1]



Sb = np.matmul(dataSb,np.transpose(dataSb))

Sw = np.matmul(dataSw,np.transpose(dataSw))

print(Sw.shape)
eigen_value, eigen_vector = np.linalg.eig(np.matmul(np.linalg.inv(Sw),Sb))

imp_eig_vector = eigen_vector[::,0:1]

print(imp_eig_vector.shape)

fisher_faces = np.matmul(np.transpose(imp_eig_vector),final_eig_faces)

print(fisher_faces.shape)

w1 = np.reshape(fisher_faces,(weights.shape[1]))

print(w1.shape)

st_dev = np.std(w1, axis = 0)

var = st_dev*st_dev

print("variance",var)

var = np.reshape(var,[1,1])

print(var.shape)

print(eig_faces.shape)


correct_count = 0

for j in range(7):

    min_dist = (10)**15

#     print(test_faces.shape)

    test_obj = test_faces[j]

    test_obj = np.reshape(test_obj,[921600,1])

#     print(eig_faces.shape)

#     print(test_obj.shape)

    test_face = np.matmul(eig_faces,test_obj)

#     print(test_face.shape)

#     print(imp_eig_vector.shape)

    test_face = np.matmul(np.transpose(imp_eig_vector),test_face)

    test_face = test_face[0]

#     print(test_face.shape)

    for i in range (fisher_faces.shape[1]):    

        dist = sp.mahalanobis(fisher_faces[0][i], test_face, np.linalg.inv(var))

        if ( dist < min_dist ):

            min_dist = dist

            min_i = i

    print(min_dist,min_i,j)

    threshold = 0.005

#     if ((min_i//5))==j:

#         correct_count += 1

    if (min_dist < threshold):

        if ((min_i//5))==j:

            correct_count += 1

    else:

        if j>=5:

            correct_count += 1

print(correct_count/7)