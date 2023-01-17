import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

from tqdm import tqdm#_notebook as tqdm

import matplotlib.pyplot as plt

import seaborn as sns

from functools import reduce

import os

from sklearn.model_selection import train_test_split

from scipy.optimize import minimize

import json

from math import sin, cos
!wget https://raw.githubusercontent.com/ApolloScapeAuto/dataset-api/master/car_instance/car_models.py

import car_models
# Load 3D models of cars

cars = {}



for car_name in car_models.car_name2id:

    with open('../input/pku-autonomous-driving/car_models_json/{}.json'.format(car_name)) as json_file:

        data = json.load(json_file)

    vertices = np.array(data['vertices'])

    vertices[:, 1] = -vertices[:, 1]

    faces = np.array(data['faces']) - 1

    cars[car_name] = vertices, faces
# From camera.zip

camera_matrix = np.array([[2304.5479, 0,  1686.2379],

                          [0, 2305.8757, 1354.9849],

                          [0, 0, 1]], dtype=np.float32)

camera_matrix_inv = np.linalg.inv(camera_matrix)



# convert euler angle to rotation matrix

def euler_to_Rot(yaw, pitch, roll):

    Y = np.array([[cos(yaw), 0, sin(yaw)],

                  [0, 1, 0],

                  [-sin(yaw), 0, cos(yaw)]])

    P = np.array([[1, 0, 0],

                  [0, cos(pitch), -sin(pitch)],

                  [0, sin(pitch), cos(pitch)]])

    R = np.array([[cos(roll), -sin(roll), 0],

                  [sin(roll), cos(roll), 0],

                  [0, 0, 1]])

    return np.dot(Y, np.dot(P, R))
img_shape = (2710, 3384)



def get_mask(car_model):

    verts, facs = car_model

    mask = np.zeros([img_shape[0], img_shape[1]], dtype=np.float32)

    # Get values

    x, y, z = 0, 0, 4

    yaw, pitch, roll = np.pi/2, 0, np.pi

    # Math

    Rt = np.eye(4)

    t = np.array([x, y, z])

    Rt[:3, 3] = t

    Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T

    Rt = Rt[:3, :]

    P = np.ones((verts.shape[0], verts.shape[1]+1))

    P[:, :-1] = verts

    P = P.T

    img_cor_points = np.dot(camera_matrix, np.dot(Rt, P))

    img_cor_points = img_cor_points.T

    img_cor_points[:, 0] /= img_cor_points[:, 2]

    img_cor_points[:, 1] /= img_cor_points[:, 2]

    img_cor_points = img_cor_points[:,:2].astype(int)

    # Drawing

    for face in facs:

        cv2.fillConvexPoly(mask, img_cor_points[face], 1)

    mask = cv2.resize(mask, (img_shape[1]//4, img_shape[0]//4))

    return mask
car_masks = {car_name: get_mask(cars[car_name]) for car_name in tqdm(cars)}
fig, axes = plt.subplots(10, 8, figsize=(30, 30))



for i, car_name in enumerate(cars):

    ax = axes[i//8, i%8]

    ax.set_title(car_name)

    ax.axis('off')

    mask = car_masks[car_name]

    ax.imshow(mask)
from sklearn.metrics import pairwise_distances



def sim_affinity(X):

    return pairwise_distances(X, metric=car_iou)



def car_iou(car1, car2):

    car1 = car1 > 0.5

    car2 = car2 > 0.5

    return 1 - (car1 & car2).sum() / (car1 | car2).sum()



car_names = sorted(list(car_masks.keys()))

X = np.array([car_masks[n] for n in car_names]).reshape(len(cars), -1)
plt.figure(figsize=(20,20))



plt.title('IOU Distances between car models')

dist_matrix = pd.DataFrame(sim_affinity(X), columns=car_names, index=car_names)

sns.heatmap(dist_matrix);
sns.distplot(np.array(dist_matrix).reshape(-1));
from sklearn.cluster import AgglomerativeClustering



clustering = AgglomerativeClustering(affinity=sim_affinity, linkage='complete', distance_threshold=0.2, n_clusters=None)

clustering.fit(X)

print('Num clusters:', max(clustering.labels_) + 1)
cluster_centers = []

cluster_dict = {}



for label in np.unique(clustering.labels_):

    this_cl_names = list(np.array(car_names)[clustering.labels_ == label])

    center = dist_matrix.loc[this_cl_names, this_cl_names].sum(0).idxmin()

    cluster_centers.append(center)

    

    fig, axes = plt.subplots(1, len(this_cl_names), figsize=(30, 8))

    print('Cluster #{}. Center: {}'.format(label + 1, center))

    if len(this_cl_names) == 1:

        axes = [axes]

    for i, car_name in enumerate(this_cl_names):

        cluster_dict[car_name] = center

        axes[i].set_title(car_name)

        if car_name == center:

            axes[i].set_title('(*) ' + car_name, fontweight="bold")

        axes[i].axis('off')

        axes[i].imshow(car_masks[car_name])

    plt.show()
# Draw cluster centers

fig, axes = plt.subplots(2, len(cluster_centers) // 2, figsize=(30, 8))

for i, car_name in enumerate(cluster_centers):

    ax = axes[i // (len(cluster_centers) // 2), i % (len(cluster_centers) // 2)]

    ax.set_title(car_name)

    ax.axis('off')

    ax.imshow(car_masks[car_name])
print('cluster_centers =', cluster_centers)
# this dict maps car name to its corresponding cluster center

print('cluster_dict =', cluster_dict)