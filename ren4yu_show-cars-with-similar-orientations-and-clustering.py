import math

import random

from pathlib import Path

from collections import defaultdict

import numpy as np

import pandas as pd

import cv2

from scipy.spatial.transform import Rotation

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
camera_matrix = np.array(

    [[2304.5479, 0,  1686.2379],

     [0, 2305.8757, 1354.9849],

     [0, 0, 1]], dtype=np.float32)



# code from https://www.kaggle.com/hocop1/centernet-baseline

def str2coords(s, names=('id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z')):

    '''

    Input:

        s: PredictionString (e.g. from train dataframe)

        names: array of what to extract from the string

    Output:

        list of dicts with keys from `names`

    '''

    coords = []

    for l in np.array(s.split()).reshape([-1, 7]):

        coords.append(dict(zip(names, l.astype('float'))))

        if 'id' in coords[-1]:

            coords[-1]['id'] = int(coords[-1]['id'])

    return coords



def get_img_coords(x, y, z):

    p = np.array([x, y, z]).T

    img_p = np.dot(camera_matrix, p)

    img_p[0] /= img_p[2]

    img_p[1] /= img_p[2]

    return img_p[0], img_p[1], z



def get_centerize_mat(x, y, z):

    yaw = 0

    pitch = -np.arctan(x / z)

    roll = np.arctan(y / z)

    return Rotation.from_euler("xyz", (roll, pitch, yaw)).as_dcm()



def get_targets(c):

    x, y, z = c["x"], c["y"], c["z"]

    roll, pitch, yaw = c["roll"], c["pitch"], c["yaw"]

    ix, iy, _ = get_img_coords(x, y, z)

    Rt2 = get_centerize_mat(x, y, z)

    Rt1 = Rt2 @ Rotation.from_euler("yxz", (pitch, yaw, roll)).as_dcm()

    rot = Rotation.from_dcm(Rt1)

    r1, r2, r3 = rot.as_euler("yxz")

    r3 = r3 - math.pi if r3 > 0 else r3 + math.pi

    return dict(x=x, y=y, z=z, r1=r1, r2=r2, r3=r3, ix=ix, iy=iy)
train = pd.read_csv("../input/pku-autonomous-driving/train.csv")



angles = []

cars = []



for i, row in tqdm(train.iterrows(), total=len(train)):

    coords = str2coords(row["PredictionString"])



    for c in coords:

        t = get_targets(c)

        t["img_id"] = row["ImageId"]

        cars.append(t)

        angles.append((t["r1"], t["r2"], t["r3"]))
def imcrop(img, bbox):

    x1, y1, x2, y2 = bbox

    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:

        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)

    return img[y1:y2, x1:x2, :]



def pad_img_to_fit_bbox(img, x1, x2, y1, y2):

    img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),

                            -min(0, x1), max(x2 - img.shape[1], 0),cv2.BORDER_REPLICATE)

    y2 += -min(0, y1)

    y1 += -min(0, y1)

    x2 += -min(0, x1)

    x1 += -min(0, x1)

    return img, x1, x2, y1, y2



def get_target_car_img(car):

    img_dir = Path("../input/pku-autonomous-driving/train_images")

    img_id, x, y, s = car["img_id"], car["ix"], car["iy"], 10000 / car["z"]

    img_path = img_dir.joinpath(img_id + ".jpg")

    img = cv2.imread(str(img_path))

    h, w = img.shape[:2]

    x1 = int(x - s / 2)

    y1 = int(y - s / 2)

    x2 = int(x + s / 2)

    y2 = int(y + s / 2)

    return imcrop(img, (x1, y1, x2, y2))
def rot_dist(rot1, rot2):

    diff = Rotation.inv(rot2) * rot1

    w = np.clip(diff.as_quat()[-1], -1., 1.)

    w = (math.acos(w) * 360) / math.pi

    if w > 180:

        w = 360 - w

    return w



def euler_dist(euler1, euler2):

    rot1 = Rotation.from_euler("xyz", euler1)

    rot2 = Rotation.from_euler("xyz", euler2)

    return rot_dist(rot1, rot2)



def car_dist(car1, car2):

    euler1 = (car1["r1"], car1["r2"], car1["r3"])

    euler2 = (car2["r1"], car2["r2"], car2["r3"])

    return euler_dist(euler1, euler2)
def show_cars(cars):

    cols, rows = 4, 4

    img_num = cols * rows

    fig = plt.figure(figsize=(20,20))



    for i in range(img_num):

        car = cars[i]

        img =  get_target_car_img(car)

        img = cv2.resize(img, (512, 512))

        ax = fig.add_subplot(rows, cols, i + 1)

        ax.imshow(img[:, :, ::-1])

        plt.axis('off')
# select one car randomly

# car = random.choice(cars)  # use this!

car = cars[10]

img = get_target_car_img(car)

plt.imshow(img[:, :, ::-1])
car_dists = [(c, car_dist(car, c)) for c in cars[:1000]]  # use only first 1000 cars because distance calculation is slow...

sorted_car_dists = sorted(car_dists, key=lambda x: x[1])
# show 16 cars with nearest orientations to the above selected car

show_cars([c[0] for c in sorted_car_dists[:16]])
def kmeans(samples, k, reduce, distance, max_iter=300):

    sample_num = len(samples)

    centroids  = [samples[i] for i in np.random.choice(sample_num, k)]

    

    for i in range(max_iter):

        dist = 0.0

        centroid_id_to_samples = defaultdict(list)



        for sample in samples:

            distances = [distance(sample, c) for c in centroids]

            nearest_id = np.argmin(np.array(distances))

            dist += distances[nearest_id]

            centroid_id_to_samples[nearest_id].append(sample)

            

        print(i, dist / sample_num)

            

        for k, v in centroid_id_to_samples.items():

            centroids[k] = reduce(v)

            

    return centroids
# code from https://github.com/christophhagen/averaging-quaternions

# https://github.com/christophhagen/averaging-quaternions/blob/master/LICENSE

def average_rotations(rots):

    # Number of quaternions to average

    M = len(rots)

    Q = np.array([q.as_quat() for q in rots])

    A = np.zeros(shape=(4, 4))



    for i in range(M):

        q = Q[i,:]

        # multiply q with its transposed version q' and add A

        A = np.outer(q,q) + A



    # scale

    A = (1.0/M)*A

    # compute eigenvalues and -vectors

    eigenValues, eigenVectors = np.linalg.eig(A)

    # Sort by largest eigenvalue

    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]

    # return the real part of the largest eigenvector (has only real part)

    return Rotation.from_quat(np.real(eigenVectors[:,0]))
rots = [Rotation.from_euler("yxz", angle) for angle in angles]
%%time

# cluster 5000 quaternions into 32 clusters

centroids = kmeans(rots[:5000], 32, average_rotations, rot_dist, max_iter=10)
centroid_id_to_cars = defaultdict(list)



for car in tqdm(cars[:5000]):

    car_rot = Rotation.from_euler("yxz", (car["r1"], car["r2"], car["r3"]))

    distances = [rot_dist(car_rot, c) for c in centroids]

    nearest_centroid_id = np.argmin(np.array(distances))

    centroid_id_to_cars[nearest_centroid_id].append(car)
# cluster 0

show_cars(random.sample(centroid_id_to_cars[0], 16))
# cluster 1

show_cars(random.sample(centroid_id_to_cars[1], 16))
# cluster 2

show_cars(random.sample(centroid_id_to_cars[2], 16))
# cluster 3

show_cars(random.sample(centroid_id_to_cars[3], 16))