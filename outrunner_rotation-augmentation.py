import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import os

import json

from math import sin, cos

from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
im_color = cv2.applyColorMap(np.arange(256).astype('uint8') , cv2.COLORMAP_HSV)[:,0,:]

CV_PI = np.pi



def rotateImage(img, alpha=0, beta=0, gamma=0):

    fx, dx = 2304.5479, 1686.2379

    fy, dy = 2305.8757, 1354.9849

    # get width and height for ease of use in matrices

    h, w = img.shape[:2]

    # Projection 2D -> 3D matrix

    A1 = np.array([[1/fx, 0, -dx/fx],

                   [0, 1/fx, -dy/fx],

                   [0, 0,    1],

                   [0, 0,    1]])

    

    # Rotation matrices around the X, Y, and Z axis

    RX = np.array([[1,          0,           0, 0],

             [0, cos(alpha), -sin(alpha), 0],

             [0, sin(alpha),  cos(alpha), 0],

             [0,          0,           0, 1]])

    

    RY = np.array([[cos(beta), 0, -sin(beta), 0],

              [0, 1,          0, 0],

              [sin(beta), 0,  cos(beta), 0],

              [0, 0,          0, 1]])

    RZ = np.array([[cos(gamma), -sin(gamma), 0, 0],

              [sin(gamma),  cos(gamma), 0, 0],

              [0,          0,           1, 0],

              [0,          0,           0, 1]])

    # Composed rotation matrix with (RX, RY, RZ)

    R = np.dot(RZ, np.dot(RX, RY))



    # 3D -> 2D matrix

    A2 = np.array([[fx, 0, dx, 0],

                   [0, fy, dy, 0],

                   [0, 0,   1, 0]])

    # Final transformation matrix

    trans = np.dot(A2,np.dot(R, A1))

    # Apply matrix transformation

    return cv2.warpPerspective(img, trans, (w,h), flags=cv2.INTER_LANCZOS4), trans, R



def draw_car(yaw, pitch, roll, x, y, z, overlay, color=(0,0,255)):

    yaw, pitch, roll = -pitch, -yaw, -roll

    Rt = np.eye(4)

    t = np.array([x, y, z])

    Rt[:3, 3] = t

    Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T

    Rt = Rt[:3, :]

    P = np.ones((vertices.shape[0],vertices.shape[1]+1))

    P[:, :-1] = vertices

    P = P.T

    img_cor_points = np.dot(k, np.dot(Rt, P))

    img_cor_points = img_cor_points.T

    img_cor_points[:, 0] /= img_cor_points[:, 2]

    img_cor_points[:, 1] /= img_cor_points[:, 2]

    draw_obj(overlay, img_cor_points, triangles, color)

    return overlay



def draw_obj(image, vertices, triangles, color):

    for t in triangles:

        coord = np.array([vertices[t[0]][:2], vertices[t[1]][:2], vertices[t[2]][:2]], dtype=np.int32)

#         cv2.fillConvexPoly(image, coord, (0,0,255))

        cv2.polylines(image, np.int32([coord]), 1, color)



# Load a 3D model of a car

with open('../input/pku-autonomous-driving/car_models_json/mazida-6-2015.json') as json_file:

    data = json.load(json_file)

vertices = np.array(data['vertices'])

vertices[:, 1] = -vertices[:, 1]

triangles = np.array(data['faces']) - 1



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



k = np.array([[2304.5479, 0,  1686.2379],

           [0, 2305.8757, 1354.9849],

           [0, 0, 1]], dtype=np.float32)
train = pd.read_csv("../input/pku-autonomous-driving/train.csv")
def draw_rot(img_name = 'ID_aa6ffba0a', alpha = 0, beta = 0, gamma = 0):

    img = cv2.imread('../input/pku-autonomous-driving/train_images/%s.jpg'%img_name)[:,:,::-1]

    pred_string = train[train.ImageId == img_name].PredictionString.iloc[0]

    items = pred_string.split(' ')

    items = np.array(items, dtype='float')

    model_types, yaws, pitches, rolls, xs, ys, zs = [items[i::7] for i in range(7)]



    alpha = alpha*CV_PI/180.

    beta = beta*CV_PI/180.

    gamma = gamma*CV_PI/180.



    dst, Mat,Rot = rotateImage(img, alpha, beta, gamma)

    overlay = np.zeros_like(dst)



    for yaw, pitch, roll, x, y, z in zip(yaws, pitches, rolls, xs, ys, zs):



        x,y,z,_ = np.dot(Rot,[x, y, z, 1])



        r1 = R.from_euler('xyz', [-pitch, -yaw, -roll], degrees=False)

        r2 = R.from_euler('xyz', [beta, -alpha, -gamma], degrees=False)



        pitch2, yaw2, roll2 = (r2*r1).as_euler('xyz')*(-1)



        color = im_color[np.random.randint(256)].tolist()

        overlay = draw_car(yaw2, pitch2, roll2, x, y, z, overlay, color)



        #print(np.array([yaw, pitch, roll, yaw2, pitch2, roll2]))

        #break



    plt.figure(figsize=(20,20))

    plt.imshow((dst*(np.sum(overlay, axis=-1)[:,:,np.newaxis]==0)+overlay), interpolation='lanczos')
#original

draw_rot(img_name = 'ID_aa6ffba0a', alpha = 0, beta = 0, gamma = 0)
draw_rot(img_name = 'ID_aa6ffba0a', alpha = 10, beta = 30, gamma = -10)