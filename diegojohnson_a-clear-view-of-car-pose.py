import numpy as np

import pandas as pd

import matplotlib.pylab as plt

from math import sin, cos

import cv2

import os



train = pd.read_csv('/kaggle/input/pku-autonomous-driving/train.csv')



camera_matrix = np.array([[2304.5479, 0,  1686.2379],

                          [0, 2305.8757, 1354.9849],

                          [0, 0, 1]], dtype=np.float32)





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





def str2coords(s, names=['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']):

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





def get_img_coords(s):

    '''

    Input is a PredictionString (e.g. from train dataframe)

    Output is two arrays:

        xs: x coordinates in the image

        ys: y coordinates in the image

    '''

    coords = str2coords(s)

    xs = [c['x'] for c in coords]

    ys = [c['y'] for c in coords]

    zs = [c['z'] for c in coords]

    P = np.array(list(zip(xs, ys, zs))).T

    img_p = np.dot(camera_matrix, P).T

    img_p[:, 0] /= img_p[:, 2]

    img_p[:, 1] /= img_p[:, 2]

    img_xs = img_p[:, 0]

    img_ys = img_p[:, 1]

    return img_xs, img_ys





def draw_points(image, points):

    for (p_x, p_y, p_z) in points:

        cv2.circle(image, (p_x, p_y), 10, (0, 255, 0), -1)

    return image



def visualize(img, coords, pose_name):

    x_l = 1.02

    y_l = 0.80

    z_l = 2.31

    

    img = img.copy()

    for point in coords:

        # Get values

        x, y, z = point['x'], point['y'], point['z']

        yaw, pitch, roll = point['yaw'], point['pitch'], point['roll']

        # Math

        Rt = np.eye(4)

        t = np.array([x, y, z])

        Rt[:3, 3] = t

        Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T

        Rt = Rt[:3, :]

        P = np.array([[x_l, -y_l, -z_l, 1],

                      [x_l, -y_l, z_l, 1],

                      [-x_l, -y_l, z_l, 1],

                      [-x_l, -y_l, -z_l, 1],

                      [0, 0, 0, 1]]).T

        img_cor_points = np.dot(camera_matrix, np.dot(Rt, P))

        img_cor_points = img_cor_points.T

        img_cor_points[:, 0] /= img_cor_points[:, 2]

        img_cor_points[:, 1] /= img_cor_points[:, 2]

        img_cor_points = img_cor_points.astype(int)

        # Drawing

        img = draw_points(img, img_cor_points[-1:])

        if pose_name == 'X':

            cv2.putText(img, str(int(x)), (img_cor_points[-1,0]-60, img_cor_points[-1,1]-20), cv2.FONT_HERSHEY_PLAIN, 6, (0, 255, 0), 3, cv2.LINE_AA)

        elif pose_name == 'Y':

            cv2.putText(img, str(int(y)), (img_cor_points[-1,0]-60, img_cor_points[-1,1]-20), cv2.FONT_HERSHEY_PLAIN, 6, (0, 255, 0), 3, cv2.LINE_AA)

        elif pose_name == 'Z':

            cv2.putText(img, str(int(z)), (img_cor_points[-1,0]-60, img_cor_points[-1,1]-20), cv2.FONT_HERSHEY_PLAIN, 6, (0, 255, 0), 3, cv2.LINE_AA)

        elif pose_name == 'Yaw':

            cv2.putText(img, str(round(yaw,1)), (img_cor_points[-1,0]-60, img_cor_points[-1,1]-20), cv2.FONT_HERSHEY_PLAIN, 6, (0, 255, 0), 3, cv2.LINE_AA)

        elif pose_name == 'Pitch':

            cv2.putText(img, str(round(pitch,1)), (img_cor_points[-1,0]-60, img_cor_points[-1,1]-20), cv2.FONT_HERSHEY_PLAIN, 6, (0, 255, 0), 3, cv2.LINE_AA)

        elif pose_name == 'Roll':

            cv2.putText(img, str(round(roll,1)), (img_cor_points[-1,0]-60, img_cor_points[-1,1]-20), cv2.FONT_HERSHEY_PLAIN, 6, (0, 255, 0), 3, cv2.LINE_AA)

        else:

            print('Please input right pose name (X, Y, Z, Yaw, Pitch, Roll)')

            

    return img



def display_pose(i, pose):

    img = cv2.imread('/kaggle/input/pku-autonomous-driving/train_images/' + train.iloc[i,0] + '.jpg')

    img = visualize(img, str2coords(train.iloc[i,1]), pose)

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    

    plt.figure(figsize=(20,20))

    plt.imshow(img)

    plt.title(pose)

    plt.show()

    

def display(i):

    for pose in ['Yaw', 'Pitch', 'Roll', 'X', 'Y', 'Z']:

        display_pose(i, pose)
index = 20

display(index)