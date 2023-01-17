import numpy as np

import pandas as pd

import os

import math



from matplotlib import pyplot as plt



import cv2

from PIL import Image

def image_names(partition='train'):

    return os.listdir(f'/kaggle/input/pku-autonomous-driving/{partition}_images/')



def load_img(img_name, partition='train'):

    image = Image.open(f'/kaggle/input/pku-autonomous-driving/{partition}_images/{img_name}')

    image_mask = Image.open(f'/kaggle/input/pku-autonomous-driving/{partition}_masks/{img_name}')

    return image, image_mask



def get_pred_string(img_name):

    name = img_name.split('.')[0]

    coords_str = TRAIN_DF[TRAIN_DF['ImageId'] == name]['PredictionString'].iloc[0]

    return coords_str

    

def add_mask(img, mask):

    mask_array = np.array(mask)

    masked = np.array(img)

    masked[mask_array > 0] = 255

    return Image.fromarray(masked)



def parse_pred_string(string):

    """ Converts an string to a list of arrays with the positions of the cars. """

    cars = []

    split_string = string.split()

    for i in range(0, len(split_string), 7):

        arr_i = split_string[i:i+7]

        arr_i = [float(i) for i in arr_i]

        arr_i[0] = int(arr_i[0])

        cars.append(arr_i)

    return cars



CAMERA_fx = 2304.5479

CAMERA_fy = 2305.8757

CAMERA_cx = 1686.2379

CAMERA_cy = 1354.9849



CAMERA_MATRIX = np.array([

    [CAMERA_fx,   0,        0],

    [0,        CAMERA_fy,   0],

    [0,           0,        1],

])



def to_cam_xy(world_coords):

    """ Converts world coordinates (X, Y, Z) to the projection on the images (x, y)"""

    p = np.array(world_coords)

    im_point = np.dot(p, CAMERA_MATRIX)

    im_point[:,0] /= p[:,2]

    im_point[:,1] /= p[:,2]

    

    im_point[:,0] += CAMERA_cx

    im_point[:,1] += CAMERA_cy

    

    return im_point
def get_rot_matrix(euler_rot):

    yaw, pitch, roll = euler_rot

    

    yaw, pitch, roll = -yaw, -pitch, -roll



    # The data reference edges seem to be rotated. This matrices work.

    # I got the idea of flipping thanks to: 

    # https://www.kaggle.com/zstusnoopy/visualize-the-location-and-3d-bounding-box-of-car#kln-87

    

    rot_x = np.array([

                        [1,     0,              0         ],

                        [0, math.cos(yaw), -math.sin(yaw) ],

                        [0, math.sin(yaw), math.cos(yaw)  ]

                    ])

         

    rot_y = np.array([

                        [math.cos(pitch),  0,      math.sin(pitch) ],

                        [0,                1,      0               ],

                        [-math.sin(pitch), 0,      math.cos(pitch) ]

                    ])

                 

    rot_z = np.array([

                        [math.cos(roll), -math.sin(roll), 0],

                        [math.sin(roll),  math.cos(roll), 0],

                        [0,               0,              1]

                    ])

                     

                     

    rotation_matrix = np.dot(rot_x, np.dot(rot_y, rot_z))

 

    return rotation_matrix



def get_point_arround(world_point, rotation_angles, offsets=[[0,0,2]]):

    """Adds points arround the center (world point) and rotates them to match the 

    car rotation (taking as origin world point).

    This can be used to calculate several points arround a vehicle (draw 3D bounding boxes, etc). 

    

    Params:

    world_point: numpy array [3] (x,y,z) of the car in the world.

    rotation_angles: numpy array [3]. (yaw, pitch, roll) in radians.

    offsets: List[List[3]] Points arround the world point (by default it is only one point 2 units ahead of the vehicle center).

    

    Returns:

        Numpy array with all the point(s) rotated acordingly to the center point.

    """

    rot_from_origin = np.eye(4)

    origin = world_point

    rot_from_origin[:3, 3] = origin

    rot_from_origin[:3, :3] = get_rot_matrix(rotation_angles)

    rot_from_origin = rot_from_origin[:3, :]

    

    points = np.ones((len(offsets), 4))

    points[:,:3] = np.array(offsets)

    points = points.T

            

    point = np.dot(rot_from_origin, points).T

    return point

    



    

def plot_car_directions(image, coords):

    """ Plots a point on each car and a green arrow pointing towards its direction (yaw pitch roll) 

    

    Parameters:

    image: PIL Image

    coords: Coordinate array of the cars (this is the parsed string from the dataset).

    """

    im = np.array(image)

    

    world_coords = [x[-3:] for x in coords]

    courses = [x[1:4] for x in coords]



    transformed = to_cam_xy(world_coords)



    points_directions = [get_point_arround(center_point, rotaton) for center_point, rotaton in zip(world_coords, courses)]

    transformed_dests = np.array([to_cam_xy(points) for points in points_directions])

    

    # Car position (in 2D)

    x0 = transformed[:,0]

    y0 = transformed[:,1]



    # Movement vector (yaw, pitch, roll) in 2D

    x1 = transformed_dests[:,:,0].flatten()

    y1 = transformed_dests[:,:,1].flatten()

    

    for i in range(len(world_coords)):

        im = cv2.arrowedLine(im, (int(x0[i]),int(y0[i])), (int(x1[i]),int(y1[i])), (0,255,0), 3, tipLength=0.06)

        im = cv2.circle(im, (int(x0[i]),int(y0[i])), 10, (255,0,0), -1)

    return Image.fromarray(im)
TRAIN_IMAGES = image_names('train')

TRAIN_DF = pd.read_csv('/kaggle/input/pku-autonomous-driving/train.csv')
num_images = 4

width = 2



fig, ax = plt.subplots(num_images//width, width, figsize=(39,30))



for i in range(num_images):

    test_image_name = TRAIN_IMAGES[400 + i]

    image, image_mask = load_img(test_image_name)



    coords_str = get_pred_string(test_image_name)

    coords_str = parse_pred_string(coords_str)



    world_coords = [x[-3:] for x in coords_str]

    courses = [x[1:4] for x in coords_str]



    transformed = to_cam_xy(world_coords)

    image = plot_car_directions(image, coords_str)

    ax[i//width, i%width].imshow(image)

    #ax[i//width, i%width].set_aspect('equal')

    ax[i//width, i%width].axis('off')

plt.subplots_adjust(wspace=0, hspace=0)