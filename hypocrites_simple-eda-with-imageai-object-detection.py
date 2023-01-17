import numpy as np

import pandas as pd

import cv2

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import os

from math import sin, cos
DATASET_DIR = '/kaggle/input/pku-autonomous-driving/'

CAM_INTRINSICS = os.path.join(DATASET_DIR, 'camera/camera_intrinsic.txt')



with open(CAM_INTRINSICS, 'r') as f:

    cam_intr = f.readlines()

print(cam_intr)



camera_matrix = np.array([[2304.5479, 0,  1686.2379],

                          [0, 2305.8757, 1354.9849],

                          [0, 0, 1]], dtype=np.float32)



df_train = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'))

df_train.head()
# @hocop1 function

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



# @hocop1 function

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

    img_zs = img_p[:, 2] # z = Distance from the camera

    return img_xs, img_ys



def hide_masked_area(img, mask, th=32):

    mask[mask >= 32] = 255

    mask[mask < 32] = 0



    img_acc = img.astype(np.int32) + mask

    img[img_acc > 255] = 255

    return img



def visualize_image(img, mask, str_coord, mask_overlay='blend'):

    

    img = visualize_bb(img, str2coords(str_coord))

    

    if mask is None:

        mask = np.zeros(img.shape, dtype=np.uint8)

    if mask_overlay=='blend':

        plt.imshow(img)

        plt.imshow(mask, alpha=0.5, cmap='gray')

    elif mask_overlay=='draw':

        img = hide_masked_area(img, mask)

        plt.imshow(img)

    elif mask_overlay=='none':

        plt.imshow(img)

        

    x, y = get_img_coords(str_coord)

    for i in range(len(x)):

        plt.text(x[i],y[i], str(i), color = 'red', fontweight = 'bold', bbox=dict(fill=False, edgecolor='red', linewidth=1))

        

def find_closest_center(xcm, ycm, xcd, ycd):

    dist_min = 100000

    dist_th = 250

    id_min = -1

    for i in range(len(xcm)):

        dist = np.sqrt((xcm[i]-xcd)**2 + (ycm[i]-ycd)**2)

        if dist < dist_min:

            dist_min = dist

            if dist_min < dist_th:

                id_min = i

    return id_min 



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



def visualize_bb(img, coords):

    x_l = 1.02

    y_l = 0.80

    z_l = 2.31

    

    img = img.copy()

    for point in coords:

        # Get values

        x, y, z = point['x'], point['y'], point['z']

        yaw, pitch, roll = -point['pitch'], -point['yaw'], -point['roll']

        # Math

        Rt = np.eye(4)

        t = np.array([x, y, z])

        Rt[:3, 3] = t

        Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T

        Rt = Rt[:3, :]

        

        P = np.array([[x_l, y_l, -z_l, 1],

                  [x_l, y_l, z_l, 1],

                  [-x_l, y_l, z_l, 1],

                  [-x_l, y_l, -z_l, 1],

                  [x_l, -y_l, -z_l, 1],

                  [x_l, -y_l, z_l, 1],

                  [-x_l, -y_l, z_l, 1],

                  [-x_l, -y_l, -z_l, 1]]).T

        

        img_cor_points = np.dot(camera_matrix, np.dot(Rt, P))

        img_cor_points = img_cor_points.T

        img_cor_points[:, 0] /= img_cor_points[:, 2]

        img_cor_points[:, 1] /= img_cor_points[:, 2]

        

        x_min = np.min(img_cor_points[:,0])

        x_max = np.max(img_cor_points[:,0])

        y_min = np.min(img_cor_points[:,1])

        y_max = np.max(img_cor_points[:,1])

        

        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 7)

    

    return img
# visualize 4 examples

idl = [4, 5, 6, 0]

plt.figure(figsize=[20, 15])

for i,id in enumerate(idl):

    img_path = os.path.join(DATASET_DIR, 'train_images', '{}.{}'.format(df_train.loc[id,'ImageId'], 'jpg'))

    tmp_im = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    mask_path = os.path.join(DATASET_DIR, 'train_masks', '{}.{}'.format(df_train.loc[id,'ImageId'], 'jpg'))

    mask = cv2.imread(mask_path)

    plt.subplot(2,2,i+1)

    visualize_image(tmp_im, mask, df_train['PredictionString'][id], mask_overlay='draw')

plt.show()

im_shape = tmp_im.shape

print(im_shape)
xs, ys = [], []



x_min = im_shape[1]

y_min = im_shape[0]

x_max = 0

y_max = 0



for i, ps in enumerate(df_train['PredictionString']):

    x, y = get_img_coords(ps)

    xs += list(x)

    ys += list(y)

    if np.min(x) < x_min:

        x_min = np.min(x)

        idx_xmin = i

    if np.min(y) < y_min:

        y_min = np.min(y)

        idx_ymin = i

    if np.max(x) > x_max:

        x_max = np.max(x)

        idx_xmax = i

    if np.max(y) > y_max:

        y_max = np.max(y)

        idx_ymax = i



print([idx_xmin, idx_xmax, idx_ymin, idx_ymax])

print([x_min, x_max, y_min, y_max])



plt.figure(figsize=(10,10))

plt.imshow(cv2.imread(DATASET_DIR + 'train_images/' + df_train['ImageId'][0] + '.jpg'), alpha=0.3)

plt.scatter(xs, ys, color='red', s=10, alpha=0.2);



idl = [idx_xmin, idx_xmax, idx_ymin, idx_ymax]

plt.figure(figsize=[20, 15])

for i,id in enumerate(idl):

    img_path = os.path.join(DATASET_DIR, 'train_images', '{}.{}'.format(df_train.loc[id,'ImageId'], 'jpg'))

    tmp_im = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    mask_path = os.path.join(DATASET_DIR, 'train_masks', '{}.{}'.format(df_train.loc[id,'ImageId'], 'jpg'))

    mask = cv2.imread(mask_path)

    plt.subplot(2,2,i+1)

    visualize_image(tmp_im, mask, df_train['PredictionString'][id], mask_overlay='draw')

plt.show()
!pip install imageai --quiet

!pip install tensorflow==1.14.0 --quiet

!pip install tensorflow-gpu==1.14.0 --quiet
from imageai.Detection import ObjectDetection





detector_ret = ObjectDetection()

detector_ret.setModelTypeAsRetinaNet()

detector_ret.setModelPath('/kaggle/input/imageaiweighs/resnet50_coco_best_v2.0.1.h5')

detector_ret.loadModel()

custom_ret = detector_ret.CustomObjects(car=True)



detector_yolo = ObjectDetection()

detector_yolo.setModelTypeAsYOLOv3()

detector_yolo.setModelPath('/kaggle/input/imageaiweighs/yolo.h5')

detector_yolo.loadModel()

custom_yolo = detector_yolo.CustomObjects(car=True)
# visualize 4 examples with object detection

idl = [4, 5, 6, 0]

fig = plt.figure(figsize=[20, 15])

for i,id in enumerate(idl):

    img_path = os.path.join(DATASET_DIR, 'train_images', '{}.{}'.format(df_train.loc[id,'ImageId'], 'jpg'))

    tmp_im = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    

    mask_path = os.path.join(DATASET_DIR, 'train_masks', '{}.{}'.format(df_train.loc[id,'ImageId'], 'jpg'))

    mask = cv2.imread(mask_path)

    img = hide_masked_area(tmp_im, mask)

    returned_image, detections = detector_ret.detectCustomObjectsFromImage(custom_objects=custom_ret, input_image=img, input_type="array", output_type="array")

    ax = fig.add_subplot(2,2,i+1)

    visualize_image(img, None, df_train['PredictionString'][id], mask_overlay='none')

    for eachObject in detections:

        box = eachObject["box_points"]

        rect = patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],linewidth=2,edgecolor='g',facecolor='none')

        ax.add_patch(rect)

plt.show()
# Yolo can be affected by part of car on which camera is installed

poly_to_hide_car = np.array([[1100,2400],[3000,2480], [3384,2640],[3384,2710], [800,2710]])



extracted_img = []

# visualize 4 examples with object detection

idl = [4, 5, 6, 0]

fig = plt.figure(figsize=[20, 15])

for i,id in enumerate(idl):

    img_path = os.path.join(DATASET_DIR, 'train_images', '{}.{}'.format(df_train.loc[id,'ImageId'], 'jpg'))

    tmp_im = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    tmp_im = cv2.fillConvexPoly(tmp_im, poly_to_hide_car, [0,0,0])

    mask_path = os.path.join(DATASET_DIR, 'train_masks', '{}.{}'.format(df_train.loc[id,'ImageId'], 'jpg'))

    mask = cv2.imread(mask_path)

    img = hide_masked_area(tmp_im, mask)

    returned_image, detections, extracted_images = detector_yolo.detectCustomObjectsFromImage(custom_objects=custom_yolo, input_image=img, input_type="array", 

                                                                                              output_type="array", extract_detected_objects=True)

    ax = fig.add_subplot(2,2,i+1)

    visualize_image(img, None, df_train['PredictionString'][id], mask_overlay='none')

    for eachObject in detections:

        box = eachObject["box_points"]

        rect = patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],linewidth=2,edgecolor='g',facecolor='none')

        ax.add_patch(rect)

    extracted_img.append(extracted_images)

plt.show()
for extracted_images in extracted_img:

    print('*')

    plt.figure(figsize=[50, 15])

    n_cars = len(extracted_images)

    for i, img in enumerate(extracted_images):

        print(img.shape)

        plt.subplot(1, n_cars, i+1)

        plt.imshow(img)

plt.show()
id = 2



img_path = os.path.join(DATASET_DIR, 'train_images', '{}.{}'.format(df_train.loc[id,'ImageId'], 'jpg'))

tmp_im = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

tmp_im = cv2.fillConvexPoly(tmp_im, poly_to_hide_car, [0,0,0])

mask_path = os.path.join(DATASET_DIR, 'train_masks', '{}.{}'.format(df_train.loc[id,'ImageId'], 'jpg'))

mask = cv2.imread(mask_path)

img = hide_masked_area(tmp_im, mask)

returned_image, detections, extracted_images = detector_yolo.detectCustomObjectsFromImage(custom_objects=custom_yolo, input_image=img, input_type="array", 

                                                                                              output_type="array", extract_detected_objects=True)

xcm, ycm = get_img_coords(df_train['PredictionString'][id])

n_det_cars = len(detections)

fig = plt.figure(figsize=[10, 20])

for i, det in enumerate(detections):

    xcd = (det['box_points'][0]+det['box_points'][2])/2

    ycd = (det['box_points'][1]+det['box_points'][3])/2

    w = (det['box_points'][2]-det['box_points'][0])/2

    h = (det['box_points'][3]-det['box_points'][1])/2

    match_id = find_closest_center(xcm, ycm, xcd, ycd)

    plt.subplot(n_det_cars,2,2*i+1)

    plt.imshow(extracted_images[i])

    plt.subplot(n_det_cars,2,2*i+2)

    if match_id != -1:

        xx = np.array(range(int(xcm[match_id]-w),int(xcm[match_id]+w)))

        xx = xx[(xx>=0) & (xx<im_shape[1])]

        matched_im = tmp_im[int(ycm[match_id]-h):int(ycm[match_id]+h),xx,:]

    else:

        matched_im = np.zeros(extracted_images[i].shape)

    plt.imshow(matched_im)

plt.show()