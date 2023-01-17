# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt

import os



from collections import defaultdict





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    print('Ilość danych:', len(filenames), ': ', dirname)

   

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from os import listdir

from os.path import isfile, join

import pandas as pd

from sklearn.model_selection import train_test_split

import gc; gc.enable() # memory is tight

import random



plt.rcParams['figure.figsize'] = (25, 14)





import tensorflow as tf

import tensorflow.keras as keras

print(tf.__version__)

!nvidia-smi
# pozwoli zachowac identyczne rezultaty, oraz zachowanie się kodu, 

# dzieki czemu nie wkrada sie losowosc to naszych obliczen i za kazdym razem otrzymujemy te same wyniki

SEED = 5000



def set_seed(SEED):

    np.random.seed(SEED)

    random.seed(SEED)

    tf.random.set_seed(SEED)

    

    return SEED

    

set_seed(SEED)



BASE_DIR = '/kaggle/input/kitti-object-detection/kitti_single/training/'

LABEL_PATH = os.path.join(BASE_DIR,'label_2')

IMAGE_PATH = os.path.join(BASE_DIR,'image_2')

CALIB_PATH = os.path.join('/kaggle/input/kitti3dcalib/training','calib')





"""

IMAGE_WIDTH = 256

IMAGE_HEIGHT = 160



 WIELKOSC OBRAZKA (INPUT): TUTAJ BARDZO WAŻNE W PRZYPADKU STOSOWANIA CONVOLUCYJNYCH SIECI TYPU UNET. 

 

 Ponieważ za każdym razem gdy w warstwie obraz jest zmniejszany 2 razy, następuję zaokrąglenie rozmiaru do liczby całkowitej.

 Gdyby w którymś momencie wielkość byłaby liczbą nieparzystą. Np:  mamy taka siec

 

conv2d_66 (Conv2D)           (None, 100, 175, 32)      9248      

_________________________________________________________________

max_pooling2d_26 (MaxPooling (None, 50, 87, 32)        0   



to w przypadku UNET gdy laczymy czesc poczatkowych warstw z kolejnymi ktore wyszly, to po wyjsciu są one 

mnożone przez dwa, więc warstwa z którą będziemy chcieli połączyć będzie miała wielkość 86 i tworzenie takiego 

modelu zwróci błąd.

Należy pamiętać by wielkości były wielokrotnościami 8. (w przypadku zmniejszania 3 razy,

albo 16 w przypadku 4 operacji zmniejszania)



https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

"""



IMAGE_WIDTH = 256

IMAGE_HEIGHT = 160
images =  [(f) for f in listdir(IMAGE_PATH) if isfile(join(IMAGE_PATH, f))]

masks = [(f) for f in listdir(LABEL_PATH) if isfile(join(LABEL_PATH, f))]

masks.sort()



df = pd.DataFrame(np.column_stack([masks]), columns=['masks'])

df['images'] = df['masks'].apply(lambda x: x[:-3]+'png')





# Sprawdzamy czy wszystkie maski posiadają powiązane obrazy

no_images = df[df['images'].apply(lambda i:  i not in images)]

if(len(no_images)>0):

    print("Niektore maski nie posiadają swoich obrazów")

    print(no_images)

else:

    print("BARDZO DOBRZE: Wszystke maski posiadaja swoje obrazy")





    

img = cv2.imread(os.path.join(IMAGE_PATH, '000015.png') )    

print('Image shape: ', img.shape)

print(df.shape)

df.head()
def get_labels(label_filename):

    """

        get_labels pozwala zwrocic. bounding boxy dla kazdego samochodu. Dzięki niemu możemy zobaczyć kontory samochodow,

        zwraca liste z wartooscciami bounding box.

        

        :param label_filename: filname like kitti_3d/{training,testing}/label_2/id.txt

        Returns Pandas DataFrame

        

        The label files contain the following information, which can be read and

        written using the matlab tools (readLabels.m, writeLabels.m) provided within

        this devkit. All values (numerical or strings) are separated via spaces,

        each row corresponds to one object. The 15 columns represent:

        #Values    Name      Description

        ----------------------------------------------------------------------------

           1    type         Describes the type of object: 'Car', 'Van', 'Truck',

                             'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',

                             'Misc' or 'DontCare'

           1    truncated    Float from 0 (non-truncated) to 1 (truncated), where

                             truncated refers to the object leaving image boundaries

           1    occluded     Integer (0,1,2,3) indicating occlusion state:

                             0 = fully visible, 1 = partly occluded

                             2 = largely occluded, 3 = unknown

           1    alpha        Observation angle of object, ranging [-pi..pi]

           4    bbox         2D bounding box of object in the image (0-based index):

                             contains left, top, right, bottom pixel coordinates

           3    dimensions   3D object dimensions: height, width, length (in meters)

           3    location     3D object location x,y,z in camera coordinates (in meters)

           1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]

           1    score        Only for results: Float, indicating confidence in

                             detection, needed for p/r curves, higher is better.

    """

    data =  pd.read_csv(os.path.join(LABEL_PATH,label_filename), sep=" ", 

                       names=['label', 'truncated', 'occluded', 'alpha', 

                              'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 

                              'bbox_ymax', 'dim_height', 'dim_width', 'dim_length', 

                              'loc_x', 'loc_y', 'loc_z', 'rotation_y', 'score'])

    

    return data

    

get_labels('000002.txt')
def open_image(image_filename):

    return cv2.imread(os.path.join(IMAGE_PATH, image_filename))

    

    

def draw_box2d_id(id):

    """

        pozwala zobazyc jakie BoundingBox sa dla naszego zestawu danych

    """

    return draw_box2d(open_image(id + '.png'),

                      get_labels(id + '.txt'))



LABEL_COLORS = {

    'Car': (255,0,0), 

    'Van': (255,255,0), 

    'Truck': (255,255,255),

    'Pedestrian': (0,255,255),

    'Person_sitting': (0,255,255), 

    

    'Cyclist': (0,128,255), 

    'Tram': (128,0,0),

    'Misc': (0,255,255),

    'DontCare': (255,255,0)

}





def draw_box2d(image, labels, ax = None):

    """

        pozwala rysowac boxy 2d dla naszego modelu.

        

        

        :param label_filename: filname like kitti_3d/{training,testing}/label_2/id.txt

        Returns Pandas DataFrame

    """

    img = image.copy()

    for index, row in labels.iterrows():

        left_corner = (int(row.bbox_xmin), int(row.bbox_ymin))

        right_corner = (int(row.bbox_xmax), int(row.bbox_ymax))

        label_color = LABEL_COLORS.get(row.label,(0,255,0))

        

        img = cv2.rectangle(img, 

                            left_corner, right_corner, label_color, 1)

        img = cv2.putText(img, str(row.label), 

                          (left_corner[0] + 10, left_corner[1] - 4) , 

                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, 

                          label_color, 1)

    if ax == None:

        plt.imshow(img)

    else:

        ax.imshow(img)



draw_box2d_id('000015')
## https://github.com/charlesq34/frustum-pointnets/tree/master/kitti



def get_calibration(id):

    filepath = os.path.join(CALIB_PATH, id + '.txt')

    

    """ Read in a calibration file and parse into a dictionary.

    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py

    """

    data = {}

    with open(filepath, "r") as f:

        for line in f.readlines():

            line = line.rstrip()

            if len(line) == 0:

                continue

            key, value = line.split(":", 1)

            # The only non-float values in these files are dates, which

            # we don't care about anyway

            try:

                data[key] = np.array([float(x) for x in value.split()])

            except ValueError:

                pass

    data['P'] = np.reshape(data['P2'], [3, 4])



    return data

    

    

def rotx(t):

    """ 3D Rotation about the x-axis. """

    c = np.cos(t)

    s = np.sin(t)

    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])





def roty(t):

    """ Rotation about the y-axis. """

    c = np.cos(t)

    s = np.sin(t)

    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])





def rotz(t):

    """ Rotation about the z-axis. """

    c = np.cos(t)

    s = np.sin(t)

    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])





def project_to_image(pts_3d, P):

    """ Project 3d points to image plane.

    Usage: pts_2d = projectToImage(pts_3d, P)

      input: pts_3d: nx3 matrix

             P:      3x4 projection matrix

      output: pts_2d: nx2 matrix

      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)

      => normalize projected_pts_2d(2xn)

      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)

          => normalize projected_pts_2d(nx2)

    """

    n = pts_3d.shape[0]

    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))

    # print(('pts_3d_extend shape: ', pts_3d_extend.shape))

    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3

    pts_2d[:, 0] /= pts_2d[:, 2]

    pts_2d[:, 1] /= pts_2d[:, 2]

    return pts_2d[:, 0:2]





# label	truncated	occluded	alpha	bbox_xmin	bbox_ymin	bbox_xmax	bbox_ymax	dim_height	dim_width	

#dim_length	loc_x	loc_y	loc_z	rotation_y	score

def compute_box_3d(obj, P):

    """ Takes an object and a projection matrix (P) and projects the 3d

        bounding box into the image plane.

        Returns:

            corners_2d: (8,2) array in left image coord.

            corners_3d: (8,3) array in in rect camera coord.

    """

    # compute rotational matrix around yaw axis

    R = roty(obj.rotation_y)



    # 3d bounding box dimensions

    l = obj.dim_length

    w = obj.dim_width

    h = obj.dim_height



    # 3d bounding box corners

    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]

    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]

    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]



    # rotate and translate 3d bounding box

    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))

    # print corners_3d.shape

    corners_3d[0, :] = corners_3d[0, :] + obj.loc_x

    corners_3d[1, :] = corners_3d[1, :] + obj.loc_y

    corners_3d[2, :] = corners_3d[2, :] + obj.loc_z

    # print 'cornsers_3d: ', corners_3d

    # only draw 3d bounding box for objs in front of the camera

    if np.any(corners_3d[2, :] < 0.1):

        corners_2d = None

        return corners_2d, np.transpose(corners_3d)



    # project the 3d bounding box into the image plane

    corners_2d = project_to_image(np.transpose(corners_3d), P)

    # print 'corners_2d: ', corners_2d

    return corners_2d, np.transpose(corners_3d)





def draw_projected_box3d(image, qs, color=(0, 255, 0), thickness=2):

    """ Draw 3d bounding box in image

        qs: (8,3) array of vertices for the 3d box in following order:

            1 -------- 0

           /|         /|

          2 -------- 3 .

          | |        | |

          . 5 -------- 4

          |/         |/

          6 -------- 7

    """

    qs = qs.astype(np.int32)

    

    for k in range(0, 4):

        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html

        i, j = k, (k + 1) % 4

        # use LINE_AA for opencv3

        # cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)

        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k + 4, (k + 1) % 4 + 4

        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)



        i, j = k, k + 4

        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

    return image







def draw_box3d_id(id):

    """

        pozwala zobazyc jakie BoundingBox sa dla naszego zestawu danych

    """

    calib = get_calibration(id)

    return draw_box3d(open_image(id + '.png'),

                      get_labels(id + '.txt'), calib)





def draw_box3d(image, labels, calib, ax = None):

    """

        pozwala rysowac boxy 3d dla naszego modelu.

    

        :param label_filename: filname like kitti_3d/{training,testing}/label_2/id.txt

        Returns Pandas DataFrame

    """

    img = image.copy()

    for index, row in labels.iterrows():

        label_color = LABEL_COLORS.get(row.label,(0,255,0))

        if row.label == "DontCare":

            left_corner = (int(row.bbox_xmin), int(row.bbox_ymin))

            right_corner = (int(row.bbox_xmax), int(row.bbox_ymax))

            img = cv2.rectangle(img, 

                            left_corner, right_corner, label_color, 1)

            continue

            

        box3d_pts_2d, _ = compute_box_3d(row, calib['P'])

        if box3d_pts_2d is None:

            continue

        

        img = draw_projected_box3d(img, box3d_pts_2d , label_color, 1)



    if ax == None:

        plt.imshow(img)

    else:

        ax.imshow(img)

        



draw_box3d_id('000015')
DTYPE = np.float64



def get_image(path):

    return cv2.imread(path)



LABEL_MASKS = {

    'Car': 0, 

    'Van': 1, 

    'Truck': 1,

    'Pedestrian': 2,

    'Person_sitting': 2, 

    

    'Cyclist': 3, 

    'Tram': 4,

    

    'Misc': 5,

    'DontCare': 5

}



LABEL_MASKS_COLORS = [

    (255,0,0), 

    (255,255,0), 

    (0,255,255),

    

    (0,128,255), 

    (128,0,0),

    

    (255,255,0)

]





LABEL_MASKS_LENGTH = 6





def create_mask(mask_dir, img_shape):

    """

        pozwala wygenerowac maske, Poniewaz w przypadku modelu UNET na wyjsciu chcemy miec zwykla maske obiektu,

        gdzie w pelni zaznaczymy gdzie są obiekty.

        

        Uwzględniliśmy w nich samochód, van, track, oraz osoby.

    """

    mask = np.zeros(shape=(img_shape[0], img_shape[1], LABEL_MASKS_LENGTH), dtype = DTYPE)

    



    with open(mask_dir) as f:

        content = f.readlines()

    content = [x.split() for x in content] 

    for item in content:

        if item[0] in LABEL_MASKS:

            ul_col, ul_row = int(float(item[4])), int(float(item[5]))

            lr_col, lr_row = int(float(item[6])), int(float(item[7]))

            

            mask[ul_row:lr_row, ul_col:lr_col, LABEL_MASKS[item[0]]] = 1 

    return mask



def get_mask_rgb(mask):

    rgb_mask = np.zeros(shape = (mask.shape[0],mask.shape[1], 3))

    for m in range(mask.shape[2]):

        m_array = mask[:,:, m]

        rgb_mask[:,:,0] += mask[:,:, m] * LABEL_MASKS_COLORS[m][0]

        rgb_mask[:,:,1] += mask[:,:, m] * LABEL_MASKS_COLORS[m][1]

        rgb_mask[:,:,2] += mask[:,:, m] * LABEL_MASKS_COLORS[m][2]

    return rgb_mask



def draw_mask(image, mask, ax = None):

    rgb_mask = get_mask_rgb(mask) / 255.0

    img = cv2.addWeighted( image, 0.9, rgb_mask, 0.5, 0)

    if ax == None:

        plt.imshow(img)

    else:

        ax.imshow(img)

        

def draw_mask_id(id):

    # ponieważ maska to wartości 0,1 a obraz to wartości 0-255, to obraz trzeba podzielić przez 255.0

    img = np.array(get_image(os.path.join(IMAGE_PATH,id + '.png')) / 255.0, dtype = DTYPE) 

    mask = create_mask(os.path.join(LABEL_PATH,  id + '.txt'), img.shape )

    draw_mask(img,mask)

    plt.show()

    

draw_mask_id('000015')
from abc import ABCMeta, abstractmethod

import imgaug.augmenters as iaa

from imgaug.augmentables.segmaps import SegmentationMapsOnImage







class DataGenerator(tf.keras.utils.Sequence):

    """

        Ta abstrakcyjna klasa to jest wszystko co trzeba zaprogramować dla ImageDataGenerator.

        Dodaliśmy metodę get_data(), która pobierze nam potrzeby batch size.

    """

    def __init__(self,indices, batch_size, shuffle):

        self.indices = indices

        self.batch_size = batch_size

        self.shuffle = shuffle

        

        self.on_epoch_end()

        

    def __len__(self):

        return len(self.indices) // self.batch_size



    def __getitem__(self, index):

        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]

        batch = [self.indices[k] for k in index]

        

        return self.get_data(batch)

    

    def sample(self):

        return self[random.randint(0,len(self))]



    def on_epoch_end(self):

        self.index = np.arange(len(self.indices))

        if self.shuffle == True:

            np.random.shuffle(self.index)



    @abstractmethod

    def get_data(self, batch):

        raise NotImplementedError
       

class KittiDataGenerator(DataGenerator):

    """

        Bounding Box dla obiektow.

    """

    def __init__(self, 

                 df,

                 shape = (IMAGE_WIDTH, IMAGE_HEIGHT),

                 

                 batch_size=32, 

                 shuffle=True, 

                 

                 augmentation = None,

                 

                 image_col = 'images',

                 mask_col = 'masks',

                 

                 label_path = LABEL_PATH,

                 image_path = IMAGE_PATH

                ):

        

        self.df = df

        self.shape = shape



        self.image_col = image_col

        self.mask_col = mask_col

        self.label_path = label_path

        self.image_path = image_path

        self.augmentation = augmentation

        

        super().__init__(self.df.index.tolist(), batch_size, shuffle)

    

    def get_x(self, index):

        return get_image(

            os.path.join(self.image_path, self.df.loc[index][self.image_col])

        ) 

    

    def get_y(self, index, shape):

        return create_mask(os.path.join(self.label_path, self.df.loc[index][self.mask_col]),shape) 





    def get_data(self, batch):

        batch_X = []

        batch_y = []

        

        for i in batch:

            image_r = self.get_x(i)

            mask_r = self.get_y(i, image_r.shape)

            

            # w przypadku cv2.resize, odwracamy wielkosc, najpierw jest wysokosc, potem szerokosc

            image_r = cv2.resize(image_r, (self.shape[0], self.shape[1]))

            # mask_r po przekształceniu nie będzie posiadac wartości 0,1 a wartości między 0 a 1

            mask_r = cv2.resize(mask_r,(self.shape[0], self.shape[1]))

            

            if self.augmentation is not None:

                mask_r = np.array(mask_r, dtype=np.int32)

                segmap = SegmentationMapsOnImage(mask_r, shape=image_r.shape)

                image_r, segmap = self.augmentation(image = image_r, segmentation_maps = segmap)

                mask_r = segmap.get_arr().astype(np.float64)

                

                

            batch_X.append(image_r)

            batch_y.append(mask_r)

        

        batch_X = np.array(batch_X)

        batch_y = np.array(batch_y)

        



#         if self.augmentation is not None:

#             print(batch_y.shape)

#             batch_y = np.expand_dims(batch_y, axis=3)

#             print(batch_y.shape)

            

#             batch_X, segmap = 

            

#             batch_y = segmap

#             print(batch_y.shape)

#             batch_y = segmap.draw()



        return batch_X  / 255.0, batch_y

    

# Define our augmentation pipeline.

# seq = iaa.Sequential([

#     iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels

#     iaa.Sharpen((0.0, 1.0)),       # sharpen the image

#     iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects segmaps)

#     iaa.ElasticTransformation(alpha=50, sigma=5)  # apply water effect (affects segmaps)

# ], random_order=True)



seq = iaa.Sequential([

    iaa.Dropout([0.00, 0.01]),      # drop 5% or 20% of all pixels

    iaa.Sharpen((0.0, 0.05)),       # sharpen the image

    iaa.Multiply((0.8, 0.9), per_channel=0.5), # brightness

    iaa.Affine(rotate=(-3, 3)),  # rotate by -45 to 45 degrees (affects segmaps)

    iaa.GammaContrast((0.5, 1.0))  

])

    

image_generator = KittiDataGenerator(df, 

                                     shape = (IMAGE_WIDTH, IMAGE_HEIGHT),

                                     augmentation = seq)

batch_X,batch_y = image_generator.sample()

print('Input : ', batch_X.shape, batch_X.dtype,' max: ', batch_X.max(),  

      '\nOutput: ', batch_y.shape, batch_y.dtype, ' max: ', batch_y.max())
#Keras

SMOOTH = 1e-5

from tensorflow.keras import backend as K

from typing import Callable, Union





def calc_IOU(y_true, y_pred, smooth=1): 

    y_true_f = y_true

    y_pred_f = y_pred

    

    intersection = keras.backend.sum(y_true_f*y_pred_f)

    

    return (2*(intersection + smooth) / (keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) + smooth))





def calc_IOU_loss(y_true, y_pred):

    return -calc_IOU(y_true, y_pred)







# dice_coef_cat = dice_coef_cat_fn(num_classes = LABEL_MASKS_LENGTH)

# dice_coef_cat_loss = dice_coef_cat_loss_fn(num_classes = LABEL_MASKS_LENGTH)



print('IOU: ', calc_IOU_loss(batch_y,batch_y) )
def dice(y_pred, y_true):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)





def fbeta(y_pred, y_true):



    pred0 = keras.layers.Lambda(lambda x : x[:,:,:,0])(y_pred)

    pred1 = keras.layers.Lambda(lambda x : x[:,:,:,1])(y_pred)

    true0 = keras.layers.Lambda(lambda x : x[:,:,:,0])(y_true)

    true1 = keras.layers.Lambda(lambda x : x[:,:,:,1])(y_true) # channel last?

    

    y_pred_0 = K.flatten(pred0)

    y_true_0 = K.flatten(true0)

    

    y_pred_1 = K.flatten(pred1)

    y_true_1 = K.flatten(true1)

    

    intersection0 = K.sum(y_true_0 * y_pred_0)

    intersection1 = K.sum(y_true_1 * y_pred_1)



    precision0 = intersection0/(K.sum(y_pred_0)+K.epsilon())

    recall0 = intersection0/(K.sum(y_true_0)+K.epsilon())

    

    precision1 = intersection1/(K.sum(y_pred_1)+K.epsilon())

    recall1 = intersection1/(K.sum(y_true_1)+K.epsilon())

    

    fbeta0 = (1.0+0.25)*(precision0*recall0)/(0.25*precision0+recall0+K.epsilon())

    fbeta1 = (1.0+4.0)*(precision1*recall1)/(4.0*precision1+recall1+K.epsilon())

    

    return ((fbeta0+fbeta1)/2.0)



def fbeta_loss(y_true, y_pred):

    return 1-fbeta(y_true, y_pred)



def dice_loss(y_true, y_pred):

    return 1-dice(y_true, y_pred)





def weighted_categorical_crossentropy(weights):

    """

    A weighted version of keras.objectives.categorical_crossentropy

    

    Variables:

        weights: numpy array of shape (C,) where C is the number of classes

    

    Usage:

        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.

        loss = weighted_categorical_crossentropy(weights)

        model.compile(loss=loss,optimizer='adam')

    """

    weights = K.variable(weights)

        

    def loss(y_true, y_pred):

        # scale predictions so that the class probas of each sample sum to 1

        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # clip to prevent NaN's and Inf's

        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        # calc

        loss = y_true * K.log(y_pred) # * weights

        loss = -K.sum(loss, -1)

        return loss

    

    return loss



def cat_dice_loss_fn(weights_n = []):

    f = weighted_categorical_crossentropy(weights_n)

    def fun(y_true, y_pred):

        return f(y_true,y_pred) + dice_loss(y_true, y_pred) # + fbeta_loss(y_true, y_pred)

    return fun



print('Fbeta: ', fbeta_loss(batch_y,batch_y))

print('Dice Loss: ', dice_loss(batch_y,batch_y))



cat_dice_loss = cat_dice_loss_fn(weights_n = np.ones(LABEL_MASKS_LENGTH, dtype=np.float64))

print('Cat Dice Loss: ', cat_dice_loss(batch_y,batch_y).shape)
def show_data(X,y, y_pred = None):

    if y_pred is None:

        y_pred = y

        

    for x_i,y_i,y_pred_i in zip(X,y,y_pred):

        im = np.array(255*x_i,dtype=np.uint8)

        im_mask = np.array(255*y_i,dtype=np.uint8)



        rgb_mask_pred = np.array(get_mask_rgb(y_pred_i),dtype=np.uint8)

        rgb_mask_true      = np.array(get_mask_rgb(y_i),dtype=np.uint8)

        

        img_pred = cv2.addWeighted(rgb_mask_pred,0.5,im,0.5,0)

        img_true = cv2.addWeighted(rgb_mask_true,0.5,im,0.5,0)

        

        loss = calc_IOU_loss(np.array([y_i], dtype=y_i.dtype),np.array([y_pred_i], dtype=y_i.dtype))

    

        plt.figure(figsize=(20,8))

        plt.subplot(1,3,1)

        plt.imshow(im)

        plt.title('Original image')

        plt.axis('off')

        plt.subplot(1,3,2)

        plt.imshow(img_pred)

        plt.title(f'Predicted masks {loss:0.4f}')

        plt.axis('off')

        plt.subplot(1,3,3)

        plt.imshow(img_true)

        plt.title('ground truth datasets')

        plt.axis('off')

        plt.tight_layout(pad=0)

        plt.show()



show_data(batch_X[:4], batch_y[:4])
set_seed(SEED)



df_train, df_val = train_test_split(df, test_size=0.25, shuffle=True)

df_train.head()
kitti_train = KittiDataGenerator(df_train,

                                 shape = (IMAGE_WIDTH, IMAGE_HEIGHT), augmentation = seq)



kitti_val =  KittiDataGenerator(df_val,

                                 shape = (IMAGE_WIDTH, IMAGE_HEIGHT))



batch_X, batch_y = kitti_train.sample()

show_data(batch_X[:4], batch_y[:4])
from tensorflow.keras import layers

from tensorflow.keras import models

from tensorflow.keras.optimizers import Adam





def upsample_conv(filters, kernel_size, strides, padding):

    return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)



def upsample_simple(filters, kernel_size, strides, padding):

    return layers.UpSampling2D(strides)





def create_model(output_classes, 

                 calc_loss = dice_loss,

                 image_width = IMAGE_WIDTH,image_height = IMAGE_HEIGHT, 

                 upsample = upsample_simple):

    # input_img = layers.Input(batch_img.shape[1:], name = 'RGB_Input')

    input_img = layers.Input((image_height, image_width,3), name = 'RGB_Input')

    pp_in_layer = input_img

             

    c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(pp_in_layer) # 

    c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(c1)

    p1 = layers.MaxPooling2D((2, 2))(c1)

    

    c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(p1)

    c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c2)

    p2 = layers.MaxPooling2D((2, 2))(c2)

    

    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p2)

    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c3)

    p3 = layers.MaxPooling2D((2, 2)) (c3)

    

    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p3)

    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c4)

    p4 = layers.MaxPooling2D(pool_size=(2, 2)) (c4)

    

    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p4)

    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

    

    

    u6 = upsample(64, (2, 2), strides=(2, 2), padding='same')(c5)

    

    u6 = layers.concatenate([u6, c4])

    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u6)

    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

    

    u7 = upsample(32, (2, 2), strides=(2, 2), padding='same')(c6)

    u7 = layers.concatenate([u7, c3])

    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u7)

    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

    

    u8 = upsample(16, (2, 2), strides=(2, 2), padding='same')(c7)

    u8 = layers.concatenate([u8, c2])

    c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u8)

    c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c8)

    

    u9 = upsample(8, (2, 2), strides=(2, 2), padding='same') (c8)

    u9 = layers.concatenate([u9, c1], axis=3)

    c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(u9)

    c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(c9)

    

    d = layers.Conv2D(output_classes, (1, 1), activation='sigmoid')(c9)

    seg_model = models.Model(inputs=[input_img], outputs=[d])



    

    seg_model.compile(optimizer=Adam(lr=1e-4),

              loss=calc_loss, 

                metrics=[calc_IOU, dice, fbeta])

    

    seg_model.summary()

    

    return seg_model



model = create_model(LABEL_MASKS_LENGTH, calc_loss = cat_dice_loss)
keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)
!pip install livelossplot

from livelossplot import PlotLossesKeras

print(PlotLossesKeras)
tf.test.is_gpu_available()
loss_history =  model.fit(

        kitti_train,

#         steps_per_epoch=20,

        epochs=20,

        validation_data=kitti_val,

        callbacks=[ PlotLossesKeras(), 

                  tf.keras.callbacks.ReduceLROnPlateau(

                      monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto',

                      min_delta=0.0001, cooldown=0, min_lr=0)

                  ]

    )
#!ls /kaggle/output
model.save_weights('/kaggle/output/model_v1')
X_batch, y_batch = kitti_val.sample()



X_batch = X_batch[:4]

y_batch = y_batch[:4]



print(X_batch.shape)
y_batch_pred = model.predict(X_batch)



y_batch_pred_treshold = tf.round(y_batch_pred)

print(X_batch.shape, y_batch.shape, y_batch_pred_treshold.shape)





# print(calc_IOU_loss(y_batch,y_batch_pred))
show_data(X_batch, y_batch,  y_batch_pred_treshold)