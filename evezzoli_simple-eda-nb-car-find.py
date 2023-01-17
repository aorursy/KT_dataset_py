#@title Import des librairies

import os

import glob

import shutil

from zipfile import ZipFile 

import glob

import json

import json

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import cv2

from math import sin, cos

import seaborn as sns

from PIL import Image, ImageOps

import os

import imp

import tensorflow as tf

import numpy as np

import random

import math

from collections import OrderedDict



from keras import models

from keras.layers import Input

from keras.layers import Convolution2D

from keras.layers import BatchNormalization

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, EarlyStopping
#@title Création des fonctions

#@markdown On crée des fonctions utiles



# Création d'un dossier

def create_repertory(rep):

    """

    fonction de création de dossier

    """

    try:

        os.mkdir(rep)

    except:

        print('Le dossier est existant')



# Affichage d'une image et son histogramme

def print_image(img):

    """

    fonction d'affichage de l'image

    """

    # On affiche l'image

    plt.figure(figsize=(20, 5))

    plt.subplot(1, 2, 1)

    plt.imshow(img)

    # On affiche l'histogramme

    plt.subplot(1, 2, 2)

    plt.hist(img.flatten(), bins=range(256))

    plt.show()
#@title Lister les dossiers

#@markdown On liste les dossiers





train_img = []

test_img = []

train_masks = []

test_masks = []

car_models_json = []



# On liste le train

files = glob.glob('/kaggle/input/pku-autonomous-driving/train_images/' + "/*.jpg")

for file in files:

    train_img.append(file)



files = glob.glob('/kaggle/input/pku-autonomous-driving/train_masks/' + "/*.jpg")

for file in files:

    train_masks.append(file)

# On liste le test

files = glob.glob('/kaggle/input/pku-autonomous-driving/test_images/' + "/*.jpg")

for file in files:

    test_img.append(file)



files = glob.glob('/kaggle/input/pku-autonomous-driving/test_masks/' + "/*.jpg")

for file in files:

    test_masks.append(file)



# On liste les json

files = glob.glob('/kaggle/input/pku-autonomous-driving/car_models_json/' + "/*.json")

for file in files:

    car_models_json.append(file)
#@title Creation des DataFrames

#@markdown On créé des dataframes



# Train_img

train_img_df = pd.DataFrame(train_img, columns= ['train_img'])

train_img_df['ImageId'] = train_img_df['train_img'].str.split('/kaggle/input/pku-autonomous-driving/train_images/', expand=True)[1].str.split('.jpg', expand=True)[0]



# Train_masks

train_masks_df = pd.DataFrame(train_masks, columns= ['train_masks'])

train_masks_df['ImageId'] = train_masks_df['train_masks'].str.split('/kaggle/input/pku-autonomous-driving/train_masks/', expand=True)[1].str.split('.jpg', expand=True)[0]



# Test_img

test_img_df = pd.DataFrame(test_img, columns= ['test_img'])

test_img_df['ImageId'] = test_img_df['test_img'].str.split('/kaggle/input/pku-autonomous-driving/test_images/', expand=True)[1].str.split('.jpg', expand=True)[0]



# Test_masks

test_masks_df = pd.DataFrame(test_masks, columns= ['test_masks'])

test_masks_df['ImageId'] = test_masks_df['test_masks'].str.split('/kaggle/input/pku-autonomous-driving/test_masks/', expand=True)[1].str.split('.jpg', expand=True)[0]



# json

car_models_json_df = pd.DataFrame(car_models_json, columns= ['car_models_json'])

car_models_json_df['ImageId'] = car_models_json_df['car_models_json'].str.split('/kaggle/input/pku-autonomous-driving/car_models_json/', expand=True)[1].str.split('.json', expand=True)[0]



# Import train.csv

train_csv = pd.read_csv("/kaggle/input/pku-autonomous-driving/train.csv")



# Concat train

train = pd.merge(train_csv, train_img_df, on='ImageId', how='outer')

train = pd.merge(train, train_masks_df, on='ImageId', how='outer')

train.head()
#@title Création du DataFrame train

images = []

model_type = []

yaw = []

pitche = []

roll = []

x = []

y = []

z = []



for i in list(range(0,train.shape[0])):

    pred_string = train.PredictionString.iloc[i]

    items = pred_string.split(' ')

    model_types, yaws, pitches, rolls, xs, ys, zs = [items[i::7] for i in range(7)]

    model_type.append(model_types)

    yaw.append(yaws)

    pitche.append(pitches)

    roll.append(rolls)

    x.append(xs)

    y.append(ys)

    z.append(zs)

    images.append(train.loc[i, 'ImageId'])

liste1 = pd.DataFrame([images, model_type, yaw, pitche, roll, x, y, z],

                     index=['ImageId', 'model_type', 'yaw', 'pitche', 'roll', 'x', 'y', 'z']).T

liste1['nb_car'] = [len(liste1['model_type'][i]) for i in range(liste1.shape[0])]

liste1.head()
#@title Distribution du nombre de voitures par image

plt.figure(figsize=(20,5))

sns.countplot(liste1['nb_car'])

plt.title('Number of cars')

plt.xlabel('Number of cars')

plt.show()
#@title Distribution des types de voitures

car_model = []

for i in range(liste1.shape[0]):

    for j in range(len(liste1['model_type'][i])):

        car_model.append(liste1['model_type'][i][j])

test = pd.DataFrame(car_model)

test[1] = 1

test = test.groupby(by=0, as_index=True).sum()

test['model_type'] = test.index

test['nb_car'] = test[1]

test = test.sort_values(by=['model_type'])

print('There is ', test[1].sum(), 'cars')



# Distribution des type de voitures

plt.figure(figsize=(20,5))

sns.barplot(x='model_type', y='nb_car', data=test, palette="deep")

plt.title('car model type')

plt.xlabel('model type')

plt.show()
#@title Distribution des yaw

yaw_list = []

for i in range(liste1.shape[0]):

    for j in range(len(liste1['yaw'][i])):

        yaw_list.append(liste1['yaw'][i][j])

test = pd.DataFrame(yaw_list)

test[0] = test[0].astype('float32')

plt.figure(figsize=(20,5))

plt.title('yaw distribution')

sns.distplot(test[0], bins=500);

plt.xlabel('yaw')

plt.show()
#@title Distribution des pitches

pitche_list = []

for i in range(liste1.shape[0]):

    for j in range(len(liste1['pitche'][i])):

        pitche_list.append(liste1['pitche'][i][j])

test = pd.DataFrame(pitche_list)

test[0] = test[0].astype('float32')

plt.figure(figsize=(20,5))

plt.title('pitche distribution')

sns.distplot(test[0], bins=500);

plt.xlabel('pitche')

plt.show()
#@title Distribution des roll

roll_list = []

for i in range(liste1.shape[0]):

    for j in range(len(liste1['roll'][i])):

        roll_list.append(liste1['roll'][i][j])

test = pd.DataFrame(roll_list)

test[0] = test[0].astype('float32')

plt.figure(figsize=(20,5))

plt.title('roll distribution')

sns.distplot(test[0], bins=500);

plt.xlabel('roll')

plt.show()
#@title Distribution des x

x_list = []

for i in range(liste1.shape[0]):

    for j in range(len(liste1['x'][i])):

        x_list.append(liste1['x'][i][j])

test = pd.DataFrame(x_list)

test[0] = test[0].astype('float32')

plt.figure(figsize=(20,5))

plt.title('x distribution')

sns.distplot(test[0], bins=500);

plt.xlabel('x')

plt.show()
#@title Distribution des y

y_list = []

for i in range(liste1.shape[0]):

    for j in range(len(liste1['y'][i])):

        y_list.append(liste1['y'][i][j])

test = pd.DataFrame(y_list)

test[0] = test[0].astype('float32')

plt.figure(figsize=(20,5))

plt.title('y distribution')

sns.distplot(test[0], bins=500);

plt.xlabel('y')

plt.show()
#@title Distribution des z

z_list = []

for i in range(liste1.shape[0]):

    for j in range(len(liste1['z'][i])):

        z_list.append(liste1['z'][i][j])

test = pd.DataFrame(z_list)

test[0] = test[0].astype('float32')

plt.figure(figsize=(20,5))

plt.title('z distribution')

sns.distplot(test[0], bins=500);

plt.xlabel('z')

plt.show()
def unpack(group):

    row = group.iloc[0]

    result = []

    data = row['PredictionString']

    while data:

        data = data.split(maxsplit=7)

        result.append(OrderedDict((

            ('image_id', row['ImageId']),

            ('model_type', int(data[0])),

            ('yaw', float(data[1])), 

            ('pitch', float(data[2])), 

            ('roll', float(data[3])), 

            ('x', float(data[4])), 

            ('y', float(data[5])),

            ('z', float(data[6]))

        )))

        data = data[7] if len(data) == 8 else ''

    return pd.DataFrame(result)



train_df = train[['ImageId', 'PredictionString']]

unpacked_train_df = train_df.groupby('ImageId', group_keys=False).apply(unpack).reset_index(drop=True)



sns.set()

sns.pairplot(unpacked_train_df)
# calculate the correlation matrix

corr = unpacked_train_df.corr()



# plot the heatmap

plt.title('Correlation y and z')

sns.heatmap(corr,

            xticklabels=corr.columns,

            yticklabels=corr.columns)
plt.title('Correlation y and z')

sns.regplot(unpacked_train_df['y'], unpacked_train_df['z'])
#@title On importe un fichier json

with open(car_models_json[0]) as json_file:

    data = json.load(json_file)

    vertices = np.array(data['vertices'])

    triangles = np.array(data['faces']) - 1

    plt.figure(figsize=(20,10))

    ax = plt.axes(projection='3d')

    ax.set_title('car_type: '+data['car_type'])

    ax.set_xlim([-4, 4])

    ax.set_ylim([-4, 4])

    ax.set_zlim([0, 3])

    ax.plot_trisurf(vertices[:,0],

                    vertices[:,2],

                    triangles,

                    -vertices[:,1], shade=True, color='blue')
#@title Augmented reality

# Load an image

img_name = train['ImageId'][50]

img = cv2.imread(f'/kaggle/input/pku-autonomous-driving/train_images/{img_name}.jpg',cv2.COLOR_BGR2RGB)[:,:,::-1]

img2 = cv2.imread(f'/kaggle/input/pku-autonomous-driving/train_masks/{img_name}.jpg',cv2.COLOR_BGR2RGB)[:,:,::-1]



# Prepare data

pred_string = train[train.ImageId == img_name].PredictionString.iloc[0]

items = pred_string.split(' ')

model_types, yaws, pitches, rolls, xs, ys, zs = [items[i::7] for i in range(7)]

liste = pd.DataFrame([model_types, yaws, pitches, rolls, xs, ys, zs],

                            index=['model_types', 'yaws', 'pitches', 'rolls', 'xs', 'ys', 'zs'])

iterations = []

for i in list(liste.columns):

    iterations.append(liste[i])



# k is camera instrinsic matrix

k = np.array([[2304.5479, 0,  1686.2379],

           [0, 2305.8757, 1354.9849],

           [0, 0, 1]], dtype=np.float32)



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



def draw_obj(image, vertices, triangles):

    for t in triangles:

        coord = np.array([vertices[t[0]][:2], vertices[t[1]][:2], vertices[t[2]][:2]], dtype=np.int32)

        # cv2.fillConvexPoly(image, coord, (0,0,255))

        cv2.polylines(image, np.int32([coord]), 1, (0,0,255))











overlay = np.zeros_like(img)

for model_type, yaw, pitch, roll, x, y, z in iterations:

    yaw, pitch, roll, x, y, z = [float(x) for x in [yaw, pitch, roll, x, y, z]]

    # I think the pitch and yaw should be exchanged

    yaw, pitch, roll = -pitch, -yaw, -roll

    Rt = np.eye(4)

    t = np.array([x, y, z])

    Rt[:3, 3] = t

    Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T

    Rt = Rt[:3, :]

    #We open the model

    with open(car_models_json[int(model_type)]) as json_file:

        data = json.load(json_file)

        vertices = np.array(data['vertices'])

        vertices[:, 1] = -vertices[:, 1]

        triangles = np.array(data['faces']) - 1



    P = np.ones((vertices.shape[0],vertices.shape[1]+1))

    P[:, :-1] = vertices

    P = P.T

    img_cor_points = np.dot(k, np.dot(Rt, P))

    img_cor_points = img_cor_points.T

    img_cor_points[:, 0] /= img_cor_points[:, 2]

    img_cor_points[:, 1] /= img_cor_points[:, 2]

    draw_obj(overlay, img_cor_points, triangles)



# Print image

alpha = .5

img = np.array(img)

cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

plt.figure(figsize=(20,20))

print_image(img)

liste



# Print mask

alpha = .5

img2 = np.array(img2)

cv2.addWeighted(overlay, alpha, img2, 1 - alpha, 0, img2)

plt.figure(figsize=(20,20))

print_image(img2)

iterations
#@title Center plot

camera_matrix = np.array([[2304.5479, 0,  1686.2379],

                          [0, 2305.8757, 1354.9849],

                          [0, 0, 1]], dtype=np.float32)

camera_matrix_inv = np.linalg.inv(camera_matrix)





def imread(path, fast_mode=False):

    img = cv2.imread(path)

    if not fast_mode and img is not None and len(img.shape) == 3:

        img = np.array(img[:, :, ::-1])

    return img





def str2coords(s):

    pred_string = s

    items = pred_string.split(' ')

    model_types, yaws, pitches, rolls, xs, ys, zs = [items[i::7] for i in range(7)]

    liste = pd.DataFrame([ model_types, yaws, pitches, rolls, xs, ys, zs],

                        index=['model_type', 'yaw', 'pitche', 'roll', 'x', 'y', 'z']).T

    coords = []

    for i in range(liste.shape[0]):

        coords.append({'id': float(liste['model_type'][i]),

                    'yaw': float(liste['yaw'][i]),

                    'pitch': float(liste['pitche'][i]),

                    'roll': float(liste['roll'][i]),

                    'x': float(liste['x'][i]),

                    'y': float(liste['y'][i]),

                    'z': float(liste['z'][i]),

                    })

    return coords



def get_img_coords(s):

    '''

    Input is a PredictionString (e.g. from train dataframe)

    Output is two arrays:

        xs: x coordinates in the image

        ys: y coordinates in the image

    '''

    coords = str2coords(s)

    xs = [float(c['x']) for c in coords]

    ys = [float(c['y']) for c in coords]

    zs = [float(c['z']) for c in coords]

    position = []

    for i in range(len(xs)):

        position.append([xs[i], ys[i], zs[i]])

    P = np.array(position).T

    img_p = np.dot(camera_matrix, P).T

    img_p[:, 0] /= img_p[:, 2]

    img_p[:, 1] /= img_p[:, 2]

    img_xs = img_p[:, 0]

    img_ys = img_p[:, 1]

    img_zs = img_p[:, 2] # z = Distance from the camera

    return img_xs, img_ys



from math import sin, cos



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



def draw_line(image, points):

    color = (255, 0, 0)

    cv2.line(image, tuple(points[0][:2]), tuple(points[3][:2]), color, 16)

    cv2.line(image, tuple(points[0][:2]), tuple(points[1][:2]), color, 16)

    cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color, 16)

    cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color, 16)

    return image





def draw_points(image, points):

    for (p_x, p_y, p_z) in points:

        cv2.circle(image, (p_x, p_y), int(1000 / p_z), (0, 255, 0), -1)

#         if p_x > image.shape[1] or p_y > image.shape[0]:

#             print('Point', p_x, p_y, 'is out of image with shape', image.shape)

    return image





def visualize(img, coords):

    # You will also need functions from the previous cells

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

        img = draw_line(img, img_cor_points)

        img = draw_points(img, img_cor_points[-1:])

    

    return img



idx = 128





fig, axes = plt.subplots(1, 2, figsize=(20,20))

img = imread('/kaggle/input/pku-autonomous-driving/train_images/' + train['ImageId'].iloc[idx] + '.jpg')

axes[0].imshow(img)

img_vis = visualize(img, str2coords(train['PredictionString'].iloc[idx]))

axes[1].imshow(img_vis)

plt.show()
# separation train test val



nb_pic = int(len(list(train['train_img'].index))*0.2)

liste_index_train = random.sample(list(train['train_img'].index), nb_pic)

a = int(len(liste_index_train)*0.2)

liste_index_test = random.sample(liste_index_train, a)

for j in liste_index_test:

    del liste_index_train[liste_index_train.index(j)]

liste_index_val = random.sample(liste_index_train, a)

for j in liste_index_val:

    del liste_index_train[liste_index_val.index(j)]
x_train = train['train_img'][liste_index_train]

x_test = train['train_img'][liste_index_test]

x_val = train['train_img'][liste_index_val]

y_train = liste1['nb_car'][liste_index_train]/44

y_test = liste1['nb_car'][liste_index_test]/44

y_val = liste1['nb_car'][liste_index_val]/44



# Pour l'entrainnement

new_train = pd.DataFrame({"x":x_train})

new_train = new_train.join(y_train)



new_test = pd.DataFrame({"x":x_test})

new_test = new_test.join(y_test)



# Pour la validation

new_val = pd.DataFrame({"x":x_val})

new_val = new_val.join(y_val)
def score(y_true, y_pred):

    if not K.is_tensor(y_pred):

        y_pred = K.constant(y_pred)

    y_true = K.cast(y_true, y_pred.dtype)

    return K.sum(K.abs(y_true/y_true)) / K.sum(K.abs(y_pred/y_true))

def sum_absolute_error(y_true, y_pred):

    if not K.is_tensor(y_pred):

        y_pred = K.constant(y_pred)

    y_true = K.cast(y_true, y_pred.dtype)

    return K.sum(K.abs(y_pred - y_true), axis=-1)
from keras.layers.merge import add, concatenate

# parametres

img_size = 512

nb_conv = 7

nb_dense = 4

units = 512

dropout = 0.5

# construction du modèle

input_shape = (1692, 1355, 3)



x = Input(shape=input_shape)

l = x

l = Convolution2D(filters=2, kernel_size=[1, 1], strides=1, activation="relu")(l)

l = BatchNormalization()(l)

l = MaxPooling2D()(l)



for i in list(range(nb_conv)): 

    l = Convolution2D(filters=2*(2**(i//2+1)), kernel_size=[3, 3], strides=1, activation="relu")(l)

    l = BatchNormalization()(l)

    l = MaxPooling2D(pool_size=(2, 2), strides=2)(l)

# Couche Flattening

l = Flatten()(l)





l = Dense(units=1024, activation="relu")(l)

l = Dropout(dropout)(l)







l = Dense(units=1, activation="linear")(l)



first_model = models.Model(x, l)

first_model.compile(loss=sum_absolute_error, optimizer='adam', metrics=[score])





first_model.summary()







train_datagen = ImageDataGenerator(rescale=1./255)



test_datagen = ImageDataGenerator(rescale=1./255)



training_set = train_datagen.flow_from_dataframe(new_train,

                                                 directory=None,

                                                 x_col='x',

                                                 y_col='nb_car',

                                                 weight_col=None,

                                                 target_size=(1692, 1355),

                                                 color_mode='rgb',

                                                 classes=None,

                                                 class_mode='raw',

                                                 batch_size=32,

                                                 shuffle=True,

                                                 seed=None,

                                                 save_to_dir=None,

                                                 save_prefix='',

                                                 save_format='jpg',

                                                 subset=None,

                                                 interpolation='nearest', 

                                                 validate_filenames=True)



test_set = test_datagen.flow_from_dataframe(new_test,

                                            directory=None,

                                            x_col='x',

                                            y_col='nb_car',

                                            weight_col=None,

                                            target_size=(1692, 1355),

                                            color_mode='rgb',

                                            classes=None,

                                            class_mode='raw',

                                            batch_size=32,

                                            shuffle=True,

                                            seed=None,

                                            save_to_dir=None,

                                            save_prefix='',

                                            save_format='jpg',

                                            subset=None,

                                            interpolation='nearest', 

                                            validate_filenames=True)



val_set = test_datagen.flow_from_dataframe(new_val,

                                            directory=None,

                                            x_col='x',

                                            y_col='nb_car',

                                            weight_col=None,

                                            target_size=(1692, 1355),

                                            color_mode='rgb',

                                            classes=None,

                                            class_mode='raw',

                                            batch_size=32,

                                            shuffle=True,

                                            seed=None,

                                            save_to_dir=None,

                                            save_prefix='',

                                            save_format='jpg',

                                            subset=None,

                                            interpolation='nearest', 

                                            validate_filenames=True)



checkpoint = ModelCheckpoint("nb_car.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)







History = first_model.fit_generator(training_set,

                                    steps_per_epoch=int(math.ceil(len(training_set.filenames)/32)), # len du jeux d'entrainement / batch

                                    epochs=20,

                                    validation_data=test_set,

                                    validation_steps=int(math.ceil(len(test_set.filenames)/32)), # len du jeux de test / batch

                                    callbacks = [checkpoint],

                                    workers=2)
plt.figure(figsize=(30,10))

y_hat = []

for i in list(range(len(val_set.filepaths))):

    test_image = Image.open(val_set.filepaths[i])

    test_image_full = np.array(test_image)

    test_image = test_image.resize((1692, 1355), resample=0)

    test_image = np.array(test_image)/255

    test_image = np.expand_dims(test_image, axis=0)

    result = first_model.predict(test_image)

    y_hat.append(int(result[0]*44))



    

plt.plot(y_hat)

plt.plot(list(liste1['nb_car'][liste_index_val]))

ecart = np.mean(np.abs(np.array(y_hat) - liste1['nb_car'][liste_index_val]))

print('mean absolute error: ', ecart)