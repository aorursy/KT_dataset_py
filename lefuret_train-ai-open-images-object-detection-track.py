import numpy as np

import pandas as pd

from __future__ import division

from PIL import Image

import matplotlib.pyplot as plt

import matplotlib.patches as patches

from io import BytesIO

import requests

import bq_helper

from sklearn.model_selection import train_test_split

import keras.backend as K

import keras_rcnn as KC

import keras

from keras.models import Model

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Convolution2D, Flatten, MaxPooling2D, Dropout, Activation, Reshape, Input

from keras.utils import to_categorical

from keras.models import load_model

from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.applications.vgg16 import decode_predictions, VGG16

import tensorflow as tf

import queue as Q

import math

import random

import os
# Return une image depuis son URL

def images_from_url(url):

    try:

        response = requests.get(url)

        return Image.open(BytesIO(response.content))

    except:

        return False
# Permet d'afficher (pour une image, une liste de bboxs et une liste de labels) une image avec les objets labellisés

def plot_bbox_label(image, bbox, label):

    im_dim_y = image.shape[0]

    im_dim_x = image.shape[1]

    

    plt.figure(figsize=(15,20))

    fig, ax = plt.subplots(1,figsize=(15,20))

    ax.imshow(image)

    

    it = 0

    for l_bbox in bbox:

        im_width = l_bbox[2] - l_bbox[0]

        im_height = l_bbox[3] - l_bbox[1]

        

        np.random.seed(seed = int(np.prod(bytearray(label[it], 'utf8'))) %2**32)

        color = np.random.rand(3,1)

        color = np.insert(color, 3, 0.7)

        

        ax.add_patch(patches.Rectangle((l_bbox[0]*im_dim_x, l_bbox[1]*im_dim_y), im_width*im_dim_x, im_height*im_dim_y, linewidth=8, edgecolor=color, facecolor='none'));

        text = ax.annotate(label[it], (l_bbox[2]*im_dim_x,l_bbox[1]*im_dim_y), bbox=dict(boxstyle="square,pad=0.3", fc=color, lw=2))

        text.set_fontsize(18)

        it = it+1

    plt.show()
# Return le résultat de la querry possèdant toutes les informations dont nous avons besoin

# [URL, nb objets sur l'image, label de l'objet, label anglais, label numérique, x min, x max, y min, y max]

def query_dataset(size):

    print("Loading bbox dataset...")

    sub_query_images = """

    (SELECT image_id, thumbnail_300k_url

    FROM `bigquery-public-data.open_images.images`

    WHERE thumbnail_300k_url IS NOT NULL)"""

    

    sub_query_box = """

    (SELECT image_id, label_name, x_min, x_max, y_min, y_max

    FROM `bigquery-public-data.open_images.annotations_bbox`

    ORDER BY image_id

    LIMIT """ + str(size) + """)"""



    sub_query_occ_img = """

    (SELECT image_id, COUNT(*) AS nb

    FROM `bigquery-public-data.open_images.annotations_bbox`

    GROUP BY image_id

    ORDER BY image_id

    LIMIT """ + str(size) + """)"""

    

    sub_query_word = """

    (SELECT label_name, label_display_name

    FROM `bigquery-public-data.open_images.dict`)

    """



    sub_query_id_word = """

    (SELECT lab.label_name, ROW_NUMBER() OVER (ORDER BY lab.label_name) - 1 AS id

    FROM (SELECT DISTINCT(label_name) FROM """ + sub_query_box + """) lab)"""



    main_query = """

    SELECT img.thumbnail_300k_url, occ.nb, box.label_name, wrd.label_display_name, idw.id, box.x_min, box.x_max, box.y_min, box.y_max

    FROM """ + sub_query_box + """ box

    INNER JOIN """ + sub_query_occ_img + """ occ ON occ.image_id = box.image_id

    INNER JOIN """ + sub_query_images + """ img ON occ.image_id = img.image_id

    INNER JOIN """ + sub_query_word + """ wrd ON wrd.label_name = box.label_name

    INNER JOIN """ + sub_query_id_word + """ idw ON idw.label_name = box.label_name

    ORDER BY thumbnail_300k_url"""

    

    print("Dataset loaded")

    return open_images.query_to_pandas_safe(main_query)
# Charge les images et les bbox/labels au bon format dans la ram

# Cette partie contient aussi des "vestiges" pour réaliser le train d'un classifier, mais nous utilisons celui de VGG16 qui est plus performant

def load_data(data_start, data_length):

    print("Loading images from URL... (From " + str(data_start) + " to " + str(data_start + data_length) + ")")

    

    tab_image = []

    tab_list_bbox = []

    tab_list_word = []

    tab_list_labl = []

    

    # Tant qu'il reste des images dans la sous partie de notre dataset

    it_tuple_image = data_start

    while it_tuple_image < (data_start + data_length):

        list_bbox = []

        list_word = []

        list_labl = []

        

        # On récupere l'image depuis son lien

        ulr_image = dataset.thumbnail_300k_url.loc[[it_tuple_image]].iloc[0]

        image = images_from_url(ulr_image)

        

        nb_bbox = dataset.nb.loc[[it_tuple_image]].iloc[0] # Le nombre de bbox = le nombre d'objet

        

        if(image != False):

            # On resize et on normalise l'image

            image_w, image_h = image.size

            taille_max = max(image_w, image_h)

            coef = 800/taille_max

            image = image.resize((int(coef*image_w), int(coef*image_h)))

            image = np.array(image)/255



            # On traite que les images qui sont en RGB (Cela supprime aussi les images plus disponible)

            if(len(image.shape) == 3):

                # On insert l'image dans tab_image avec la valeur des pixels normalisé

                tab_image.append(image)



                # Pour chaque bbox de l'image, on la stock dans une liste qu'on stock dans tab_list_bbox

                for it_bbox in range (0, nb_bbox):

                    it_tuple_bbox = it_tuple_image + it_bbox

                    if(it_tuple_bbox < data_start + data_length):

                        list_bbox.append([dataset.x_min.loc[[it_tuple_bbox]].iloc[0], dataset.y_min.loc[[it_tuple_bbox]].iloc[0], dataset.x_max.loc[[it_tuple_bbox]].iloc[0], dataset.y_max.loc[[it_tuple_bbox]].iloc[0]])



                        one_hot = np.zeros(600)

                        one_hot[dataset.id.loc[[it_tuple_bbox]].iloc[0]] = 1

                        list_word.append(one_hot)

                        

                        label = dataset.label_display_name.loc[[it_tuple_bbox]].iloc[0]

                        list_labl.append(label)



                tab_list_bbox.append(list_bbox)

                tab_list_word.append(list_word)

                tab_list_labl.append(list_labl)

            # Pour comprendre ce saut il faut comprendre la structure de dataset_bbox

        it_tuple_image = it_tuple_image + nb_bbox

                                                               

    tab_image = np.array(tab_image)

    tab_list_bbox = np.array(tab_list_bbox)

    tab_list_word = np.array(tab_list_word)

    tab_list_labl = np.array(tab_list_labl)

    

    print("Image loaded")

    

    return [tab_image, tab_list_bbox, tab_list_word, tab_list_labl]
# Calcul l'IoU pour 2 box au format xmin ymin xmax ymax

def IoU(bbox1, bbox2):

    w_intersect = (bbox1[2] - bbox1[0]) + (bbox2[2] - bbox2[0]) - (max(bbox1[2], bbox2[2]) - min(bbox1[0], bbox2[0]))

    h_intersect = (bbox1[3] - bbox1[1]) + (bbox2[3] - bbox2[1]) - (max(bbox1[3], bbox2[3]) - min(bbox1[1], bbox2[1]))

    

    if(w_intersect < 0 or h_intersect < 0):

        return 0

    

    intersect = w_intersect * h_intersect



    union_1 = (bbox1[2]-bbox1[0]) * (bbox1[3]-bbox1[1])

    union_2 = (bbox2[2]-bbox2[0]) * (bbox2[3]-bbox2[1])

    

    union = union_1 + union_2 - intersect



    return intersect/union
# Créer des anchors avec le centre (x,y) et la largeur/hauteur de la convolution (de réduction 16)

def generate_anchors(center_x, center_y, conv_w, conv_h):

    anchor_ratio = [[1, 1], [1, 2], [2, 1]]

    anchor_coef = [1, 2, 4]

    anchor_size = 128

    

    anchor_list = []

    

    for ratio in anchor_ratio:

        for coef in anchor_coef:

            anchor_width = (anchor_size*coef*ratio[0]) / (conv_w*16)

            anchor_height = (anchor_size*coef*ratio[1]) / (conv_h*16)

            anchor_x = (center_x/conv_w) - (anchor_width/2)

            anchor_y = (center_y/conv_h) - (anchor_height/2)

            anchor = [anchor_x, anchor_y, anchor_x+anchor_width, anchor_y+anchor_height]

            

            anchor_list.append(anchor)

    

    anchor_list = np.array(anchor_list)

    

    return anchor_list
# Opération de RoI pooling sur un tableau et une taille de shape*shape

# Il y a 3 lignes similaire en fonction du type d'opération utilisé pour le pooling (apres test la moyenne est mieu que le max)

def RoI(array, shape):

    result = np.zeros((shape, shape, array.shape[2]))

    for i in range (0, shape):

        for j in range (0, shape):

            sub_array = array[int(i*array.shape[0]/shape):int((i+1)*array.shape[0]/shape), int(j*array.shape[1]/shape):int((j+1)*array.shape[1]/shape)]

            #result[i][j] = np.amax(np.amax(sub_array, axis = 0), axis = 0)

            result[i][j] = np.mean(np.mean(sub_array, axis = 0), axis = 0)

            #result[i][j] = np.amin(np.amin(sub_array, axis = 0), axis = 0)

    return result
# Extrait la partie convolution de VGG16

def generate_conv():

    vgg16_net = VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

    model = Model(input=vgg16_net.layers[0].input, output=vgg16_net.layers[17].output)

    return model
# Une accuracy custom, elle est légérement capable de dépasser 1 mais sinon avec keras,

# Pour une loss custom, l'accuracy est bugé (c'est un bug connu)

def acc(y_true, y_pred): return K.mean(K.round(y_pred)*y_true + (1.-K.round(y_pred))*(1.-y_true))
# Loss custom pour le classifier du rpn, on igniore les cas ou la prédiction est (0, 0) et on renforce l'apprentissage lors de la présence d'un objet

def custom_loss_rpn_cls(y_true, y_pred):

    shape = K.shape(y_true)

    depth = shape[0]

        

    # Retire les [0, 0] (c'est à dire entre objet et pas d'objet)

    new_y_pred = K.zeros((depth, 2))

    for i in range (0, 9):

        cond = K.equal(y_true[:, 2*i:2*i+2], [0,0])

        cond = tf.math.logical_and(cond[:,0], cond[:,1])

        cond = K.concatenate([cond, cond])

        cond = K.reshape(cond, (depth,2))

        

        temp = K.switch(cond, y_true[:, 2*i:2*i+2], y_pred[:, 2*i:2*i+2])

        if i == 0:

            new_y_pred = temp

        else:

            new_y_pred = K.concatenate([new_y_pred, temp])



    new_y_pred = K.reshape(new_y_pred, (depth, 18))

    cls = K.binary_crossentropy(y_true, new_y_pred)

    

    # Renforce les [1,0] (c'est à dire la présence d'objet)

    new_cls = K.zeros((depth, 2))

    for i in range (0, 9):

        cond = K.equal(y_true[:, 2*i:2*i+2], [0,1])

        cond = tf.math.logical_and(cond[:,0], cond[:,1])

        cond = K.concatenate([cond, cond])

        cond = K.reshape(cond, (depth,2))

        

        temp = K.switch(cond, cls[:, 2*i:2*i+2], cls[:, 2*i:2*i+2]*4.7)

        if i == 0:

            new_cls = temp

        else:

            new_cls = K.concatenate([new_cls, temp])

    new_cls = K.reshape(new_cls, (depth, 18))

    

    # On re multipli pour compenser les [0,0]

    # Les coeficients ont était trouvé expérimentalement

    return K.mean(new_cls*2.1)
# Return le model utilisé pour le classifier du rpn

def generate_rpn_cls():

    model = Sequential()

    

    model.add(Flatten(input_shape=(3, 3, 512)))

    

    model.add(Dense(512))

    model.add(Activation('relu'))

    model.add(Dropout(0.4))



    model.add(Dense(18))

    model.add(Activation('sigmoid'))

    model.add(Dropout(0.4))



    model.compile(loss=custom_loss_rpn_cls, optimizer='adam', metrics=[acc])

    

    return model
# Fonction smoothL1 (fonction connue)

def smoothL1(y_true, y_pred):

    x   = K.abs(y_true - y_pred)

    x   = K.switch(x < 1, x*x, x)

    return  x
# Loss custom pour le regresseur du rpn, on igniore les cas ou la prédiction est (0, 0)

def custom_loss_rpn_reg(y_true, y_pred):

    shape = K.shape(y_true)

    depth = shape[0]

        

    # Retire les [0, 0, 0, 0] c'est à dire qu'il n'y a pas de présence d'objet

    new_y_pred = K.zeros((depth, 4))

    for i in range (0, 9):

        cond = K.equal(y_true[:, 2*i:2*i+4], [0,0,0,0])

        cond = tf.math.logical_and(tf.math.logical_and(cond[:,0], cond[:,1]), tf.math.logical_and(cond[:,2], cond[:,3]))

        cond = K.concatenate([cond, cond, cond, cond])

        cond = K.reshape(cond, (depth,4))

        

        temp = K.switch(cond, y_true[:, 2*i:2*i+4], y_pred[:, 2*i:2*i+4])

        if i == 0:

            new_y_pred = temp

        else:

            new_y_pred = K.concatenate([new_y_pred, temp])



    new_y_pred = K.reshape(new_y_pred, (depth, 36))

    reg = smoothL1(y_true, new_y_pred)

    

    # On re multipli pour compenser les [0,0,0,0]

    # Les coeficients ont était trouvé expérimentalement

    return K.mean(reg)*9
# Return le model utilisé pour le regresseur du rpn

def generate_rpn_reg():

    model = Sequential()

    

    model.add(Flatten(input_shape=(3, 3, 512)))

    

    model.add(Dense(512))

    model.add(Activation('relu'))

    model.add(Dropout(0.4))

    

    model.add(Dense(512))

    model.add(Activation('relu'))

    model.add(Dropout(0.4))



    model.add(Dense(36))

    model.add(Activation('linear'))

    model.add(Dropout(0.4))



    model.compile(loss=custom_loss_rpn_reg, optimizer='adam', metrics=[acc])

    

    return model
# Return le model utilisé pour le classifier

def generate_cls_nn():

    vgg16_net = VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

    

    l_input = Input(shape=(7, 7, 512))

    l_flatten = vgg16_net.get_layer("flatten")

    l_fc1 = vgg16_net.get_layer("fc1")

    l_fc2 = vgg16_net.get_layer("fc2")

    l_output = vgg16_net.get_layer("predictions")

    model = Model(input=l_input, output=l_output(l_fc2(l_fc1(l_flatten(l_input)))))

    return model
# Genere les données pour les rendre compatible avec les réseaux du rpn

def generate_feature_label_rpn():

    features = []

    labels = []

    

    nb_feature_map = list_feature_map.shape[0]

    

    nb_max = []

    for i in range(0,9):

        nb_max.append(0)

    

    # Pour chaque 

    for it_feature_map in range (0, nb_feature_map):

        feature_map = list_feature_map[it_feature_map]

        

        feature_map_width = feature_map.shape[1]

        feature_map_height = feature_map.shape[0]

        

        # Chaque sliding window

        for x in range (1, feature_map_width - 1):

            for y in range (1, feature_map_height - 1):

                

                sub_labels = []

                sub_anchor = []

                

                window_valid = False

                

                list_anchors = generate_anchors(x, y, feature_map_width, feature_map_height)

                

                it_anch = -1

                good_it_anch = it_anch

                

                # Chaque anchors

                for anchor in list_anchors:

                    it_anch = it_anch + 1

                    

                    anchor_cross = not(anchor[0] >= 0 and anchor[1] >= 0 and anchor[0] + anchor[2] < 1 and anchor[1] + anchor[3] < 1)

                    anchor_cross = False

                    anchor_valid = False

                    anchor_empty = True



                    # Si l'anchor est > 0.7 , ou > 0.3 une fois

                    for bbox in data_bbox[it_feature_map]:

                        if(IoU(anchor, bbox) > 0.7):

                            anchor_valid = True

                            window_valid = True

                            break

                        if(IoU(anchor, bbox) > 0.3):

                            anchor_empty = False

                    

                    # Cas anchor valide

                    if(anchor_valid and not(anchor_cross)):

                        good_it_anch = it_anch

                        sub_labels.append(1.)

                        sub_labels.append(0.)

                        

                        # Partie coordonnées relatif à l'anchor 

                        bbox_x, bbox_y, bbox_xm, bbox_ym = bbox

                        bbox_width = bbox_xm - bbox_x

                        bbox_height = bbox_ym - bbox_y

                        anchor_x, anchor_y, anchor_xm, anchor_ym = anchor

                        anchor_width = anchor_xm - anchor_x

                        anchor_height = anchor_ym - anchor_y

                        

                        sub_anchor.append((bbox_x - anchor_x)/anchor_width)

                        sub_anchor.append((bbox_y - anchor_y)/anchor_height)

                        sub_anchor.append(math.log(bbox_width/anchor_width))

                        sub_anchor.append(math.log(bbox_height/anchor_height))

                    # Cas anchor vide

                    elif(anchor_empty and not(anchor_cross)):

                        sub_labels.append(0.)

                        sub_labels.append(1.)



                        sub_anchor.append(0.)

                        sub_anchor.append(0.)

                        sub_anchor.append(0.)

                        sub_anchor.append(0.)

                    # Cas anchor entre les 2

                    else:

                        sub_labels.append(0.)

                        sub_labels.append(0.)



                        sub_anchor.append(0.)

                        sub_anchor.append(0.)

                        sub_anchor.append(0.)

                        sub_anchor.append(0.)

                

                # Normalisation (que les window avec un objet et pas plus de 400 objet par anchor (400 est un nombre trouvé expérimentalement))

                if(window_valid and nb_max[good_it_anch] < 400):

                    nb_max[good_it_anch] = nb_max[good_it_anch] + 1

                    

                    features.append(feature_map[y-1:y+2, x-1:x+2])

                    labels.append(np.array(sub_labels + sub_anchor))

            

    features = np.array(features)

    labels = np.array(labels)



    return features, labels
# Vestige de notre classifer (celui qui remplacer vgg)

def generate_feature_label_cls():

    features = []

    labels = []

    

    counter = 0

    

    nb_image_conv = data_conv.shape[0]

    for it_image_conv in range (0, nb_image_conv):

        for it_bbox in range (0, len(data_bbox[it_image_conv])):

            list_bbox = data_bbox[it_image_conv]

            bbox = list_bbox[it_bbox]

            features.append(add_black_border(data_conv[it_image_conv][int(bbox[0]*13):int(bbox[2]*13), int(bbox[1]*13):int(bbox[3]*13)]))

            labels.append(data_word[it_image_conv][it_bbox])

            

    features = np.array(features)

    labels = np.array(labels)

    

    return features, labels
# Return la convolution d'une image

def pred_conv(image):

    return conv_net.predict(np.array([image]))[0]
# Variable dataset

open_images = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="open_images")

query_size = 80000

dataset_size = 1000

dataset = query_dataset(query_size)
conv_net = generate_conv()
rpn_cls_net = generate_rpn_cls()
rpn_reg_net = generate_rpn_reg()
cls_net = generate_cls_nn()
# Charge des images de 0 à dataset_size (1000) dans la ram, ansi que leur label

data_image, data_bbox, data_word, data_labl = load_data(0, dataset_size)

del dataset
# Transforme les images en convolution

list_feature_map = np.array(list(map(pred_conv, data_image)))

del data_image
# Genere les données pour les rendre compatible avec les réseaux du rpn

# (les données sont aussi mélangées et une partie de validation est extraite)

features_rpn, labels_rpn = generate_feature_label_rpn()

X_train_rpn, X_valid_rpn, y_train_rpn, y_valid_rpn = train_test_split(features_rpn, labels_rpn, test_size=0.1, random_state=42)

del features_rpn

del labels_rpn
# Apprentissage du réseau "régression" du rpn

rpn_reg_net.fit(X_train_rpn, y_train_rpn[:, 18:54], validation_data=(X_valid_rpn, y_valid_rpn[:, 18:54]), epochs=100, batch_size=32)
# Apprentissage du réseau "classifier" du rpn

rpn_cls_net.fit(X_train_rpn, y_train_rpn[:, 0:18], validation_data=(X_valid_rpn, y_valid_rpn[:, 0:18]), epochs=2000, batch_size=64)
# On sauvegarde les réseaux du rpn pour pouvoir les utiliser sans train

rpn_reg_net.save("reg.h5")

rpn_cls_net.save("cls.h5")
def predict(URL, threshold_cls_rpn = 0.7, threshold_cls_vgg = 0.5, threshold_iou = 0.8):

    # L'image est téléchargé et mise au bon format (le format vgg est non normalisé, celui de notre rpn l'est car nous donne de meilleurs résultats)

    image_test = images_from_url(URL)

    if(image_test != False):

        image_test_w, image_test_h = image_test.size

        taille_max = max(image_test_w, image_test_h)

        coef = 800/taille_max

        image_test = image_test.resize((int(coef*image_test_w), int(coef*image_test_h)))

        image_test_vgg = np.array(image_test)

        image_test = np.array(image_test)/255

        if(len(image_test.shape) == 3):

            anchor_test_valid = []



            image_test_conv_vgg = pred_conv(image_test_vgg)

            image_test_conv = pred_conv(image_test)



            # On passe sur chaque pixel de la convolution

            for x in range (1, image_test_conv.shape[1] - 1):

                for y in range (1, image_test_conv.shape[0] - 1):

                    anchor_valid = False

                    anchor_empty = True



                    # On effectue une prédiction sur la fenetre glissant centré sur le pixel actuel

                    pred_cls = rpn_cls_net.predict(np.array([image_test_conv[y-1:y+2, x-1:x+2]]))[0]

                    pred_reg = rpn_reg_net.predict(np.array([image_test_conv[y-1:y+2, x-1:x+2]]))[0]



                    # On test pour toutes les anchors si le rpn à detecté un objet

                    list_anchors = generate_anchors(x, y, image_test_conv.shape[1], image_test_conv.shape[0])

                    for k in range(0, 9):

                        # Si on trouve un objet à plus de 70% de sureté

                        if(pred_cls[k*2] >= threshold_cls_rpn):

                            # On recupère les infos de l'anchor

                            anchor = list_anchors[k]

                            anchor_x, anchor_y, anchor_xm, anchor_ym = anchor

                            anchor_width = anchor_xm - anchor_x

                            anchor_height = anchor_ym - anchor_y



                            # On recupère les infos de la prédiction

                            pred_reg_x, pred_reg_y, pred_reg_w, pred_reg_h = pred_reg[k*4:k*4+4]



                            # On test si l'anchor ne sort pas de l'écrant

                            cond1 = anchor_x+(pred_reg_x*anchor_width) >= 0

                            cond2 = anchor_y+(pred_reg_y*anchor_height) >= 0

                            cond3 = anchor_x+(pred_reg_x*anchor_width) + (10**pred_reg_w)*anchor_width < 1

                            cond4 = anchor_y+(pred_reg_y*anchor_height) + (10**pred_reg_h)*anchor_height < 1

                            if(cond1 and cond2 and cond3 and cond4):

                                # On calcul le xmin/max ymin/max de la prediction relativement à l'image

                                it_min_x = int((anchor_x+(pred_reg_x*anchor_width)) * image_test_conv.shape[1])

                                it_max_x = int((anchor_x+(pred_reg_x*anchor_width) + (10**pred_reg_w)*anchor_width) * image_test_conv.shape[1])

                                it_min_y = int((anchor_y+(pred_reg_y*anchor_height)) * image_test_conv.shape[0])

                                it_max_y = int((anchor_y+(pred_reg_y*anchor_height) + (10**pred_reg_h)*anchor_height) * image_test_conv.shape[0])



                                # Si la prédiction est plus large que du 7*7 (minimum du classifier)

                                if(it_max_y-it_min_y >= 7 and it_max_x-it_min_x >= 7):

                                    # Prédiction des 5 premières classes que vgg trouve

                                    label = decode_predictions(cls_net.predict(np.array([RoI(image_test_conv_vgg[it_min_y:it_max_y, it_min_x:it_max_x], 7)])), top=5)[0]

                                    # Si la confiance accordé à la top classe de vgg est de plus de 50%

                                    if(label[0][2] >= threshold_cls_vgg):

                                        # On stock les données au plus simple pour les traiter avec le nonmax

                                        anchor_test_valid.append([label[0][2], [[label[0][1], label[1][1], label[2][1], label[3][1], label[4][1]], anchor_x+(pred_reg_x*anchor_width), anchor_y+(pred_reg_y*anchor_height), anchor_x+(pred_reg_x*anchor_width) + (10**pred_reg_w)*anchor_width, anchor_y+(pred_reg_y*anchor_height) + (10**pred_reg_h)*anchor_height]])  



            # SECTION NONMAX

            # Le but est de supprimer les overlaps au dessus d'un seuil pour les mêmes classes (il suffit d'une coresspondance dans les 5 premières classes)

            anchor_test_valid = np.array(anchor_test_valid)      

            q = Q.PriorityQueue()

            for a in anchor_test_valid:

                q.put((1-a[0] + random.random()/100000,a[1]))

            anchor_test_valid = []

            size = q.qsize()

            for i in range (0, size):

                var_i = q.get()

                found_one = False

                for a in anchor_test_valid:

                    for labelnb in range(0, 5):

                        if(IoU(a[1], var_i[1][1:5]) >= 1 - threshold_iou and a[0] == var_i[1][0][labelnb]):

                            found_one = True

                if(not found_one):

                    anchor_test_valid.append([var_i[1][0][0], var_i[1][1:5]])

            anchor_test_valid = np.array(anchor_test_valid)

            if(anchor_test_valid.shape[0] != 0):

                plot_bbox_label(image_test, anchor_test_valid[:, 1], anchor_test_valid[:, 0])

            else:

                plt.figure(figsize=(15,20))

                plt.imshow(image_test_vgg)

                plt.show()
def predict_multiple(list_URL):

    for URL in list_URL:

        url_pred = predict(URL)
# UNE VOITURE :   https://www.usinenouvelle.com/mediatheque/4/5/4/000626454_image_896x598/dacia-sandero.jpg

# 2 VOITURES :   https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/s17-2051-fine-1553003760.jpg

# N VOITURES :   https://cdn-images-1.medium.com/max/1600/1*ICvAO8mPCA_sXOzW9zeM7g.jpeg

# 2 VOITURES :   https://ischool.syr.edu/infospace/wp-content/files/2015/10/toyota-and-lexus-car-on-road--e1444655872784.jpg

# ZOO : http://www.mdjunited.com/medias/images/zoo.jpg



url_images_test = ['https://www.usinenouvelle.com/mediatheque/4/5/4/000626454_image_896x598/dacia-sandero.jpg',

                   'https://images5.alphacoders.com/393/393962.jpg',

                   'https://images.unsplash.com/photo-1544776527-68e63addedf7?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&w=1000&q=80',

                   'https://www.autocar.co.uk/sites/autocar.co.uk/files/styles/gallery_slide/public/images/car-reviews/first-drives/legacy/gallardo-0638.jpg?itok=-So1NoXA', 

                   'http://www.mdjunited.com/medias/images/zoo.jpg']



predict_multiple(url_images_test)