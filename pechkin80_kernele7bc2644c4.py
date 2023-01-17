#!/usr/bin/env python

# coding: utf-8



# # The Nature Conservancy Fisheries Monitoring



# https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring



import os

import json

from glob import glob

import sys

import cv2

import random

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from IPython.display import HTML

import base64



import tensorflow as tf

from tensorflow import keras

import keras.backend as K



# # Загружаем разметку

import os

print(os.listdir("./"))



#CUDA_VISIBLE_DEVICES=''

# TODO: скачайте данные и сохраните в директорию:

TRAIN_PREFIX = '../input/the-nature-conservancy-fisheries-monitoring/train'

VALIDATION_PREFIX = './data/fish/test_stg1'

ORIGINAL_IMG_HEIGHT = 750

ORIGINAL_IMG_WIDTH = 1200

IMG_HEIGHT = 468#int(750/1.6)

IMG_WIDTH = 752#int(1200/1.6)



ANCHOR_WIDTH = 100#150.

ANCHOR_HEIGHT = 100#150. 



label_encoder = dict()

str_labels = []



FEATURE_SHAPE = (14,23) #(14,23)



GRID_STEP_H = IMG_HEIGHT / FEATURE_SHAPE[0]

GRID_STEP_W = IMG_WIDTH / FEATURE_SHAPE[1]



ANCHOR_CENTERS = np.mgrid[GRID_STEP_H/2:IMG_HEIGHT:GRID_STEP_H,

                          GRID_STEP_W/2:IMG_WIDTH:GRID_STEP_W]



def load_boxes(path_mask = '../input/boxesjson/boxes/boxes/*.json', prefix=TRAIN_PREFIX):

    boxes = dict()

    for path in glob(path_mask):

        label = os.path.basename(path).split('_', 1)[0]

        with open(path) as src:

            boxes[label] = json.load(src)

            for annotation in boxes[label]:

                basename = os.path.basename(annotation['filename'])

                annotation['filename'] = os.path.join(TRAIN_PREFIX, label.upper(), basename)

            for annotation in boxes[label]:

                for rect in annotation['annotations']:

                    rect['x'] += rect['width'] / 2

                    rect['y'] += rect['height'] / 2

    return boxes







def load_valid_boxes():

    return load_boxes(path_mask = './data/fish/validation_boxes/*.json',

		      prefix = VALIDATION_PREFIX)



def load_NoFiles():

    files = list()

    files = [file for file in glob('./data/fish/train/NoF/*.jpg')]

    return files



def draw_boxes(annotation, rectangles=None, image_size=None):

    

    def _draw(img, rectangles, scale_x, scale_y, color=(0, 255, 0)):

        for rect in rectangles:

            pt1 = (int((rect['x'] - rect['width'] / 2) * scale_x),

                   int((rect['y'] - rect['height'] / 2) * scale_y))

            pt2 = (int((rect['x'] + rect['width'] / 2) * scale_x),

                   int((rect['y'] + rect['height'] / 2) * scale_y))

            img = cv2.rectangle(img.copy(), pt1, pt2, 

                                color=color, thickness=4)

        return img

    

    scale_x, scale_y = 1., 1. 

    

    img = cv2.imread(annotation['filename'], cv2.IMREAD_COLOR)[...,::-1]

    if image_size is not None:

        scale_x = 1. * image_size[0] / img.shape[1]

        scale_y = 1. * image_size[1] / img.shape[0]

        img = cv2.resize(img, image_size)

        

    img = _draw(img, annotation['annotations'], scale_x, scale_y)

    

    if rectangles is not None:

        img = _draw(img, rectangles, 1., 1., (255, 0, 0))



    return img



def make_labels(aux_lb = []):

	labels = []

	for path in glob('../input/boxesjson/boxes/boxes/*.json'):

		labels.append(os.path.basename(path).split('_', 1)[0].upper())

	for str1 in aux_lb:

		labels.append(str1)

	labels_cat = pd.get_dummies(labels)

	labels_cat = labels_cat.sort_values(by=labels[0], ascending=False).values.tolist()

	label_dict = {dkey: dval for dkey, dval in zip(labels, labels_cat)}

	global label_encoder;

	label_encoder = label_dict

	global str_labels

	str_labels = labels

	return label_dict, labels



label_encoder, str_labels = make_labels([])



def get_feature_tensor():

	#features = keras.applications.vgg16.VGG16(include_top=False,

	#				          weights='imagenet',

	#				          input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

	#output = features.layers[-1].output

	#global FEATURE_SHAPE

	#FEATURE_SHAPE = (output.shape[1].value,

    #             output.shape[2].value)

    #FEATURE_SHAPE = (14, 23)

	global GRID_STEP_H

	GRID_STEP_H = IMG_HEIGHT / FEATURE_SHAPE[0]

	global GRID_STEP_W

	GRID_STEP_W = IMG_WIDTH / FEATURE_SHAPE[1]

	global ANCHOR_CENTERS

	ANCHOR_CENTERS = np.mgrid[GRID_STEP_H/2:IMG_HEIGHT:GRID_STEP_H,

		                  GRID_STEP_W/2:IMG_WIDTH:GRID_STEP_W]

	return features



def iou(rect, x_scale, y_scale, anchor_x, anchor_y,

        anchor_w=ANCHOR_WIDTH, anchor_h=ANCHOR_HEIGHT):

    

    rect_x1 = (rect['x'] - rect['width'] / 2) * x_scale

    rect_x2 = (rect['x'] + rect['width'] / 2) * x_scale

    

    rect_y1 = (rect['y'] - rect['height'] / 2) * y_scale

    rect_y2 = (rect['y'] + rect['height'] / 2) * y_scale

    

    anch_x1, anch_x2 = anchor_x - anchor_w / 2, anchor_x + anchor_w / 2

    anch_y1, anch_y2 = anchor_y - anchor_h / 2, anchor_y + anchor_h / 2

    

    dx = (min(rect_x2, anch_x2) - max(rect_x1, anch_x1))

    dy = (min(rect_y2, anch_y2) - max(rect_y1, anch_y1))

    

    intersection = dx * dy if (dx > 0 and dy > 0) else 0.

    

    anch_square = (anch_x2 - anch_x1) * (anch_y2 - anch_y1)

    rect_square = (rect_x2 - rect_x1) * (rect_y2 - rect_y1)

    union = anch_square + rect_square - intersection

    

    return intersection / union



def encode_anchors(annotation, img_shape, iou_thr=0.):

    encoded = np.zeros(shape=(FEATURE_SHAPE[0],

                              FEATURE_SHAPE[1], 5 + len(str_labels)), 

                              dtype=np.float32)

    x_scale = 1. * img_shape[1]/ORIGINAL_IMG_WIDTH

    y_scale = 1. * img_shape[0]/ORIGINAL_IMG_HEIGHT

    label = annotation['filename'].split("train/", 1)[1]

    label = label.split("img", 1)[0]

    label = label.split("/", 1)[0]

    for rect in annotation['annotations']:

        scores = []

        for row in range(FEATURE_SHAPE[0]):

            for col in range(FEATURE_SHAPE[1]):

                anchor_x = ANCHOR_CENTERS[1, row, col]

                anchor_y = ANCHOR_CENTERS[0, row, col]

                score = iou(rect, x_scale, y_scale, anchor_x, anchor_y)

                scores.append((score, anchor_x, anchor_y, row, col))

        

        scores = sorted(scores, reverse=True)

        if scores[0][0] < iou_thr:

            scores = [scores[0]]  # default anchor

        else:

            scores = [e for e in scores if e[0] > iou_thr]



        for score, anchor_x, anchor_y, row, col in scores:

            dx = (anchor_x - rect['x'] * x_scale) / ANCHOR_WIDTH

            dy = (anchor_y - rect['y'] * y_scale) / ANCHOR_HEIGHT

            dw = (ANCHOR_WIDTH - rect['width'] * x_scale) / ANCHOR_WIDTH

            dh = (ANCHOR_HEIGHT - rect['height'] * y_scale) / ANCHOR_HEIGHT

            encoded[row, col] = [1., dx, dy, dw, dh] + label_encoder[label]

            #encoded[row, col] = [1., dx, dy, dw, dh] + label_encoder[label]

    return encoded



def _sigmoid(x):

    return 1. / (1. + np.exp(-x))



def decode_prediction(prediction, conf_thr=0.1):

    rectangles = []

    conf = 0

    maxcol = 0

    maxrow = 0

    for row in range(FEATURE_SHAPE[0]):

        for col in range(FEATURE_SHAPE[1]):

            class_probabilities = list(range(len(str_labels)))

            logit =  prediction[row, col][0]

            new_conf = _sigmoid(logit)

            if (new_conf > conf):

                conf = new_conf

                maxcol = col

                maxrow = row

    logit, dx, dy, dw, dh =  prediction[maxrow, maxcol][0:5]

    conf = _sigmoid(logit)

    class_logits =  prediction[maxrow, maxcol][5:]

    class_probabilities = _sigmoid(class_logits)

    class_probab_NoF = (1 - conf)

    class_probab_Other = int(conf > 0.5)*(1 - max(class_probabilities))

    class_norma = sum(class_probabilities) + class_probab_NoF + class_probab_Other

    class_probabilities = class_probabilities/class_norma

    class_probab_NoF = class_probab_NoF/class_norma

    class_probab_Other = class_probab_Other/class_norma

    if ((class_probab_Other <= 0.5) & (class_probab_NoF <= 0.5)):

        class_label = [int(e > 0.5) for e in class_probabilities]

        class_name = [key for key in label_encoder.keys() if (label_encoder[key] == class_label)]

        class_label.append(0)

        class_label.append(0)

    elif (class_probab_Other > 0.5):

        class_label = [0, 0, 0, 0, 0, 0, 0, 1]

        class_name = "OTHER"

    elif (class_probab_NoF > 0.5):

        class_label = [0, 0, 0, 0, 0, 0, 1, 0]

        class_name = "NoF"

    class_probabilities = np.append(class_probabilities, class_probab_NoF)

    class_probabilities = np.append(class_probabilities, class_probab_Other)

    class_probabilities = np.append(class_probabilities[:4] , [class_probabilities[6:] , class_probabilities[4:6]])

    anchor_x = ANCHOR_CENTERS[1, maxrow, maxcol]

    anchor_y = ANCHOR_CENTERS[0, maxrow, maxcol]

    rectangles.append({'x': anchor_x - dx * ANCHOR_WIDTH,

	           'y': anchor_y - dy * ANCHOR_HEIGHT,

	           'width': ANCHOR_WIDTH - dw * ANCHOR_WIDTH,

	           'height': ANCHOR_HEIGHT - dh * ANCHOR_HEIGHT,

	           'conf': conf,

	           'class_probab': class_probabilities,

	           'class_name': class_name})

    return rectangles



boxes = load_boxes()



def load_img(path, target_size=(IMG_WIDTH, IMG_HEIGHT)):

    img = cv2.imread(path, cv2.IMREAD_COLOR)[...,::-1]

    img_shape = img.shape

    img_resized = cv2.resize(img, target_size)

    return img_shape, keras.applications.vgg16.preprocess_input(img_resized.astype(np.float32))



K = tf.keras.backend



batch_loss_arr = tf.Variable(0.0)



def confidence_loss(y_true, y_pred):

    #return tf.constant(-0.0)*y_pred

    #return confidence_loss(0.5, 1)

    conf_loss = K.binary_crossentropy(y_true[..., 0], 

                                      y_pred[..., 0],

                                      from_logits=True)

    return conf_loss

    #return  tf.cond(tf.equal(K.shape(y_true)[1], tf.constant(6)), 

    #               lambda: tf.constant(-0.0)*y_pred, 

    #               lambda: conf_loss)

    



def smooth_l1(y_true, y_pred):

    abs_loss = K.abs(y_true[..., 1:5] - y_pred[..., 1:5])

    square_loss = 0.5 * K.square(y_true[..., 1:5] - y_pred[..., 1:5])

    mask = K.cast(K.greater(abs_loss, 1.), 'float32')

    total_loss = (abs_loss - 0.5) * mask + 0.5 * square_loss * (1. - mask)

    return K.sum(total_loss, axis=-1)

copyten = 111



       

def total_loss(y_true, y_pred, neg_pos_ratio=3):

    batch_size = K.shape(y_true)[0]

    

    # TODO: добавьте функцию потерь для классификации детекции

    

    y_true = K.reshape(y_true, (batch_size, -1, 11))

    y_pred = K.reshape(y_pred, (batch_size, -1, 11))

    pos_mask =  y_true[...,5] == 0

    

    y_true_pos = tf.boolean_mask(y_true,y_true[...,0])

    y_pred_pos = tf.boolean_mask(y_pred,y_true[...,0])

    class_loss = K.categorical_crossentropy(y_true_pos[...,5:], 

                                                y_pred_pos[...,5:],

                                                from_logits=True,

                                                axis=-1)

    class_loss = K.mean(class_loss)

    #pos_class_loss = K.sum(class_loss * y_true[..., 0], axis=-1)

    pos_class_loss = K.mean(class_loss)

    # confidence loss

    conf_loss = K.mean(confidence_loss(y_true, y_pred))

    

    conf_loss = 0.5*conf_loss + 0.5*pos_class_loss



    # smooth l1 loss

    loc_loss = smooth_l1(y_true, y_pred)

    

    # positive examples loss

    pos_conf_loss = K.sum(conf_loss * y_true[..., 0], axis=-1)

    pos_loc_loss = K.sum(loc_loss * y_true[..., 0], axis=-1)

    

    # negative examples loss

    anchors = K.shape(y_true)[1]

    num_pos = K.sum(y_true[..., 0], axis=-1)

    num_pos_avg = K.mean(num_pos)

    num_neg = K.min([neg_pos_ratio * (num_pos_avg) + 1., K.cast(anchors, 'float32')])

    

    # hard negative mining

    neg_conf_loss, _ = tf.nn.top_k(conf_loss * (1. - y_true[..., 0]),

                                   k=K.cast(num_neg, 'int32'))



    neg_conf_loss = K.sum(neg_conf_loss, axis=-1)

    

    # total conf loss

    total_conf_loss = (neg_conf_loss + pos_conf_loss) / (num_neg + num_pos + 1e-32)

    loc_loss = pos_loc_loss / (num_pos + 1e-32)

    result = total_conf_loss + 0.5 * loc_loss

    global batch_loss_arr

    batch_loss_arr = tf.cond(tf.equal(tf.rank(batch_loss_arr) ,tf.rank(result)), 

    lambda: (result +  batch_loss_arr)/2,

    lambda: result)

    return batch_loss_arr

#def debug_loss(y_true, y_pred, neg_pos_ratio=3):

#    return tf.cond(tf.equal(tf.size(y_true), tf.constant(6)), 

#                   lambda: tf.constant(0.1)*y_pred, 

#                   lambda: total_loss(y_pred, y_true, neg_pos_ratio))

def debug_loss(y_true, y_pred, neg_pos_ratio=3):

    def debug_loss2(y_true, y_pred, neg_pos_ratio=3):

        batch_size = K.shape(y_true)[0]

        #resultten = tf.zeros_like(y_pred)*y_pred

        y_pred = tf.reshape(y_pred, (batch_size, -1,5 + len(str_labels)))

        y_true = K.reshape(y_true, (batch_size, -1, 5 + len(str_labels)))

        #rrr = tf.constant([0,0,0])#y_true([0])

        #yy_true = y_true[0,0,...]

        print_op = tf.print("y_true: ", K.shape(y_true),"y_pred: ", K.shape(y_pred))

        #print_op = tf.print("y_true: ", y_true[2])

        #y_true = K.reshape(y_true, (batch_size, -1, 5))

        print_op = tf.print("y_true: ", K.shape(y_true),"y_pred: ", K.shape(y_pred))

        with tf.control_dependencies([print_op]):

            return tf.zeros_like(y_pred)*y_pred

    return debug_loss2(y_true, y_pred, neg_pos_ratio)



#features = get_feature_tensor()

#output = features.layers[-1].output

model = keras.models.load_model('../input/model3-tesla/model3-fishes-vgg16.hdf5', 

                                        custom_objects={'total_loss': total_loss})



if True:

    example = boxes['lag'][7]

    print(example)

    _, sample_img = load_img(example['filename'])

    with tf.device('/device:CPU:0'):

        pred = model.predict(np.array([sample_img,]))[0]



    decoded = decode_prediction(pred, conf_thr=0.0)

    decoded = sorted(decoded, key=lambda e: -e['conf'])



    plt.figure(figsize=(6, 6), dpi=120)

    img = draw_boxes(example, decoded[:3], (IMG_WIDTH, IMG_HEIGHT))

    plt.imshow(img)

    #plt.title('{}x{}'.format(*img.shape));

    print(decoded[0]['class_probab'])

    plt.title(decoded[0]['class_name'][0]);

    

def make_table():

    ptable = pd.DataFrame(columns=['image', 'ALB', 'BET', 'DOL', 'LAG',

                                   'NoF', 'OTHER', 'SHARK','YFT'])

    for i, file in enumerate(glob('../input/the-nature-conservancy-fisheries-monitoring/test_stg1/*.jpg')):

        bn = os.path.basename(file)

        #bn = "test_stg1/" + bn

        _, sample_img = load_img(file)

        pred = model.predict(np.array([sample_img,]))[0]

        decoded = decode_prediction(pred, conf_thr=0.5)

        predictions = decoded[0]['class_probab']#[0, 1, 0, 1, 0, 1, 0, 1]

        print(i,predictions)

        pred_str = [str(ps) for ps in predictions]

        ptable.loc[len(ptable)] = [bn] + pred_str

        

    for i, file in enumerate(glob('../input/test-stg2/test_stg2/test_stg2/*.jpg')):

        bn = os.path.basename(file)

        bn = "test_stg2/" + bn

        _, sample_img = load_img(file)

        pred = model.predict(np.array([sample_img,]))[0]

        decoded = decode_prediction(pred, conf_thr=0.5)

        predictions = decoded[0]['class_probab']#[0, 1, 0, 1, 0, 1, 0, 1]

        print(i,predictions)

        pred_str = [str(ps) for ps in predictions]

        ptable.loc[len(ptable)] = [bn] + pred_str

    return ptable        

# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



if True:

    print("start_predictions")

    pred_table = make_table()

    pred_table.to_csv("netol_submit_stg1_stg2.csv", index=False)

    #create_download_link(pred_table)

    #print("url=",url)

    print(os.listdir("./"))



# import the modules we'll need









# create a random sample dataframe

#df = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))



# create a link to download the dataframe
