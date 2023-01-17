# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
### Imports ###

import os

import cv2

import csv

import pickle

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.patches as patches



from tensorflow.python.keras import backend as K

from tensorflow.python.keras.models import Model

from tensorflow.python.keras.layers import Input, Layer



import sys

sys.path.append('../input/extras/imports/')



from imports.face_detec.preprocessing import parse_annotation_csv, Config, augment

from imports.face_detec.modeling import (nn_base, rpn_layer, RoiPoolingConv, classifier_layer, union, intersection, iou,

                      calc_rpn, get_new_img_size, get_anchor_gt, rpn_loss_regr, rpn_loss_cls, 

                      class_loss_regr, class_loss_cls, non_max_suppression_fast, apply_regr_np,

                      apply_regr, calc_iou, rpn_to_roi, get_img_output_length)

from imports.face_recog.align import AlignDlib

from imports.face_recog.model_small import create_model

from imports.face_recog.triplet_loss import TripletLossLayer

### Load Data ###

images = []

images_names = []



inputs = "../input/asn10e-final-submission-coml-facial-recognition"

imgs = os.listdir(inputs)

for img_file in imgs:

    if(img_file.endswith(".jpg")):

        images_names.append(img_file)

        images.append(cv2.imread(os.path.join(inputs, img_file)))

        

plt.imshow(cv2.cvtColor(images[0],cv2.COLOR_BGR2RGB))

plt.show()
### Split input image ###

# This code must go through the input image list "images" and create a 2d array of the image cropped into 1000x1000 images

cropped = [] # A 2d list of the cropped images

image_offsets = [] # A list of the coordinates of the top left part of each corisponding cropped image so adjustments can be made to the outputs

step = 1000



for img in images:

    cropped.append([img[y:y+step,x:x+step] for y in range(0,img.shape[0],step) for x in range(0,img.shape[1],step)])

    image_offsets.append(int((img.shape[1] - 1) / 1000 + 1))

    

plt.imshow(cv2.cvtColor(cropped[0][0],cv2.COLOR_BGR2RGB))

plt.show()
# define helper functions

def format_img_size(img, C):

    """ formats the image size based on config """

    img_min_side = float(C.im_size)

    (height,width,_) = img.shape

        

    if width <= height:

        ratio = img_min_side/width

        new_height = int(ratio * height)

        new_width = int(img_min_side)

    else:

        ratio = img_min_side/height

        new_width = int(ratio * width)

        new_height = int(img_min_side)

    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    return img, ratio    



def format_img_channels(img, C):

    """ formats the image channels based on config """

    img = img[:, :, (2, 1, 0)]

    img = img.astype(np.float32)

    img[:, :, 0] -= C.img_channel_mean[0]

    img[:, :, 1] -= C.img_channel_mean[1]

    img[:, :, 2] -= C.img_channel_mean[2]

    img /= C.img_scaling_factor

    img = np.transpose(img, (2, 0, 1))

    img = np.expand_dims(img, axis=0)

    return img



def format_img(img, C):

    """ formats an image for model prediction based on config """

    img, ratio = format_img_size(img, C)

    img = format_img_channels(img, C)

    return img, ratio



# Method to transform the coordinates of the bounding box to its original size

def get_real_coordinates(ratio, x1, y1, x2, y2):

    real_x1 = int(round(x1 // ratio))

    real_y1 = int(round(y1 // ratio))

    real_x2 = int(round(x2 // ratio))

    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2 ,real_y2)



def align_image(img):

    alignment = AlignDlib('../input/models/landmarks.dat')

    bb = alignment.getLargestFaceBoundingBox(img)

    if bb is None:

        return cv2.resize(img, (96,96))

    else:

        return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img), landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)



# set up the paths

output_path = '../input/models/face_detec/'

weight_output_path = os.path.join(output_path, 'weights.hdf5') # model weights

history_output_path = os.path.join(output_path, 'history.csv') # record data (used to save the history of losses, classification accuracy and mean average precision)

config_output_path = os.path.join(output_path, 'config.pickle') # config file



# load the config file

with open(config_output_path, 'rb') as f_in:

    C = pickle.load(f_in)



# turn off any data augmentation at test time

C.use_horizontal_flips = False

C.use_vertical_flips = False

C.rot_90 = False



# apply the spatial pyramid pooling to the proposed regions

class_mapping = C.class_mapping

class_mapping = {v: k for k, v in class_mapping.items()}
### Get Face Locations ###



# Load model Face Detec

# rebuild the model

input_shape_img = (None, None, 3)

num_features = 512

input_shape_features = (None, None, num_features)



img_input = Input(shape=input_shape_img)

roi_input = Input(shape=(C.num_rois, 4))

feature_map_input = Input(shape=input_shape_features)



# define the base network (VGG here, can be Resnet50, Inception, etc)

shared_layers = nn_base(img_input, trainable=True)



# define the RPN, built on the base layers

num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)

rpn_layers = rpn_layer(shared_layers, num_anchors)



classifier = classifier_layer(feature_map_input, roi_input, C.num_rois, nb_classes=len(C.class_mapping))



model_rpn = Model(img_input, rpn_layers)

model_classifier_only = Model([feature_map_input, roi_input], classifier)



model_classifier = Model([feature_map_input, roi_input], classifier)



print('Loading weights from {}'.format(weight_output_path))

model_rpn.load_weights(weight_output_path, by_name=True)

model_classifier.load_weights(weight_output_path, by_name=True)



model_rpn.compile(optimizer='sgd', loss='mse')

model_classifier.compile(optimizer='sgd', loss='mse')



# Load model Face Recog

# Input for anchor, positive and negtive images

in_a = Input(shape=(96,96,3), name="img_a")

in_p = Input(shape=(96,96,3), name="img_p")

in_n = Input(shape=(96,96,3), name="img_n")



# create the base model from model_small

model_sm = create_model()



# Output embedding vectors from anchor, positive, negative images

# The model weights are shared (Triplet network)

emb_a = model_sm(in_a)

emb_b = model_sm(in_p)

emb_n = model_sm(in_n)



triplet_loss_layer = TripletLossLayer(alpha=0.2, name='triplet_loss_layer')([emb_a, emb_b, emb_n])



model_sm = Model([in_a, in_p, in_n], triplet_loss_layer)

model_sm.load_weights('../input/models/face_recog.hdf5') # change to best model



base_model = model_sm.layers[3]
### Loop Methods ###



def get_bboxes(R, C):

    bbox_threshold = 0.8

    bboxes = {}

    probs = {}

    for jk in range(R.shape[0]//C.num_rois + 1):

        ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)

        if ROIs.shape[1] == 0:

            break

        if jk == R.shape[0]//C.num_rois:

            #pad R

            curr_shape = ROIs.shape

            target_shape = (curr_shape[0],C.num_rois,curr_shape[2])

            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)

            ROIs_padded[:, :curr_shape[1], :] = ROIs

            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]

            ROIs = ROIs_padded

        [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

        # Calculate bboxes coordinates on resized image

        for ii in range(P_cls.shape[1]):

            # Ignore 'bg' class

            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):

                continue

            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

            if cls_name not in bboxes:

                bboxes[cls_name] = []

                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])

            try:

                (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]

                tx /= C.classifier_regr_std[0]

                ty /= C.classifier_regr_std[1]

                tw /= C.classifier_regr_std[2]

                th /= C.classifier_regr_std[3]

                x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)

            except:

                pass

            bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])

            probs[cls_name].append(np.max(P_cls[0, ii, :]))

    return bboxes, probs



def condense_bboxes(bboxes, probs, ratio):

    all_dets = []

    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

    for key in bboxes:

        bbox = np.array(bboxes[key])

        new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.2)

        

        for jk in range(new_boxes.shape[0]):

            (x1, y1, x2, y2) = new_boxes[jk,:]

            # Calculate real coordinates on original image

            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

            new_boxes[jk] = [real_x1, real_y1, real_x2, real_y2]

            all_dets.append((key,100*new_probs[jk]))

    

    return new_boxes, all_dets
face_vectors = []



for img_num, input_image in enumerate(cropped):

    print("Image: " + str(img_num))

    plt.imshow(cv2.cvtColor(input_image[0],cv2.COLOR_BGR2RGB))

    plt.show()

    

    face_vectors.append(([],[]))

    

    for cropped_num, cropped_image in enumerate(input_image):

        print("Cropped: " + str(cropped_num))

        

        X, ratio = format_img(cropped_image, C)

        X = np.transpose(X, (0, 2, 3, 1))



        # Get prediction

        [Y1, Y2, F] = model_rpn.predict(X)

        R = rpn_to_roi(Y1, Y2, C, K.image_data_format(), overlap_thresh=0.7)

        

        bboxes, probs = get_bboxes(R, C)

        if(not bboxes):

            continue

        #print(bboxes["Human face"][0])

        

        new_boxes, all_dets = condense_bboxes(bboxes, probs, ratio)

        print(len(new_boxes))

        

        for coords in new_boxes:

            temp_img = cropped_image[coords[1]:coords[3], coords[0]:coords[2]]

            temp_img = align_image(temp_img)

            temp_img = temp_img.astype('float32')

            face_vec = base_model.predict(np.expand_dims(temp_img/255.0, axis=0))

            face_vectors[img_num][0].append(face_vec[0])

            

            x_offset = step * (cropped_num % image_offsets[img_num])

            y_offset = step * int(cropped_num / image_offsets[img_num])

            x2 = min(coords[2] + x_offset, x_offset + step)

            y2 = min(coords[3] + y_offset, y_offset + step)

            

            face_vectors[img_num][1].append([coords[0] + x_offset, coords[1] + y_offset, x2, y2])

        

    #print(face_vectors[img_num])
# Load the face vectors

base_vectors = list(csv.reader(open("../input/extras/face_vects.csv")))

base_vectors = np.asarray(base_vectors, dtype=np.float64, order='C')



# Make the distance 2d list

distances = []

for img in range(len(face_vectors)):

    distances.append([])

    for vect in face_vectors[img][0]:

        distances[img].append([np.sum(np.square(vect - base_vect)) for base_vect in base_vectors])

print(len(distances[0]))
# Get people from the distance analysis

people = []

for img in range(len(distances)):

    people.append(([],[]))

    p = 0

    if(len(distances[img]) > 0):

        max_dist = np.argmax(distances[img])

        for face in range(len(distances[img])):

            min_dist = np.argmin(distances[img][face])

            people[img][0].append(min_dist)

            people[img][1].append(face_vectors[img][1][face])

            for i in range(len(distances[img])):

                distances[img][i][min_dist] = max_dist

        

print(people)
img_0 = [25,34,40,37,3,2,20,27,39,33,14,17,4,46,29,35,11,41,8,42,7,1,15,0,32,16,21,13,31,30,22,19,23,24,43,36,10,44]

img_1 = [28,26,38,45,12,9]



with open('submission.csv', 'w', newline='') as file:

    writer = csv.writer(file)

    writer.writerow(["filename", "person_id", "id", "xmin", "xmax", "ymin", "ymax"])

    for person_id in img_0:

        try:

            i = people[1][0].index(person_id)

            writer.writerow([images_names[1], person_id, "0_" + str(person_id), people[1][1][i][0], people[1][1][i][2], people[1][1][i][1], people[1][1][i][3]])

        except:

            writer.writerow([images_names[1], person_id, "0_" + str(person_id), 0,0,0,0])

    for person_id in img_1:

        try:

            i = people[0][0].index(person_id)

            writer.writerow([images_names[0], person_id, "1_" + str(person_id), people[0][1][i][0], people[0][1][i][2], people[0][1][i][1], people[0][1][i][3]])

        except:

            writer.writerow([images_names[0], person_id, "1_" + str(person_id), 0,0,0,0])
names=['Addison Waller','Alan Chen','Alexander Chu','Alexander Wu','Allen Tu','Ben Moskowitz','Caleb Yoshida','Chenyu Zhang','David Baek','Derek Zhang','Jared Habermehl','Jason Trehan','Jessica Qin','Joseph Chan','Justin Ashbaugh','Kevin Chen','Khushi Bhansali','Kyle Hassold','Maxwell Hampel','Meghana Karumuri','Mory Diaby','Nicholas Karch','Noah Grinspoon','Pranay Kuppa','Priyanka Mehta','Raymond Tu','Richard Gao','Rohan Cowlagi','Rung-Chuan Lo','Ryan Butler','Saachi Sahni','Sagar Saxena','Sakshum Kulshrestha','Sean Huang','Shreyas Srinivasan','Siddhesh Gupta','Sidharsh Joshi','Siyao Li','Siyuan Peng','Stefan Obradovic',"Tanmay Prakash", "Tianxiao Yang", "Tucker Siegel", "Viktor Murray", "Vineeth Vajipey", "Vladimir Leung","Wesley Chen"]



for img in range(len(people)):

    temp_image = cv2.imread('../input/asn10e-final-submission-coml-facial-recognition/'+images_names[img])

    plt.figure(figsize=(10,10))

    for i in range(len(people[img][0])):

        cv2.rectangle(temp_image, (people[img][1][i][0], people[img][1][i][1]), (people[img][1][i][2],people[img][1][i][3]), (120, 120, 255), 4)

        cv2.putText(temp_image, names[people[img][0][i]], (people[img][1][i][0], people[img][1][i][1]), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 1)

    plt.imshow(cv2.cvtColor(temp_image,cv2.COLOR_BGR2RGB))

    plt.show()