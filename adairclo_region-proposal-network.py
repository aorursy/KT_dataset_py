import numpy as np

import pandas as pd

import os

import time

import pydicom

import matplotlib.pyplot as plt

import cv2

import tensorflow.keras as keras

import keras.backend as K
def rect_convert_x1y1wh_to_x1y1x2y2(rect_x1y1wh):

    rect = np.array(rect_x1y1wh)



    if rect.ndim == 1:

        rect = rect.reshape((1, 4))



    rect[:, 2] = rect[:, 0] + rect[:, 2]

    rect[:, 3] = rect[:, 1] + rect[:, 3]



    if rect.shape[0] == 1:

        rect = rect.reshape(-1)



    return rect



def rect_convert_x1y1wh_to_cxcywh(rect_x1y1wh):

    rect = np.array(rect_x1y1wh)



    if rect.ndim == 1:

        rect = rect.reshape((1, 4))



    rect[:, 0] = rect[:, 0] + rect[:, 2] // 2

    rect[:, 1] = rect[:, 1] + rect[:, 3] // 2



    if rect.shape[0] == 1:

        rect = rect.reshape(-1)



    return rect



def rect_convert_cxcywh_to_x1y1x2y2(rect_cxcywh):

    rect = np.array(rect_cxcywh)



    if rect.ndim == 1:

        rect = rect.reshape((1, 4))



    rect[:, 0] = rect[:, 0] - rect[:, 2] // 2

    rect[:, 1] = rect[:, 1] - rect[:, 3] // 2

    rect[:, 2] = rect[:, 0] + rect[:, 2]

    rect[:, 3] = rect[:, 1] + rect[:, 3]



    if rect.shape[0] == 1:

        rect = rect.reshape(-1)



    return rect



def rect_convert_cxcywh_to_x1y1wh(rect_cxcywh):

    rect = np.array(rect_cxcywh)



    if rect.ndim == 1:

        rect = rect.reshape((1, 4))



    rect[:, 0] = rect[:, 0] - rect[:, 2] // 2

    rect[:, 1] = rect[:, 1] - rect[:, 3] // 2



    if rect.shape[0] == 1:

        rect = rect.reshape(-1)



    return rect



def rect_convert_x1y1x2y2_to_cxcywh(rect_x1y1x2y2):

    rect = np.array(rect_x1y1x2y2)



    if rect.ndim == 1:

        rect = rect.reshape((1, 4))



    rect[:, 2] = rect[:, 2] - rect[:, 0]

    rect[:, 3] = rect[:, 3] - rect[:, 1]

    rect[:, 0] = rect[:, 0] + rect[:, 2] // 2

    rect[:, 1] = rect[:, 1] + rect[:, 3] // 2

    

    if rect.shape[0] == 1:

        rect = rect.reshape(-1)



    return rect



def rect_convert_x1y1x2y2_to_x1y1wh(rect_x1y1x2y2):

    rect = np.array(rect_x1y1x2y2)



    if rect.ndim == 1:

        rect = rect.reshape((1, 4))



    rect[:, 2] = rect[:, 2] - rect[:, 0]

    rect[:, 3] = rect[:, 3] - rect[:, 1]



    if rect.shape[0] == 1:

        rect = rect.reshape(-1)



    return rect



def rect_clip_outside(rect_cxcywh, width, height):

    rect_x1y1x2y2 = rect_convert_cxcywh_to_x1y1x2y2(rect_cxcywh)



    if rect_x1y1x2y2.ndim == 1:

        rect_x1y1x2y2 = rect_x1y1x2y2.reshape((1, 4))



    rect_x1y1x2y2[:, 0] = np.maximum(0, rect_x1y1x2y2[:, 0])

    rect_x1y1x2y2[:, 1] = np.maximum(0, rect_x1y1x2y2[:, 1])

    rect_x1y1x2y2[:, 2] = np.maximum(0, np.minimum(width, rect_x1y1x2y2[:, 2]))

    rect_x1y1x2y2[:, 3] = np.maximum(0, np.minimum(height, rect_x1y1x2y2[:, 3]))



    result = rect_convert_x1y1x2y2_to_cxcywh(rect_x1y1x2y2)

    return result



def tool_iou(base_rect_x1y1x2y2, compare_rects_x1y1x2y2):

    base_area = (base_rect_x1y1x2y2[2] - base_rect_x1y1x2y2[0]) * (base_rect_x1y1x2y2[3] - base_rect_x1y1x2y2[1])

    compare_areas = (compare_rects_x1y1x2y2[:, 2] - compare_rects_x1y1x2y2[:, 0]) * (compare_rects_x1y1x2y2[:, 3] - compare_rects_x1y1x2y2[:, 1])



    intersect_x1 = np.maximum(base_rect_x1y1x2y2[0], compare_rects_x1y1x2y2[:, 0])

    intersect_y1 = np.maximum(base_rect_x1y1x2y2[1], compare_rects_x1y1x2y2[:, 1])

    intersect_x2 = np.minimum(base_rect_x1y1x2y2[2], compare_rects_x1y1x2y2[:, 2])

    intersect_y2 = np.minimum(base_rect_x1y1x2y2[3], compare_rects_x1y1x2y2[:, 3])

    intersect_w = np.maximum(intersect_x2 - intersect_x1, 0)

    intersect_h = np.maximum(intersect_y2 - intersect_y1, 0)

    intersect_area = intersect_w * intersect_h

    union_area = base_area + compare_areas - intersect_area



    result = intersect_area / union_area



    return result



def tool_nms(rects_cxcywh, scores, threshold):

    r = []

    s = []



    # sore by score

    indexes = np.argsort(scores)



    complete_bool = np.zeros(indexes.shape, dtype = "bool")

    index = len(indexes) - 1

    while index >= 0:

        # get highest score

        i = indexes[index]



        if not complete_bool[i]:

            rect = rects_cxcywh[i]

            r.append(rect)

            s.append(scores[i])

            complete_bool[i] = True



            iou_result = None

            compare_rects = rects_cxcywh[complete_bool == False]

            if compare_rects.shape[0] > 0:

                rect_x1y1x2y2 = rect_convert_cxcywh_to_x1y1x2y2(rect)

                compare_rects_x1y1x2y2 = rect_convert_cxcywh_to_x1y1x2y2(compare_rects)

                if compare_rects_x1y1x2y2.ndim == 1:

                    compare_rects_x1y1x2y2 = compare_rects_x1y1x2y2.reshape((1, 4))



                iou_result = tool_iou(rect_x1y1x2y2, compare_rects_x1y1x2y2)



                complete_bool[complete_bool == False] = (iou_result > threshold)



        index -= 1



    return (np.array(r), np.array(s))
ANCHOR_SCALES = [ 64 * 64, 128 * 128, 256 * 256 ] # 512 * 512

ANCHOR_ASPECT_RATIO = [ 1.0, 0.5, 2.0 ]

ANCHOR_NUM = len(ANCHOR_SCALES) * len(ANCHOR_ASPECT_RATIO)



def generate_anchors(width, height, stride):

    rows = height // stride

    cols = width // stride



    scales = len(ANCHOR_SCALES)

    ratios = len(ANCHOR_ASPECT_RATIO)

    k = scales * ratios



    anchors = np.zeros((rows, cols, k, 4), dtype = "int")



    for r in range(rows):

        cy = r * stride

        for c in range(cols):

            cx = c * stride

            k_num = 0

            for scale in ANCHOR_SCALES:

                for ratio in ANCHOR_ASPECT_RATIO:

                    w = (int)(np.sqrt(scale * ratio))

                    h = w // ratio

                    anchors[r, c, k_num] = [cx, cy, w, h]

                    k_num += 1



    return anchors



def generate_anchors_train_target(anchors, gt_boxes, img_width, img_height, positive_threshold = 0.7, negative_threshold = 0.3):

    gt_cls = np.ones(anchors.shape[:3]).flatten() * (-1)

    gt_reg = np.zeros((anchors.shape[0] * anchors.shape[1] * anchors.shape[2], 4))



    gt_box_count = len(gt_boxes)

    positive_anchors_gt = np.zeros((anchors.shape[0] * anchors.shape[1] * anchors.shape[2], 4))

    positive_anchors_iou = np.zeros((anchors.shape[:3])).flatten()

    negative_anchors_map = np.zeros((anchors.shape[:3])).flatten()



    anchors_flatten = anchors.reshape((-1, 4))



    # check valid anchors

    anchors_x1y1x2y2_faltten = rect_convert_cxcywh_to_x1y1x2y2(anchors_flatten)

    valid_bool = (((anchors_x1y1x2y2_faltten[:, 0] < 0) \

                 + (anchors_x1y1x2y2_faltten[:, 1] < 0) \

                 + (anchors_x1y1x2y2_faltten[:, 2] > img_height) \

                 + (anchors_x1y1x2y2_faltten[:, 3] > img_width)) == False)

    valid_anchors_count = np.sum(valid_bool)

    positive_valid_anchors_gt = np.zeros((valid_anchors_count, 4))

    positive_valid_anchors_iou = np.zeros((valid_anchors_count,))

    negative_valid_anchors_map = np.zeros((valid_anchors_count,))



    for gt_box in gt_boxes:

        gt_box_x1y1x2y2 = rect_convert_cxcywh_to_x1y1x2y2(gt_box)

        iou = tool_iou(gt_box_x1y1x2y2, anchors_x1y1x2y2_faltten[valid_bool])

        positive_gt_bool = (iou > positive_threshold)

        positive_gt_count = np.sum(positive_gt_bool)

        if positive_gt_count > 0:

            greater_iou_bool = (iou > positive_anchors_iou[valid_bool])

            positive_valid_anchors_iou[greater_iou_bool] = iou[greater_iou_bool]

            positive_valid_anchors_gt[greater_iou_bool] = gt_box

        else:

            iou_max_index = np.argmax(iou)

            # just let the anchor box with max iou assign to this gt box

            # set iou to 100 just to not assign to other gt box

            positive_valid_anchors_iou[iou_max_index] = 100

            positive_valid_anchors_gt[iou_max_index] = gt_box



        negative_gt_bool = (iou < negative_threshold)

        negative_valid_anchors_map[negative_gt_bool] += 1



    positive_anchors_gt[valid_bool] = positive_valid_anchors_gt

    positive_anchors_iou[valid_bool] = positive_valid_anchors_iou

    negative_anchors_map[valid_bool] = negative_valid_anchors_map



    # class: positive

    positive_bool = (positive_anchors_iou > positive_threshold)

    gt_cls[positive_bool] = 1

    # class: negative

    negative_bool = np.logical_and(negative_anchors_map == gt_box_count, np.logical_not(positive_bool))

    gt_cls[negative_bool] = 0



    # regression coefficient

    positive_count = np.sum(positive_bool)

    gt_reg_positive = np.zeros((positive_count, 4))

    gt_reg_positive[:, 0] = (positive_anchors_gt[positive_bool][:, 0] - anchors_flatten[positive_bool][:, 0]) / anchors_flatten[positive_bool][:, 2]

    gt_reg_positive[:, 1] = (positive_anchors_gt[positive_bool][:, 1] - anchors_flatten[positive_bool][:, 1]) / anchors_flatten[positive_bool][:, 3]

    gt_reg_positive[:, 2] = np.log(positive_anchors_gt[positive_bool][:, 2] / anchors_flatten[positive_bool][:, 2])

    gt_reg_positive[:, 3] = np.log(positive_anchors_gt[positive_bool][:, 3] / anchors_flatten[positive_bool][:, 3])

    gt_reg[positive_bool] = gt_reg_positive



    gt_cls = gt_cls.reshape(anchors.shape[:3])

    gt_reg = gt_reg.reshape((anchors.shape[0], anchors.shape[1], -1))



    return (gt_cls, gt_reg)



def generate_anchors_train_batch_target(anchors, gt_boxes, img_width, img_height, positive_threshold = 0.7, negative_threshold = 0.3):

    (gt_cls, gt_reg) = generate_anchors_train_target(anchors, gt_boxes, img_width, img_height, positive_threshold, negative_threshold)



    batch_size = 256

    positive_size_max = 128



    gt_cls_flatten = gt_cls.reshape((-1,))

    gt_reg_flatten = gt_reg.reshape((-1, 4))



    positive_map_bool = (gt_cls_flatten == 1)

    negative_map_bool = (gt_cls_flatten == 0)



    positive_count = np.sum(positive_map_bool)

    negative_count = np.sum(negative_map_bool)



    positive_size = min(positive_size_max, positive_count)

    negative_size = min(batch_size - positive_size, negative_count)

    batch_size = positive_size + negative_size



    gt_cls_positive_samples_index = np.random.choice(positive_count, positive_size, replace = False)

    gt_cls_negative_samples_index = np.random.choice(negative_count, negative_size, replace = False)



    positive_sample_map_bool = positive_map_bool[positive_map_bool]

    positive_sample_map_bool[:] = False

    positive_sample_map_bool[gt_cls_positive_samples_index] = True

    positive_map_bool[positive_map_bool] = positive_sample_map_bool



    negative_sample_map_bool = negative_map_bool[negative_map_bool]

    negative_sample_map_bool[:] = False

    negative_sample_map_bool[gt_cls_negative_samples_index] = True

    negative_map_bool[negative_map_bool] = negative_sample_map_bool

    

    gt_cls_mask = np.zeros(gt_cls.shape, dtype = gt_cls.dtype)

    gt_reg_mask = np.zeros(gt_reg.shape, dtype = gt_reg.dtype)



    gt_cls_mask_flatten = gt_cls_mask.reshape((-1,))

    gt_reg_mask_flatten = gt_reg_mask.reshape((-1, 4))



    gt_cls_mask_flatten[positive_map_bool] = 1

    gt_cls_mask_flatten[negative_map_bool] = 1



    gt_reg_mask_flatten[positive_map_bool] = 1



    gt_cls_final = np.concatenate([gt_cls, gt_cls_mask], axis = -1)

    gt_reg_final = np.concatenate([gt_reg, gt_reg_mask], axis = -1)



    return (gt_cls_final, gt_reg_final)



def get_roi(anchors, anchors_cls, anchors_reg, img_width, img_height, positive_threshold = 0.5, nms_threshold = 0.7, top_n = 2000):

    anchor_rows, anchor_cols, anchor_k, = anchors.shape[:3]



    all_anchors = anchors.reshape((-1, 4))

    all_anchors_cls = anchors_cls.reshape((-1,))

    all_anchors_reg = anchors_reg.reshape((-1, 4))



    positive_condition = (all_anchors_cls > positive_threshold)

    positive_anchors = all_anchors[positive_condition]

    positive_anchors_cls = all_anchors_cls[positive_condition]

    positive_anchors_reg = all_anchors_reg[positive_condition]



    # convert "anchor box" to "refined box" using "regression coefficients"

    proposals = np.zeros((positive_anchors.shape), dtype = "int")

    proposals[:, 0] = positive_anchors[:, 2] * positive_anchors_reg[:, 0] + positive_anchors[:, 0]

    proposals[:, 1] = positive_anchors[:, 3] * positive_anchors_reg[:, 1] + positive_anchors[:, 1]

    proposals[:, 2] = positive_anchors[:, 2] * np.exp(positive_anchors_reg[:, 2])

    proposals[:, 3] = positive_anchors[:, 3] * np.exp(positive_anchors_reg[:, 3])

    

    # clip outside

    proposals = rect_clip_outside(proposals, img_width, img_height)

    if proposals.ndim == 1:

        proposals = proposals.reshape((1, 4))

    

    # remove width <= 0 or height <= 0

    proposals_valid = np.logical_not(np.logical_or(proposals[:, 2] <= 0, proposals[:, 3] <= 0))



    # TODO: pre_top_n



    if proposals[proposals_valid].shape[0] == 0:

        return ([], [])



    r, p = tool_nms(proposals[proposals_valid], positive_anchors_cls[proposals_valid], nms_threshold)



    r = r[:top_n]

    p = p[:top_n]



    return (r, p)
def rpn_loss_cls(y_true, y_pred):

    # y_true: batch, rows, cols, k * 2

    # y_pred: batch, rows, cols, k

    

    spliter = K.shape(y_true)[-1] // 2

    # important!!

    mask = y_true[:, :, :, spliter:]

    y_true_value = y_true[:, :, :, :spliter]



    mask = K.reshape(mask, [-1])

    y_true_value = K.reshape(y_true_value, [-1])

    y_pred_value = K.reshape(y_pred, [-1])

    

    LAMBDA = 1.0

    N_CLS = K.maximum(K.sum(mask), 1e-7) # 256.0

    

    return (LAMBDA * (K.sum(mask * K.binary_crossentropy(y_true_value, y_pred_value)) / N_CLS))



def rpn_loss_reg(y_true, y_pred):

    # y_true: batch, rows, cols, k*4 * 2

    # y_pred: batch, rows, cols, k*4

    

    def smooth_l1(y_true, y_pred):

        diff = y_pred - y_true

        diff_abs = K.abs(diff)

        diff_abs_less_one_bool = K.less(diff_abs, 1)

        diff_abs_less_one_mask = K.cast(diff_abs_less_one_bool, "float32")

        

        return (((diff * diff) * 0.5) * diff_abs_less_one_mask + (diff_abs - 0.5) * (1 - diff_abs_less_one_mask))

    

    spliter = K.shape(y_true)[-1] // 2

    # important!!

    mask = y_true[:, :, :, spliter:]

    y_true_value = y_true[:, :, :, :spliter]



    mask = K.reshape(mask, [-1, 4])

    y_true_value = K.reshape(y_true_value, [-1, 4])

    y_pred_value = K.reshape(y_pred, [-1, 4])

    

    LAMBDA = 1.0 # 10.0

    N_REG = K.maximum(K.sum(mask), 1e-7) # 2400.0

    

    return (LAMBDA * (K.sum(mask * smooth_l1(y_true_value, y_pred_value)) / N_REG))



IMG_ORI_SIZE = 1024

IMG_SIZE_MIN = 600

IMG_SCALE = IMG_ORI_SIZE / IMG_SIZE_MIN

IMG_MEAN_SUBSTRACTION = [103.939, 116.779, 123.68]



def preprocess_image(img):

    img_ori_height, img_ori_width = img.shape[:2]



    img_ori_min = min(img_ori_height, img_ori_width)

    scale = IMG_SIZE_MIN / img_ori_min

    img_new_height = (int)(img_ori_height * scale)

    img_new_width = (int)(img_ori_width * scale)



    scale_img = cv2.resize(img, (img_new_width, img_new_height))

    

    # TODO: max limitation



    # mean substraction

    scale_img = scale_img.astype("float")

    scale_img -= IMG_MEAN_SUBSTRACTION



    return scale_img
BASE_CNN_DOWNSAMPLE_RATIO = 16

BASE_CNN_TRAINABLE_LAYER_START_INDEX = 0 #7



# base cnn

def model_base_cnn():

    vgg16Model = keras.applications.vgg16.VGG16(include_top = False, weights = None, pooling = None)

    vgg16Model.load_weights("/kaggle/input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")

    

    # trainable

    for (i, l) in enumerate(vgg16Model.layers):

        if i < BASE_CNN_TRAINABLE_LAYER_START_INDEX:

            l.trainable = False

        else:

            l.trainable = True

            

    last_conv_layer_output = vgg16Model.get_layer("block5_conv3").output

    

    return (vgg16Model.input, last_conv_layer_output)



# rpn core

def model_rpn_core(x, anchors_num):

    x = keras.layers.Conv2D(512, 3, strides = 1, padding = "same", 

                            activation = "relu", 

                            kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None), 

                            name = "rpn_conv")(x)

    x_cls = keras.layers.Conv2D(anchors_num, 1, strides = 1, 

                                activation = "sigmoid", 

                                kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None), 

                                name = "rpn_conv_cls")(x)

    x_reg = keras.layers.Conv2D(anchors_num * 4, 1, strides = 1, 

                                activation = "linear", 

                                kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None), 

                                name = "rpn_conv_reg")(x)



    return [ x_cls, x_reg ]



# the rpn model

base_cnn_input, base_cnn_output = model_base_cnn()

rpn_output = model_rpn_core(base_cnn_output, ANCHOR_NUM)
rpn_model = keras.models.Model(inputs = base_cnn_input, outputs = rpn_output)

rpn_model.summary()
DATA_TRAIN_IMG_DIR = "/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images"



# stage_2_train_labels.csv

# stage_2_detailed_class_info.csv

DATA_TRAIN_LABEL_PATH = "/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv"



train_labels = pd.read_csv(DATA_TRAIN_LABEL_PATH)

print(train_labels.head())

print(train_labels.shape)



train_labels = train_labels.to_numpy()

print(train_labels.shape)



train_img_files = os.listdir(DATA_TRAIN_IMG_DIR)



train_info_dict = {}



for f in train_img_files:

    file_path = os.path.join(DATA_TRAIN_IMG_DIR, f)

#     dcm_data = pydicom.read_file(file_path)

#     ori_img = dcm_data.pixel_array

    key = os.path.splitext(f)[0]

    train_info_dict[key] = { "file_path": file_path, "img": None, "bbox": [] }



for record in train_labels:

    patientId = record[0]

    target = record[5]

    if target:

        x = int(record[1] / IMG_SCALE)

        y = int(record[2] / IMG_SCALE)

        width = int(record[3] / IMG_SCALE)

        height = int(record[4] / IMG_SCALE)

        cx = x + width // 2

        cy = y + height // 2

        #use cx,cy,w,h

        bbox = (cx, cy, width, height)

        

        train_info_dict[patientId]["bbox"].append(bbox)



# just train the images that have bbox

train_img_info_keys = [ k for (k, v) in train_info_dict.items() if len(v["bbox"]) > 0 ]

train_img_info_keys = np.array(train_img_info_keys)



print("number of all train image: {}".format(len(train_info_dict)))

print("number of train image with bbox: {}".format(len(train_img_info_keys)))
TRAIN_INIT_LEARNING_RATE = 0.001



optimizer = keras.optimizers.SGD(TRAIN_INIT_LEARNING_RATE, 0.9)

rpn_model.compile(loss = [rpn_loss_cls, rpn_loss_reg], optimizer = optimizer)
%%time



train_ts_start = time.time()



TRAIN_TOTAL_EPOCH_NUM = 20



train_img_count = len(train_img_info_keys)

for ep in range(TRAIN_TOTAL_EPOCH_NUM):

    # shuffle the images

    np.random.shuffle(train_img_info_keys)

    

    loss_total = 0

    loss_cls = 0

    loss_reg = 0

    

    for (i, k) in enumerate(train_img_info_keys):

        img_file_path = train_info_dict[k]["file_path"]

        bbox = train_info_dict[k]["bbox"]

        

        dcm_data = pydicom.read_file(file_path)

        ori_img = dcm_data.pixel_array

        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_GRAY2RGB)

        img = preprocess_image(ori_img)

        

        anchors = generate_anchors(img.shape[1], img.shape[0], BASE_CNN_DOWNSAMPLE_RATIO)

        (gt_cls, gt_reg) = generate_anchors_train_batch_target(anchors, bbox, img.shape[1], img.shape[0], 0.5, 0.3)

        

        X = np.expand_dims(img, axis = 0)

        Y = [np.expand_dims(gt_cls, axis = 0), np.expand_dims(gt_reg, axis = 0)]

        

        losses = rpn_model.train_on_batch(X, Y)

        loss_total += losses[0]

        loss_cls += losses[1]

        loss_reg += losses[2]

        

    print("Epoch {}: total loss: {}; cls loss: {}; reg loss: {}".format(ep + 1, loss_total / train_img_count, loss_cls / train_img_count, loss_reg / train_img_count))

        

train_ts_end = time.time()
print(train_ts_end - train_ts_start)
%%time



DATA_TEST_IMG_DIR = "/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_test_images"



test_img_files = os.listdir(DATA_TEST_IMG_DIR)

test_img_files.sort()



test_predict_result = []



for (i, f) in enumerate(test_img_files):

    file_path = os.path.join(DATA_TEST_IMG_DIR, f)

    

    dcm_data = pydicom.read_file(file_path)

    ori_img = dcm_data.pixel_array

    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_GRAY2RGB)

    img = preprocess_image(ori_img)

    

    X = np.expand_dims(img, axis = 0)

    

    Y = rpn_model.predict(X)

    Y_cls = Y[0]

    Y_reg = Y[1]

    

    (regions, probs) = get_roi(anchors, Y_cls, Y_reg, img.shape[1], img.shape[0], positive_threshold = 0.85, nms_threshold = 0.3, top_n = 4)

    

    predict_str = ""

    if len(regions) > 0:

        regions_final = rect_convert_cxcywh_to_x1y1wh(regions)

        if regions_final.ndim == 1:

            regions_final = regions_final.reshape((1, 4))

        regions_final = regions_final * IMG_SCALE

        str_list = []

        for (r, p) in zip(regions_final, probs):

            str_list.append("{:.2f} {:.0f} {:.0f} {:.0f} {:.0f}".format(p, r[0], r[1], r[2], r[3]))

        predict_str = " ".join(str_list)

        

    test_predict_result.append(predict_str)
# patientId,PredictionString

# 0000a175-0e68-4ca4-b1af-167204a7e0bc,0.5 0 0 100 100 0.5 0 0 100 100



# stage_2_sample_submission.csv



test_img_file_names = [ os.path.splitext(f)[0] for f in test_img_files ]



result_data_pd = pd.DataFrame(data = { "patientId": test_img_file_names, "PredictionString": test_predict_result })

print(result_data_pd.head())

result_data_pd.to_csv("submission.csv", index = False)