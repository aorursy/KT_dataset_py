import colorsys

import imghdr

import cv2

import os

import random

from keras import backend as K



import numpy as np

from PIL import Image, ImageDraw, ImageFont





def read_classes(classes_path):

    with open(classes_path) as f:

        class_names = f.readlines()

    class_names = [c.strip() for c in class_names]

    return class_names





def read_anchors(anchors_path):

    with open(anchors_path) as f:

        anchors = f.readline()

        anchors = [float(x) for x in anchors.split(',')]

        anchors = np.array(anchors).reshape(-1, 2)

    return anchors





def generate_colors(class_names):

    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]

    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))

    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(10101)  # Fixed seed for consistent colors across runs.

    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.

    random.seed(None)  # Reset seed to default.

    return colors





def scale_boxes(boxes, image_shape):

    """ Scales the predicted boxes in order to be drawable on the image"""

    height = image_shape[0]

    width = image_shape[1]

    image_dims = K.stack([height, width, height, width])

    image_dims = K.reshape(image_dims, [1, 4])

    boxes = boxes * image_dims

    return boxes





def preprocess_image(imag, model_image_size):

    #image_type = imghdr.what(img_path)

    image = Image.fromarray(imag)

    image = image.convert('RGB')

    #image_data = cv2.resize(image,

    #model_image_size, interpolation = cv2.INTER_AREA)

    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)

    image_data = np.array(resized_image, dtype='float32')

    #image_data = np.asarray(image_data,dtype='float32')

    image_data /= 255.

    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    return image, image_data





def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):

    results = []

    font = ImageFont.truetype(font='../input/fontdata/FiraMono-Medium.otf',

                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

    thickness = (image.size[0] + image.size[1]) // 300



    for i, c in reversed(list(enumerate(out_classes))):

        predicted_class = class_names[c]

        box = out_boxes[i]

        score = out_scores[i]



        label = '{} {:.2f}'.format(predicted_class, score)



        draw = ImageDraw.Draw(image)

        label_size = draw.textsize(label, font)



        top, left, bottom, right = box

        top = max(0, np.floor(top + 0.5).astype('int32'))

        left = max(0, np.floor(left + 0.5).astype('int32'))

        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))

        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

        results.append([label, left, top, right, bottom])

       # print(label, (left, top), (right, bottom))



        if top - label_size[1] >= 0:

            text_origin = np.array([left, top - label_size[1]])

        else:

            text_origin = np.array([left, top + 1])



        # My kingdom for a good redistributable image drawing library.

        for i in range(thickness):

            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])

        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])

        draw.text(text_origin, label, fill=(0, 0, 0), font=font)

        del draw

    return results
from functools import reduce





def compose(*funcs):

    """Compose arbitrarily many functions, evaluated left to right.



    Reference: https://mathieularose.com/function-composition-in-python/

    """

    # return lambda x: reduce(lambda v, f: f(v), funcs, x)

    if funcs:

        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)

    else:

        raise ValueError('Composition of empty sequence not supported.')

import functools

from functools import partial



from keras.layers import Conv2D, MaxPooling2D

from keras.layers.advanced_activations import LeakyReLU

from keras.layers.normalization import BatchNormalization

from keras.models import Model

from keras.regularizers import l2







# Partial wrapper for Convolution2D with static default argument.

_DarknetConv2D = partial(Conv2D, padding='same')





@functools.wraps(Conv2D)

def DarknetConv2D(*args, **kwargs):

    """Wrapper to set Darknet weight regularizer for Convolution2D."""

    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}

    darknet_conv_kwargs.update(kwargs)

    return _DarknetConv2D(*args, **darknet_conv_kwargs)





def DarknetConv2D_BN_Leaky(*args, **kwargs):

    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""

    no_bias_kwargs = {'use_bias': False}

    no_bias_kwargs.update(kwargs)

    return compose(

        DarknetConv2D(*args, **no_bias_kwargs),

        BatchNormalization(),

        LeakyReLU(alpha=0.1))





def bottleneck_block(outer_filters, bottleneck_filters):

    """Bottleneck block of 3x3, 1x1, 3x3 convolutions."""

    return compose(

        DarknetConv2D_BN_Leaky(outer_filters, (3, 3)),

        DarknetConv2D_BN_Leaky(bottleneck_filters, (1, 1)),

        DarknetConv2D_BN_Leaky(outer_filters, (3, 3)))





def bottleneck_x2_block(outer_filters, bottleneck_filters):

    """Bottleneck block of 3x3, 1x1, 3x3, 1x1, 3x3 convolutions."""

    return compose(

        bottleneck_block(outer_filters, bottleneck_filters),

        DarknetConv2D_BN_Leaky(bottleneck_filters, (1, 1)),

        DarknetConv2D_BN_Leaky(outer_filters, (3, 3)))





def darknet_body():

    """Generate first 18 conv layers of Darknet-19."""

    return compose(

        DarknetConv2D_BN_Leaky(32, (3, 3)),

        MaxPooling2D(),

        DarknetConv2D_BN_Leaky(64, (3, 3)),

        MaxPooling2D(),

        bottleneck_block(128, 64),

        MaxPooling2D(),

        bottleneck_block(256, 128),

        MaxPooling2D(),

        bottleneck_x2_block(512, 256),

        MaxPooling2D(),

        bottleneck_x2_block(1024, 512))





def darknet19(inputs):

    """Generate Darknet-19 model for Imagenet classification."""

    body = darknet_body()(inputs)

    logits = DarknetConv2D(1000, (1, 1), activation='softmax')(body)

    return Model(inputs, logits)

import sys



import numpy as np

import tensorflow as tf

from keras import backend as K

from keras.layers import Lambda

from keras.layers.merge import concatenate

from keras.models import Model







sys.path.append('..')



voc_anchors = np.array(

    [[1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]])



voc_classes = [

    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",

    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",

    "pottedplant", "sheep", "sofa", "train", "tvmonitor"

]





def space_to_depth_x2(x):

    """Thin wrapper for Tensorflow space_to_depth with block_size=2."""

    # Import currently required to make Lambda work.

    # See: https://github.com/fchollet/keras/issues/5088#issuecomment-273851273

    import tensorflow as tf

    return tf.space_to_depth(x, block_size=2)





def space_to_depth_x2_output_shape(input_shape):

    """Determine space_to_depth output shape for block_size=2.



    Note: For Lambda with TensorFlow backend, output shape may not be needed.

    """

    return (input_shape[0], input_shape[1] // 2, input_shape[2] // 2, 4 *

            input_shape[3]) if input_shape[1] else (input_shape[0], None, None,

                                                    4 * input_shape[3])





def yolo_body(inputs, num_anchors, num_classes):

    """Create YOLO_V2 model CNN body in Keras."""

    darknet = Model(inputs, darknet_body()(inputs))

    conv20 = compose(

        DarknetConv2D_BN_Leaky(1024, (3, 3)),

        DarknetConv2D_BN_Leaky(1024, (3, 3)))(darknet.output)



    conv13 = darknet.layers[43].output

    conv21 = DarknetConv2D_BN_Leaky(64, (1, 1))(conv13)

    # TODO: Allow Keras Lambda to use func arguments for output_shape?

    conv21_reshaped = Lambda(

        space_to_depth_x2,

        output_shape=space_to_depth_x2_output_shape,

        name='space_to_depth')(conv21)



    x = concatenate([conv21_reshaped, conv20])

    x = DarknetConv2D_BN_Leaky(1024, (3, 3))(x)

    x = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(x)

    return Model(inputs, x)





def yolo_head(feats, anchors, num_classes):

    """Convert final layer features to bounding box parameters.



    Parameters

    ----------

    feats : tensor

        Final convolutional layer features.

    anchors : array-like

        Anchor box widths and heights.

    num_classes : int

        Number of target classes.



    Returns

    -------

    box_xy : tensor

        x, y box predictions adjusted by spatial location in conv layer.

    box_wh : tensor

        w, h box predictions adjusted by anchors and conv spatial resolution.

    box_conf : tensor

        Probability estimate for whether each box contains any object.

    box_class_pred : tensor

        Probability distribution estimate for each box over class labels.

    """

    num_anchors = len(anchors)

    # Reshape to batch, height, width, num_anchors, box_params.

    anchors_tensor = K.reshape(K.variable(anchors), [1, 1, 1, num_anchors, 2])

    # Static implementation for fixed models.

    # TODO: Remove or add option for static implementation.

    # _, conv_height, conv_width, _ = K.int_shape(feats)

    # conv_dims = K.variable([conv_width, conv_height])



    # Dynamic implementation of conv dims for fully convolutional model.

    conv_dims = K.shape(feats)[1:3]  # assuming channels last

    # In YOLO the height index is the inner most iteration.

    conv_height_index = K.arange(0, stop=conv_dims[0])

    conv_width_index = K.arange(0, stop=conv_dims[1])

    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])



    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.

    # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)

    conv_width_index = K.tile(K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])

    conv_width_index = K.flatten(K.transpose(conv_width_index))

    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))

    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])

    conv_index = K.cast(conv_index, K.dtype(feats))

    

    feats = K.reshape(feats, [-1, conv_dims[0], conv_dims[1], num_anchors, num_classes + 5])

    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))



    # Static generation of conv_index:

    # conv_index = np.array([_ for _ in np.ndindex(conv_width, conv_height)])

    # conv_index = conv_index[:, [1, 0]]  # swap columns for YOLO ordering.

    # conv_index = K.variable(

    #     conv_index.reshape(1, conv_height, conv_width, 1, 2))

    # feats = Reshape(

    #     (conv_dims[0], conv_dims[1], num_anchors, num_classes + 5))(feats)



    box_confidence = K.sigmoid(feats[..., 4:5])

    box_xy = K.sigmoid(feats[..., :2])

    box_wh = K.exp(feats[..., 2:4])

    box_class_probs = K.softmax(feats[..., 5:])



    # Adjust preditions to each spatial grid point and anchor size.

    # Note: YOLO iterates over height index before width index.

    box_xy = (box_xy + conv_index) / conv_dims

    box_wh = box_wh * anchors_tensor / conv_dims



    return box_confidence, box_xy, box_wh, box_class_probs





def yolo_boxes_to_corners(box_xy, box_wh):

    """Convert YOLO box predictions to bounding box corners."""

    box_mins = box_xy - (box_wh / 2.)

    box_maxes = box_xy + (box_wh / 2.)



    return K.concatenate([

        box_mins[..., 1:2],  # y_min

        box_mins[..., 0:1],  # x_min

        box_maxes[..., 1:2],  # y_max

        box_maxes[..., 0:1]  # x_max

    ])





def yolo_loss(args,

              anchors,

              num_classes,

              rescore_confidence=False,

              print_loss=False):

    """YOLO localization loss function.



    Parameters

    ----------

    yolo_output : tensor

        Final convolutional layer features.



    true_boxes : tensor

        Ground truth boxes tensor with shape [batch, num_true_boxes, 5]

        containing box x_center, y_center, width, height, and class.



    detectors_mask : array

        0/1 mask for detector positions where there is a matching ground truth.



    matching_true_boxes : array

        Corresponding ground truth boxes for positive detector positions.

        Already adjusted for conv height and width.



    anchors : tensor

        Anchor boxes for model.



    num_classes : int

        Number of object classes.



    rescore_confidence : bool, default=False

        If true then set confidence target to IOU of best predicted box with

        the closest matching ground truth box.



    print_loss : bool, default=False

        If True then use a tf.Print() to print the loss components.



    Returns

    -------

    mean_loss : float

        mean localization loss across minibatch

    """

    (yolo_output, true_boxes, detectors_mask, matching_true_boxes) = args

    num_anchors = len(anchors)

    object_scale = 5

    no_object_scale = 1

    class_scale = 1

    coordinates_scale = 1

    pred_xy, pred_wh, pred_confidence, pred_class_prob = yolo_head(

        yolo_output, anchors, num_classes)



    # Unadjusted box predictions for loss.

    # TODO: Remove extra computation shared with yolo_head.

    yolo_output_shape = K.shape(yolo_output)

    feats = K.reshape(yolo_output, [

        -1, yolo_output_shape[1], yolo_output_shape[2], num_anchors,

        num_classes + 5

    ])

    pred_boxes = K.concatenate(

        (K.sigmoid(feats[..., 0:2]), feats[..., 2:4]), axis=-1)



    # TODO: Adjust predictions by image width/height for non-square images?

    # IOUs may be off due to different aspect ratio.



    # Expand pred x,y,w,h to allow comparison with ground truth.

    # batch, conv_height, conv_width, num_anchors, num_true_boxes, box_params

    pred_xy = K.expand_dims(pred_xy, 4)

    pred_wh = K.expand_dims(pred_wh, 4)



    pred_wh_half = pred_wh / 2.

    pred_mins = pred_xy - pred_wh_half

    pred_maxes = pred_xy + pred_wh_half



    true_boxes_shape = K.shape(true_boxes)



    # batch, conv_height, conv_width, num_anchors, num_true_boxes, box_params

    true_boxes = K.reshape(true_boxes, [

        true_boxes_shape[0], 1, 1, 1, true_boxes_shape[1], true_boxes_shape[2]

    ])

    true_xy = true_boxes[..., 0:2]

    true_wh = true_boxes[..., 2:4]



    # Find IOU of each predicted box with each ground truth box.

    true_wh_half = true_wh / 2.

    true_mins = true_xy - true_wh_half

    true_maxes = true_xy + true_wh_half



    intersect_mins = K.maximum(pred_mins, true_mins)

    intersect_maxes = K.minimum(pred_maxes, true_maxes)

    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)

    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]



    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    true_areas = true_wh[..., 0] * true_wh[..., 1]



    union_areas = pred_areas + true_areas - intersect_areas

    iou_scores = intersect_areas / union_areas



    # Best IOUs for each location.

    best_ious = K.max(iou_scores, axis=4)  # Best IOU scores.

    best_ious = K.expand_dims(best_ious)



    # A detector has found an object if IOU > thresh for some true box.

    object_detections = K.cast(best_ious > 0.6, K.dtype(best_ious))



    # TODO: Darknet region training includes extra coordinate loss for early

    # training steps to encourage predictions to match anchor priors.



    # Determine confidence weights from object and no_object weights.

    # NOTE: YOLO does not use binary cross-entropy here.

    no_object_weights = (no_object_scale * (1 - object_detections) *

                         (1 - detectors_mask))

    no_objects_loss = no_object_weights * K.square(-pred_confidence)



    if rescore_confidence:

        objects_loss = (object_scale * detectors_mask *

                        K.square(best_ious - pred_confidence))

    else:

        objects_loss = (object_scale * detectors_mask *

                        K.square(1 - pred_confidence))

    confidence_loss = objects_loss + no_objects_loss



    # Classification loss for matching detections.

    # NOTE: YOLO does not use categorical cross-entropy loss here.

    matching_classes = K.cast(matching_true_boxes[..., 4], 'int32')

    matching_classes = K.one_hot(matching_classes, num_classes)

    classification_loss = (class_scale * detectors_mask *

                           K.square(matching_classes - pred_class_prob))



    # Coordinate loss for matching detection boxes.

    matching_boxes = matching_true_boxes[..., 0:4]

    coordinates_loss = (coordinates_scale * detectors_mask *

                        K.square(matching_boxes - pred_boxes))



    confidence_loss_sum = K.sum(confidence_loss)

    classification_loss_sum = K.sum(classification_loss)

    coordinates_loss_sum = K.sum(coordinates_loss)

    total_loss = 0.5 * (

        confidence_loss_sum + classification_loss_sum + coordinates_loss_sum)

    if print_loss:

        total_loss = tf.Print(

            total_loss, [

                total_loss, confidence_loss_sum, classification_loss_sum,

                coordinates_loss_sum

            ],

            message='yolo_loss, conf_loss, class_loss, box_coord_loss:')



    return total_loss





def yolo(inputs, anchors, num_classes):

    """Generate a complete YOLO_v2 localization model."""

    num_anchors = len(anchors)

    body = yolo_body(inputs, num_anchors, num_classes)

    outputs = yolo_head(body.output, anchors, num_classes)

    return outputs





def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=.6):

    """Filter YOLO boxes based on object and class confidence."""



    box_scores = box_confidence * box_class_probs

    box_classes = K.argmax(box_scores, axis=-1)

    box_class_scores = K.max(box_scores, axis=-1)

    prediction_mask = box_class_scores >= threshold



    # TODO: Expose tf.boolean_mask to Keras backend?

    boxes = tf.boolean_mask(boxes, prediction_mask)

    scores = tf.boolean_mask(box_class_scores, prediction_mask)

    classes = tf.boolean_mask(box_classes, prediction_mask)



    return boxes, scores, classes





def yolo_eval(yolo_outputs,

              image_shape,

              max_boxes=10,

              score_threshold=.6,

              iou_threshold=.5):

    """Evaluate YOLO model on given input batch and return filtered boxes."""

    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    boxes, scores, classes = yolo_filter_boxes(

        box_confidence, boxes, box_class_probs, threshold=score_threshold)

    

    # Scale boxes back to original image shape.

    height = image_shape[0]

    width = image_shape[1]

    image_dims = K.stack([height, width, height, width])

    image_dims = K.reshape(image_dims, [1, 4])

    boxes = boxes * image_dims



    # TODO: Something must be done about this ugly hack!

    max_boxes_tensor = K.variable(max_boxes, dtype='int32')

    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))

    nms_index = tf.image.non_max_suppression(

        boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)

    boxes = K.gather(boxes, nms_index)

    scores = K.gather(scores, nms_index)

    classes = K.gather(classes, nms_index)

    

    return boxes, scores, classes





def preprocess_true_boxes(true_boxes, anchors, image_size):

    """Find detector in YOLO where ground truth box should appear.



    Parameters

    ----------

    true_boxes : array

        List of ground truth boxes in form of relative x, y, w, h, class.

        Relative coordinates are in the range [0, 1] indicating a percentage

        of the original image dimensions.

    anchors : array

        List of anchors in form of w, h.

        Anchors are assumed to be in the range [0, conv_size] where conv_size

        is the spatial dimension of the final convolutional features.

    image_size : array-like

        List of image dimensions in form of h, w in pixels.



    Returns

    -------

    detectors_mask : array

        0/1 mask for detectors in [conv_height, conv_width, num_anchors, 1]

        that should be compared with a matching ground truth box.

    matching_true_boxes: array

        Same shape as detectors_mask with the corresponding ground truth box

        adjusted for comparison with predicted parameters at training time.

    """

    height, width = image_size

    num_anchors = len(anchors)

    # Downsampling factor of 5x 2-stride max_pools == 32.

    # TODO: Remove hardcoding of downscaling calculations.

    assert height % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'

    assert width % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'

    conv_height = height // 32

    conv_width = width // 32

    num_box_params = true_boxes.shape[1]

    detectors_mask = np.zeros(

        (conv_height, conv_width, num_anchors, 1), dtype=np.float32)

    matching_true_boxes = np.zeros(

        (conv_height, conv_width, num_anchors, num_box_params),

        dtype=np.float32)



    for box in true_boxes:

        # scale box to convolutional feature spatial dimensions

        box_class = box[4:5]

        box = box[0:4] * np.array(

            [conv_width, conv_height, conv_width, conv_height])

        i = np.floor(box[1]).astype('int')

        j = min(np.floor(box[0]).astype('int'),1)

        best_iou = 0

        best_anchor = 0

                

        for k, anchor in enumerate(anchors):

            # Find IOU between box shifted to origin and anchor box.

            box_maxes = box[2:4] / 2.

            box_mins = -box_maxes

            anchor_maxes = (anchor / 2.)

            anchor_mins = -anchor_maxes



            intersect_mins = np.maximum(box_mins, anchor_mins)

            intersect_maxes = np.minimum(box_maxes, anchor_maxes)

            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)

            intersect_area = intersect_wh[0] * intersect_wh[1]

            box_area = box[2] * box[3]

            anchor_area = anchor[0] * anchor[1]

            iou = intersect_area / (box_area + anchor_area - intersect_area)

            if iou > best_iou:

                best_iou = iou

                best_anchor = k

                

        if best_iou > 0:

            detectors_mask[i, j, best_anchor] = 1

            adjusted_box = np.array(

                [

                    box[0] - j, box[1] - i,

                    np.log(box[2] / anchors[best_anchor][0]),

                    np.log(box[3] / anchors[best_anchor][1]), box_class

                ],

                dtype=np.float32)

            matching_true_boxes[i, j, best_anchor] = adjusted_box

    return detectors_mask, matching_true_boxes

import os

import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow

import scipy.io

import scipy.misc

import numpy as np

import pandas as pd

import PIL

import tensorflow as tf

from skimage.transform import resize

from keras import backend as K

from keras.layers import Input, Lambda, Conv2D

from keras.models import load_model, Model





%matplotlib inline
def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):

    box_scores = box_confidence*box_class_probs

    box_classes = K.argmax(box_scores,-1)

    box_class_scores = K.max(box_scores,-1)

    filtering_mask = box_class_scores>threshold

    scores = tf.boolean_mask(box_class_scores,filtering_mask)

    boxes = tf.boolean_mask(boxes,filtering_mask)

    classes = tf.boolean_mask(box_classes,filtering_mask)

 

    return scores, boxes, classes
def iou(box1, box2):

    xi1 = max(box1[0],box2[0])

    yi1 = max(box1[1],box2[1])

    xi2 = min(box1[2],box2[2])

    yi2 = min(box1[3],box2[3])

    inter_area = (yi2-yi1)*(xi2-xi1)

    box1_area = (box1[3]-box1[1])*(box1[2]-box1[0])

    box2_area = (box2[3]-box2[1])*(box2[2]-box2[0])

    union_area = box1_area+box2_area-inter_area

    iou = inter_area/union_area

 

    return iou
def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):

    max_boxes_tensor = K.variable(max_boxes, dtype='int32')

    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))

    nms_indices = tf.image.non_max_suppression(boxes,scores,max_boxes,iou_threshold)

    scores = K.gather(scores,nms_indices)

    boxes = K.gather(boxes,nms_indices)

    classes = K.gather(classes,nms_indices)



    return scores, boxes, classes
yolo_outputs = (tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1),

                   tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),

                   tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),

                   tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1))
def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):

    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = score_threshold)

    boxes = scale_boxes(boxes, image_shape)

    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)



    return scores, boxes, classes
scores, boxes, classes = yolo_eval(yolo_outputs)
sess = K.get_session()

class_names = read_classes("../input/yoloh5file/coco_classes.txt")

anchors = read_anchors("../input/yoloh5file/yolo_anchors.txt")



yolo_model = load_model("../input/yoloh5file/yolo.h5")
yolo_model.output
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
def predict(sess, image_file):

    image, image_data = preprocess_image( image_file, model_image_size = (608, 608))

    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})

    #print('Found {} boxes for {}'.format(len(out_boxes), image_file))

    # Generate colors for drawing bounding boxes.

    colors = generate_colors(class_names)

    # Draw bounding boxes on the image file

    results = draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)

    # Save the predicted bounding box on the image

    #image.save(os.path.join("out", image_file), quality=90)

    # Display the results in the notebook

    #output_image = scipy.misc.imread(os.path.join("out", image_file))

    return out_scores, out_boxes, out_classes,image,results
def campare2coordinate2(cor1,cor2,thresh):

    distance = math.sqrt(sum([(a - b)**2 for a, b in zip(cor1, cor2)]))

    

    if(distance<=thresh):

        

        return True

    else:

        return False

def distance(cor1,cor2):

    return math.sqrt(sum([(a - b)**2 for a, b in zip(cor1, cor2)]))

def findclosestcar(i,image2,thresh=132):

    temp = {}

    for j in image2:

        if(campare2coordinate2(i,j,thresh)==True):

            temp.update({distance(i,j):j})

            #temp1.append(distance(i,j))

    if(len(temp)!=0):

        minkey = sorted(temp)[0]

        return temp[minkey]

    else:

        cor = []

        return cor

   

  
def findvalue(i,dic):

    for a,b in dic.items():

        if (i == a):

            return True

        

    return False

            

    
globalcar = []

def add2dic(te,list2,list1):

    t= ''

    for i in list1:

        t = t + ' ' + str(i)



    te[t] = list2

    return t

def campare2coordinate2(cor1,cor2,thresh):

    distance = math.sqrt(sum([(a - b)**2 for a, b in zip(cor1, cor2)]))

    

    if(distance<=thresh):

        

        return True

    else:

        return False

def distance(cor1,cor2):

    return math.sqrt(sum([(a - b)**2 for a, b in zip(cor1, cor2)]))

def findclosestcar(i,image2,thresh=132):

    temp = {}

    for j in image2:

        if(campare2coordinate2(i,j,thresh)==True):

            temp.update({distance(i,j):j})

            #temp1.append(distance(i,j))

    if(len(temp)!=0):

        minkey = sorted(temp)[0]

        return temp[minkey]

    else:

        cor = []

        return cor

def campare2coordinate(cor1,cor2):

    return (abs(cor1[0]-cor2[0])+abs(cor1[1]-cor2[1])),cor2

import math

'''def localcars2(image1,image2,thresh):

    dic = {}

    for i in range(0,len(image1)):

        for j in range(0,len(image2)):

            if(campare2coordinate2(image1[i],image2[j],thresh) is True):

                add2dic(dic,image1[i],image2[j])

                #del image1[i]

                #del image2[j]

    return dic'''

def localcars2(image1,image2,thresh):

    print('localcars2 func')

    print(image1,image2)

    dic = {}

    memory_obj = image2

    if(len(image2)!=0):

        for i in image1:

            cor = findclosestcar(i,memory_obj)

            if(len(cor)!=0):

                add2dic(dic,i,cor)

                memory_obj.remove(cor)

    return dic

                

                

                        

                       

                       

            

def add2dic(te,list2,list1):

    t= ''

    for i in list1:

        t = t + ' ' + str(i)



    te[t] = list2

    return t

def findhighfromdic(dic):

    ln = list(dic.values())

    num = []

    for i in ln:

        num.append(int(i[-1]))

    return max(num)

def drawuniquecars(image1,image2,first,globalcars,traffic_count):

    if first==True:

        traffic_count = 1

        globalcars = {}

   

        count = 1

        for i in image1:

            tst = ''

            for term in i:

                tst = tst + ' ' + str(term)

            globalcars[tst] = 'car'+str(count)

            count += 1

   

    dic = localcars2(image1.copy(),image2.copy(),132)

    print(dic)

    countcars = findhighfromdic(globalcars)

    



    for i in image2:

        temp = ''

        for term in i:

            temp = temp + ' ' + str(term)

        if(findvalue(temp,dic)==True):

            array = dic.get(temp)

            temp2 = ''

            for term in array:

                temp2 = temp2 + ' ' + str(term)

            

            car = globalcars.get(temp2)

           

            globalcars[temp] = car

        elif(findvalue(temp,dic)==False):

            countcars = countcars+1

            globalcars[temp] = 'car' + str(countcars)

        '''elif(bool(dic)==False): 

            countcars = countcars+1

            globalcars[temp] = 'car' + str(countcars)'''

    return globalcars,traffic_count



        

        

            

    

    

                
def midpoint(args):

    x,y,a,b =args

    return int(round((x+a)/2)),int(round((y+b)/2))
def conv(tup):

    return (int(tup[0]),int(tup[1]))
def conv2coor(s):

    coor = s.split(' ')

    return int(coor[1][:]),int(coor[2][:])





    

    

    
def globalarraymodify(gc,newcor1):

    length = len(newcor1)

    temp = gc.copy()

    for i in range(0,length):

       del gc[list(temp.keys())[i]]

    return gc



  

g = {' 242.0 530.0': 'car1', ' 232.0 532.0': 'car1', ' 676.0 535.0': 'car2'}

globalarraymodify(g,[1,2])
results1 = [['car 0.69',1,2],['traffic 0.3',19,22]]

carresult = []

for i in results1:

        if 'car' in i[0]:

            carresult.append(i)

carresult
def fill_list(dic):

    ln = list(dic.keys())

    newln = []

    for i in ln:

        newln.append(conv2coor(i))

    return newln
def conv2dicformat(coor):

    tst = ''

    for term in coor:

        tst = tst + ' ' + str(term)

    return tst
def result(img,img2,globalcars,traffic_count,resultlist,first=False):   

    #img = plt.imread('../input/testingtracking/fc2_save_2018-09-05-163418-0423.bmp')

    #img2 = plt.imread('../input/testingtracking1/fc2_save_2018-09-05-163418-0422.bmp')

    image_shape = float(img.shape[0]), float(img.shape[1])

    scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

    out_scores, out_boxes, out_classes,image1,results1 = predict(sess, img)

    out_scores, out_boxes, out_classes,image2,results2 = predict(sess, img2)

    

    img1 = np.asarray(image1) 

    img2 = np.asarray(image2)

        

    cor1 = []

    

    cor2 = []

    newcor1 = []

    newcor2 = []

    carresult1 = []

    carresult2 = []

    traffic1 = []

    traffic2 = []

    

    for i in results1:

        if 'car' in i[0]:

            carresult1.append(i)

        elif 'traffic' in i[0]:

            traffic1.append(i)

            

    for i in results2:

        if 'car' in i[0]:

            carresult2.append(i)

        elif 'traffic' in i[0]:

            traffic2.append(i)

    if (len(traffic1) >=1 or len(traffic2) >=1 ):

        if(len(traffic1)>=1 and len(traffic2)==0 and first==True):

            traffic_count += 1

        if(len(traffic1)==0 and len(traffic2)>=1):

            traffic_count += 1

        if(first==True and len(traffic1)>=1):

            resultlist.append(['trafficlight'+str(traffic_count),conv(traffic1[0][1:])])

            cv2.putText(img1,'trafficlight'+str(traffic_count),conv(traffic1[0][1:]) , cv2.FONT_HERSHEY_SIMPLEX,  1,  255, 2, cv2.LINE_AA)

            resultlist.append(['trafficlight'+str(traffic_count),conv(traffic1[0][1:])])

        if(first==True and len(traffic2)>=1):

            cv2.putText(img1,'trafficlight'+str(traffic_count),conv(traffic2[0][1:]) , cv2.FONT_HERSHEY_SIMPLEX,  1,  255, 2, cv2.LINE_AA)

            resultlist.append(['trafficlight'+str(traffic_count),conv(traffic2[0][1:])])

            

        if(len(traffic2)>=1 and first==False):

            cv2.putText(img2,'trafficlight'+str(traffic_count),conv(traffic2[0][1:]) , cv2.FONT_HERSHEY_SIMPLEX,  1,  255, 2, cv2.LINE_AA)

            resultlist.append(['trafficlight'+str(traffic_count),conv(traffic2[0][1:])])

    if(len(carresult1)>0 or len(carresult2)>0):

        #return img,img2,globalcars,resultlist

        

        for i in carresult1:

            cor1.append(i[1:])

            #cor2.append(j[1:])

        for j in carresult2:

            #cor1.append(i[1:])

            cor2.append(j[1:])



        for i in cor1:

            newcor1.append(midpoint(i))

            #newcor2.append(midpoint(j))

        for j in cor2:



        #newcor1.append(midpoint(i))

            newcor2.append(midpoint(j))

        if(len(carresult1)==0):

            newcor1 = fill_list(globalcars)

        print(newcor1,newcor2)

        globalcars,dic = drawuniquecars(newcor1,newcor2,first,globalcars,traffic_count)

        cars_in_1 = len(results1)







        listofimage1values = []

        listofimage2values = []

        



        #print(cor1)

        #print(cor1)

        '''if(len(newcor1)<len(newcor2)):

            count = len(newcor1) 

            for i in range(0,len(newcor2)):

                tst = ''

                for term in newcor2[i]:

                    tst = tst + ' ' + str(term)

                count  += 1 

                globalcars[tst] = 'car' + str(count)'''



        '''for i in range(0,cars_in_1):

            #listofimage1values.append(list(globalcars.values())[i])

            a,b = conv2coor(list(globalcars.keys())[i])

            cv2.putText(img1,list(globalcars.values())[i],(a,b) , cv2.FONT_HERSHEY_SIMPLEX,  2,  255, 3, cv2.LINE_AA)

        for i in range(cars_in_1,len(globalcars)):

            #listofimage2values.append(list(globalcars.values())[i])

            a,b = conv2coor(list(globalcars.keys())[i])

            cv2.putText(img2,list(globalcars.values())[i],(a,b) , cv2.FONT_HERSHEY_SIMPLEX,  2,  255, 3, cv2.LINE_AA)

        #print(listofimage1values)

        #print(globalcars)'''





        if(len(carresult1)>0 and first==True):

            temp = []



            for i in newcor1:

                cv2.putText(img1,globalcars.get(conv2dicformat(i)),conv(i) , cv2.FONT_HERSHEY_SIMPLEX,  2,  255, 3, cv2.LINE_AA)

                temp.append(globalcars.get(conv2dicformat(i)))

            resultlist.append([temp,carresult1])

        if(len(carresult2)>0):

            temp = []

            for i in newcor2:

                temp.append(globalcars.get(conv2dicformat(i)))

                cv2.putText(img2,globalcars.get(conv2dicformat(i)),conv(i) , cv2.FONT_HERSHEY_SIMPLEX,  2,  255, 3, cv2.LINE_AA)

            resultlist.append([temp, carresult2])













        '''for i in range(0,len(newcor1)):

            cv2.putText(img1,listofimage1values[i],conv(newcor1[i]) , cv2.FONT_HERSHEY_SIMPLEX,  2,  255, 3, cv2.LINE_AA)

        for i in range(0,len(newcor2)):

            cv2.putText(img2,listofimage2values[i],conv(newcor2[i]) ,  cv2.FONT_HERSHEY_SIMPLEX,  2,  255, 3, cv2.LINE_AA)'''

        '''if(len(results2)>0):

            newgc = globalarraymodify(globalcars,newcor1)

        else:

            newgc = globalcars'''



   

    return img1,img2,globalcars,resultlist,traffic_count
listn = []

for i,j,k in os.walk('../input/testing100images2/test/test/'):

    for filename in k:

        print(i)

        listn.append(plt.imread('../input/testing100images2/test/test/'+filename))
len(listn)
img = listn[0]

image_shape = float(img.shape[0]), float(img.shape[1])

scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

out_scores, out_boxes, out_classes,image,results = predict(sess, img)

#listn = [img,traffic,img,traffic,img,traffic,traffic,traffic]
results
import requests

f = open('0000001.jpg','wb')

f.write(requests.get('https://storage.googleapis.com/proudcity/sanrafaelca/uploads/2018/05/traffic-signal.jpg').content)

f.close()
listn = listn[:6]
len(listn)
count= 0

gc = {}

fr = []

tc = 0

resultlist = []

for i in range(0,len(listn)-1):

    if i==0:

        globalcars = {}

        x,y,gc,resultlist,tc = result(listn[i],listn[i+1],globalcars,tc,resultlist,first=True)

        fr.append(x)

        fr.append(y)

    else:

        x,y,gc,resultlist,tc = result(listn[i],listn[i+1],gc,tc,resultlist,first=False)

        fr.append(y)

        

        

        

        
resultlist
def to_csv(listr):

    label = []

    coor = []

    for i in listr:

        label.append(i[0])

        tcoor = []

        for j in i[1]:

            mid = int(abs(j[1]-j[3])/2)

            tcoor.append([mid,j[2]])

        coor.append(tcoor)

    data = {'label':label,'coordinates':coor}

    df = pd.DataFrame(data=data,columns=['label','coordinates'])

    

    return df.to_csv('./data.csv',index=False)

            
print(to_csv(resultlist))
coor = pd.read_csv('./data.csv')

coor
len(listn)
for i in listn:

    plt.imshow(i)

    plt.show()

    
for i in fr:

    plt.imshow(i)

    plt.show()
img = plt.imread('../input/finaldataset/create/create/image (1).bmp')

image_shape = float(img.shape[0]), float(img.shape[1])

scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

listn = []

result = []

filen = []

for i,j,k in os.walk('../input/finaldataset/create/create/'):

    for filename in k:

        img = plt.imread('../input/finaldataset/create/create/'+filename)

        out_scores, out_boxes, out_classes,image,results = predict(sess, img)

        temp1 = ""

        for i in results:

            temp = ""

            for j in i:

                temp = temp + str(j) + " "

            temp = temp + '\n'

            temp1 += temp

        

        '''print(filename[:-3] + 'txt')   

        f = open('./'+filename[:-3] + 'txt','w+')

        f.write(temp1)'''

        filen.append(filename)

        result.append(temp1)
filen
with open('download.txt', 'w') as f:

    for item in result:

        f.write("%s\n" % item)

f.close()

with open('name.txt', 'w') as a:

    for item in filen:

        a.write("%s\n" % item)

a.close()
predicted = result.copy()
import random

truth = [random.randint(1,100) for i in range(25) ]