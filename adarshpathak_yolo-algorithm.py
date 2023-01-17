import numpy as np

import random

import scipy.misc

import pandas as pd



from functools import reduce

import functools

from functools import partial



import colorsys

import imghdr

import os

import sys

import argparse



import tensorflow as tf

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()



from tensorflow.python.keras import backend as K

#from keras import backend as K

from keras.layers import Lambda

from keras.layers.merge import concatenate

from keras.models import Model

from keras.layers import Conv2D, MaxPooling2D

from keras.layers.advanced_activations import LeakyReLU

from keras.layers.normalization import BatchNormalization

from keras.models import Model

from keras.regularizers import l2

from keras.layers import Input

from tensorflow.keras.models import load_model, Model



import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow

import scipy.io

import PIL

from PIL import Image
DIR_path = '/kaggle/input/imagesforkernel/'
def compose(*funcs):

    if funcs:

        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)

    else:

        raise ValueError('Composition of empty sequence not supported.')
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
voc_anchors = np.array(

    [[1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]])



voc_classes = [

    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",

    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",

    "pottedplant", "sheep", "sofa", "train", "tvmonitor"

]





def space_to_depth_x2(x):

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

    num_anchors = len(anchors)

    # Reshape to batch, height, width, num_anchors, box_params.

    anchors_tensor = K.reshape(K.variable(anchors), [1, 1, 1, num_anchors, 2])



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

          anchors,

          num_classes,

          image_shape,

          max_boxes=20,

          score_threshold=.6,

          iou_threshold=.5):

    num_layers = len(yolo_outputs)

    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] # default setting

    input_shape = K.shape(yolo_outputs[0])[1:3] * 32

    boxes = []

    box_scores = []

    for l in range(num_layers):

        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],

                anchors[anchor_mask[l]], num_classes, input_shape, image_shape)

        boxes.append(_boxes)

        box_scores.append(_box_scores)

    boxes = K.concatenate(boxes, axis=0)

    box_scores = K.concatenate(box_scores, axis=0)



    mask = box_scores >= score_threshold

    max_boxes_tensor = K.constant(max_boxes, dtype='int32')

    boxes_ = []

    scores_ = []

    classes_ = []

    for c in range(num_classes):

        class_boxes = tf.boolean_mask(boxes, mask[:, c])

        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

        nms_index = tf.image.non_max_suppression(

            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)

        class_boxes = K.gather(class_boxes, nms_index)

        class_box_scores = K.gather(class_box_scores, nms_index)

        classes = K.ones_like(class_box_scores, 'int32') * c

        boxes_.append(class_boxes)

        scores_.append(class_box_scores)

        classes_.append(classes)

    boxes_ = K.concatenate(boxes_, axis=0)

    scores_ = K.concatenate(scores_, axis=0)

    classes_ = K.concatenate(classes_, axis=0)



    return boxes_, scores_, classes_





def preprocess_true_boxes(true_boxes, anchors, image_size):

    height, width = image_size

    num_anchors = len(anchors)

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



def preprocess_image(img_path, model_image_size):

    image_type = imghdr.what(img_path)

    image = Image.open(img_path)

    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)

    image_data = np.array(resized_image, dtype='float32')

    image_data /= 255.

    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    return image, image_data



def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):

    

    font = ImageFont.truetype(font='/kaggle/input/imagesforkernel/FiraMono-Medium.otf',size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

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

        print(label, (left, top), (right, bottom))



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
def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):

    box_scores = box_confidence * box_class_probs

    box_classes = K.argmax(box_scores, axis=-1)

    box_class_scores = K.max(box_scores, axis=-1)



    filtering_mask = box_class_scores >= threshold 

    scores = tf.boolean_mask(box_class_scores, filtering_mask)

    print (boxes, filtering_mask)

    boxes = tf.boolean_mask(boxes, filtering_mask)

    classes = tf.boolean_mask(box_classes, filtering_mask)

    return scores, boxes, classes



def iou(box1, box2):

    xi1 = max(box1[0], box2[0])

    yi1 = max(box1[1], box2[1])

    xi2 = min(box1[2], box2[2])

    yi2 = min(box1[3], box2[3])

    inter_area = max(yi2-yi1, 0) * max(xi2-xi1, 0)   



    box1_area = (box1[3] - box1[1]) *  (box1[2] - box1[0])

    box2_area = (box2[3] - box2[1]) *  (box2[2] - box2[0])

    union_area = box1_area + box2_area - inter_area

    

    iou = inter_area / union_area

    return iou



def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):

    max_boxes_tensor = K.variable(max_boxes, dtype='int32')

    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))

    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)



    scores = K.gather(scores, nms_indices)

    boxes = K.gather(boxes, nms_indices)

    classes = K.gather(classes, nms_indices)

    return scores, boxes, classes



def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):

    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)

    boxes = scale_boxes(boxes, image_shape)

    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

    return scores, boxes, classes
sess = K.get_session()
class_names = read_classes("/kaggle/input/imagesforkernel/coco_classes.txt")

anchors = read_anchors("/kaggle/input/imagesforkernel/yolo_anchors.txt")

image_shape = (720., 1280.)
!wget https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5
yolo_model = load_model('yolo.h5')
yolo_model.summary()
yolo_outputs = yolo_head(yolo_model.output[0], anchors, len(class_names))
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

def predict(sess, image_file):

    image, image_data = preprocess_image("/kaggle/input/imagesforkernel/" + image_file, model_image_size = (608, 608))

    feed_dict={ yolo_model.input: image_data, K.learning_phase(): 0 }

    out_scores, out_boxes, out_classes = sess.run(fetches=[scores, boxes, classes], feed_dict=feed_dict)

    print('Found {} boxes for {}'.format(len(out_boxes), image_file))

    colors = generate_colors(class_names)

    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)

    image.save(os.path.join("out", image_file), quality=90)

    output_image = scipy.misc.imread(os.path.join("out", image_file))

    imshow(output_image)

    

    return out_scores, out_boxes, out_classes
#out_scores, out_boxes, out_classes = predict(sess, "test.jpg")