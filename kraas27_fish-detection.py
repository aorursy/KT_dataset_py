%matplotlib inline

import cv2

import random

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import tensorflow as tf

from tensorflow import keras

import os
# types = []

# for i in os.listdir("../input/the-nature-conservancy-fisheries-monitoring/train")[:-1]:

#     types.append(i.lower())

# types.remove('other')

types = ['alb', 'bet', 'dol', 'lag', 'nof', 'shark', 'yft']
from sklearn.feature_extraction.text import CountVectorizer

count_v = CountVectorizer()
Types = count_v.fit_transform(types)
type_ = ['bet']

count_v.transform(type_).toarray()
import json

from glob import glob



# TODO: скачайте данные и сохраните в директорию:

TRAIN_PREFIX = '../input/the-nature-conservancy-fisheries-monitoring/train'



def load_boxes():

    boxes = dict()

    for path in glob('../input/fish-json/*.json'):

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
boxes = load_boxes()  # разметка детекций
boxes['bet'][4]
pd.DataFrame([(k, len(v)) for k, v in boxes.items()],

             columns=['class', 'count'])
plt.figure(figsize=(6, 6), dpi=120)

img = draw_boxes(boxes['bet'][70])

plt.imshow(img)

plt.title('{}x{}'.format(*img.shape));
annotations = sum([box['annotations']

                  for box in sum(boxes.values(), [])], [])



widths = [rect['width'] for rect in annotations]

heights = [rect['height'] for rect in annotations]



plt.hist(widths)

plt.hist(heights);
IMG_HEIGHT = 750

IMG_WIDTH = 1200



features = keras.applications.vgg16.VGG16(include_top=False,

                                          weights='imagenet',

                                          input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))



feature_tensor = features.layers[-1].output



# дообучаем последние 5 слоев

for layer in features.layers[:-5]:

    layer.trainable = False
FEATURE_SHAPE = (feature_tensor.shape[1].value,

                 feature_tensor.shape[2].value)



GRID_STEP_H = IMG_HEIGHT / FEATURE_SHAPE[0]

GRID_STEP_W = IMG_WIDTH / FEATURE_SHAPE[1]



ANCHOR_WIDTH = 150.

ANCHOR_HEIGHT = 150. 



ANCHOR_CENTERS = np.mgrid[GRID_STEP_H/2:IMG_HEIGHT:GRID_STEP_H,

                          GRID_STEP_W/2:IMG_WIDTH:GRID_STEP_W]
feature_tensor.shape
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



def encode_anchors(annotation, img_shape, iou_thr=0.5):

    encoded = np.zeros(shape=(FEATURE_SHAPE[0],

                              FEATURE_SHAPE[1], 5), dtype=np.float32)

    x_scale = 1. * IMG_WIDTH / img_shape[1]

    y_scale = 1. * IMG_HEIGHT / img_shape[0]

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

            encoded[row, col] = [1., dx, dy, dw, dh]

            

    return encoded



def _sigmoid(x):

    return 1. / (1. + np.exp(-x))



def decode_prediction(prediction, conf_thr=0.1):

    rectangles = []

    for row in range(FEATURE_SHAPE[0]):

        for col in range(FEATURE_SHAPE[1]):

            logit, dx, dy, dw, dh = prediction[row, col]

            conf = _sigmoid(logit)

            if conf > conf_thr:

                anchor_x = ANCHOR_CENTERS[1, row, col]

                anchor_y = ANCHOR_CENTERS[0, row, col]

                rectangles.append({'x': anchor_x - dx * ANCHOR_WIDTH,

                                   'y': anchor_y - dy * ANCHOR_HEIGHT,

                                   'width': ANCHOR_WIDTH - dw * ANCHOR_WIDTH,

                                   'height': ANCHOR_HEIGHT - dh * ANCHOR_HEIGHT,

                                   'conf': conf})

    return rectangles
example = boxes['alb'][175]



encoded = encode_anchors(example, (IMG_HEIGHT, IMG_WIDTH))



decoded = decode_prediction(encoded, conf_thr=0.5)

decoded = sorted(decoded, key = lambda e: -e['conf'])



plt.figure(figsize=(6, 6), dpi=120)

plt.imshow(draw_boxes(example, decoded[:10]))
K = tf.keras.backend
def confidence_loss(y_true, y_pred):

    conf_loss = K.binary_crossentropy(y_true[..., 0], 

                                      y_pred[..., 0],

                                      from_logits=True)

    return conf_loss



def smooth_l1(y_true, y_pred):

    abs_loss = K.abs(y_true[..., 1:] - y_pred[..., 1:])

    square_loss = 0.5 * K.square(y_true[..., 1:] - y_pred[..., 1:])

    mask = K.cast(K.greater(abs_loss, 1.), 'float32')

    total_loss = (abs_loss - 0.5) * mask + 0.5 * square_loss * (1. - mask)

    return K.sum(total_loss, axis=-1)



def total_loss(y_true_1, y_pred_1, neg_pos_ratio=3):

    batch_size = K.shape(y_true_1)[0]

    

    # TODO: добавьте функцию потерь для классификации детекции

#     class_loss = K.categorical_crossentropy(y_true_2, y_pred_2)

    

    y_true_1 = K.reshape(y_true_1, (batch_size, -1, 5))

    y_pred_1 = K.reshape(y_pred_1, (batch_size, -1, 5))



    # confidence loss

    conf_loss = confidence_loss(y_true_1, y_pred_1)

    

    # smooth l1 loss

    loc_loss = smooth_l1(y_true_1, y_pred_1)

    

    # positive examples loss

    pos_conf_loss = K.sum(conf_loss * y_true_1[..., 0], axis=-1)

    pos_loc_loss = K.sum(loc_loss * y_true_1[..., 0], axis=-1)

    

    # negative examples loss

    anchors = K.shape(y_true_1)[1]

    num_pos = K.sum(y_true_1[..., 0], axis=-1)

    num_pos_avg = K.mean(num_pos)

    num_neg = K.min([neg_pos_ratio * (num_pos_avg) + 1., K.cast(anchors, 'float32')])

    

    # hard negative mining

    neg_conf_loss, _ = tf.nn.top_k(conf_loss * (1. - y_true_1[..., 0]),

                                   k=K.cast(num_neg, 'int32'))



    neg_conf_loss = K.sum(neg_conf_loss, axis=-1)

    

    # total conf loss

    total_conf_loss = (neg_conf_loss + pos_conf_loss) / (num_neg + num_pos + 1e-32)

    loc_loss = pos_loc_loss / (num_pos + 1e-32)

    

    return total_conf_loss + 0.5 * loc_loss
def load_img(path, target_size=(IMG_WIDTH, IMG_HEIGHT)):

    img = cv2.imread(path, cv2.IMREAD_COLOR)[...,::-1]

    img_shape = img.shape

    img_resized = cv2.resize(img, target_size)

    return img_shape, keras.applications.vgg16.preprocess_input(img_resized.astype(np.float32))



def data_generator(boxes, batch_size=32):

    boxes_ = sum(boxes.values(), [])

    while True:

        random.shuffle(boxes_)

        for i in range(len(boxes_)//batch_size):

            X, y_1, y_2 = [], [], []

            for j in range(i*batch_size,(i+1)*batch_size):

                img_shape, img = load_img(boxes_[j]['filename'])

                # TODO: добавьте one-hot encoding в разметку для классов

                name = [boxes_[j]['filename'].split('/')[-2].lower()]

                y_2.append(count_v.transform(name).toarray().reshape(7))

                y_1.append(encode_anchors(boxes_[j], img_shape))

                X.append(img)

            yield (np.array(X), {'box_output': np.array(y_1), 'category_output': np.array(y_2)})
from tensorflow.keras.layers import Layer



class ROIPoolingLayer(Layer):

    """ Implements Region Of Interest Max Pooling 

        for channel-first images and relative bounding box coordinates """

    def __init__(self, pooled_height, pooled_width, **kwargs):

        self.pooled_height = pooled_height

        self.pooled_width = pooled_width

        

        super(ROIPoolingLayer, self).__init__(**kwargs)

        

    def compute_output_shape(self, input_shape):

        """ Returns the shape of the ROI Layer output

        """

        feature_map_shape, rois_shape = input_shape # image and roi

        assert feature_map_shape[0] == rois_shape[0]

        batch_size = feature_map_shape[0]

        n_rois = rois_shape[1]

        n_channels = feature_map_shape[3]

        return (batch_size, n_rois, self.pooled_height, 

                self.pooled_width, n_channels)



    def call(self, x):

        """ Maps the input tensor of the ROI layer to its output"""

        

        def curried_pool_rois(x): 

          return ROIPoolingLayer._pool_rois(x[0], x[1], 

                                            self.pooled_height, 

                                            self.pooled_width)

        

        pooled_areas = tf.map_fn(curried_pool_rois, x, dtype=tf.float32)



        return pooled_areas

    

    @staticmethod

    def _pool_rois(feature_map, rois, pooled_height, pooled_width):

        """ Applies ROI pooling for a single image and varios ROIs"""

        

        def curried_pool_roi(roi): 

          return ROIPoolingLayer._pool_roi(feature_map, roi, 

                                           pooled_height, pooled_width)

        

        pooled_areas = tf.map_fn(curried_pool_roi, rois, dtype=tf.float32)

        return pooled_areas

    

    @staticmethod

    def _pool_roi(feature_map, roi, pooled_height, pooled_width):

        """ Applies ROI pooling to a single image and a single region of interest"""



        # Compute the region of interest        

        feature_map_height = int(feature_map.shape[0])

        feature_map_width  = int(feature_map.shape[1])

        

        h_start = tf.cast(feature_map_height * roi[0], 'int32')

        w_start = tf.cast(feature_map_width  * roi[1], 'int32')

        h_end   = tf.cast(feature_map_height * roi[2], 'int32')

        w_end   = tf.cast(feature_map_width  * roi[3], 'int32')

        

        region = feature_map[h_start:h_end, w_start:w_end, :]

        

        # Divide the region into non overlapping areas

        region_height = h_end - h_start

        region_width  = w_end - w_start

        h_step = tf.cast( region_height / pooled_height, 'int32')

        w_step = tf.cast( region_width  / pooled_width , 'int32')

        

        areas = [[(

                    i*h_step, 

                    j*w_step, 

                    (i+1)*h_step if i+1 < pooled_height else region_height, 

                    (j+1)*w_step if j+1 < pooled_width else region_width

                   ) 

                   for j in range(pooled_width)] 

                  for i in range(pooled_height)]

        

        # take the maximum of each area and stack the result

        def pool_area(x): 

          return tf.math.reduce_max(region[x[0]:x[2], x[1]:x[3], :], axis=[0,1])

        

        pooled_features = tf.stack([[pool_area(x) for x in row] for row in areas])

        return pooled_features
def sigmoid_tf(x):

    return 1. / (1. + tf.math.exp(-x))
act__ = 0.5

def preprocessing_for_roi(tensor): # немного кривой код, если будет время доведу до ума

    batch_ = []

    x_min, y_min, x_max, y_max = 1, 1, tensor.shape[1], tensor.shape[2]

    batch_n = 10

    for batch in range(batch_n):

        for row in range(tensor.shape[1]):

            for col in range(tensor.shape[2]):

                act = sigmoid_tf(tensor[batch][row][col][0])

                if tf.math.less(act, act__) is not None:

                    continue

                else:

                    x_min = row

                    y_min = col

                break

            break

        for row in range(x_min, tensor.shape[1]):

            act = sigmoid_tf(tensor[batch][row][y_min][0])

            if tf.math.less(act, act__) is not None:

                x_max = row - 1

        for col in range(y_min, tensor.shape[2]):

            act = sigmoid_tf(tensor[batch][x_min][col][0])

            if tf.math.less(act, act__) is not None:

                y_max = col - 1

        x_Min, y_Min, x_Max, y_Max = tf.truediv(x_min, tensor.shape[1]), tf.truediv(y_min, tensor.shape[2]), tf.truediv(x_max, tensor.shape[1]), tf.truediv(y_max, tensor.shape[2])

        coor = np.array([x_Min, y_Min, x_Max, y_Max])

        batch_.append(np.array([coor]))            

    return np.array(batch_)
pooled_height = 7

pooled_width = 7

batch_size = 10

feature_maps_shape = (batch_size, feature_tensor.shape[1],

                      feature_tensor.shape[2], feature_tensor.shape[3])

feature_maps_tf = tf.placeholder(tf.float32, shape=feature_maps_shape)



roiss_tf = tf.placeholder(tf.float32, shape=(batch_size, 1, 4))

roi_layer = ROIPoolingLayer(pooled_height, pooled_width)

pooled_features = roi_layer([feature_maps_tf, roiss_tf])
output = keras.layers.BatchNormalization()(feature_tensor)



seed = 29

kernek_initializer = keras.initializers.glorot_normal(seed=seed)

# roi_layer = ROIPoolingLayer(pooled_height, pooled_width)



# TODO: добавьте выходы для классификации детекции



output_1 = keras.layers.Conv2D(5, kernel_size=(1, 1), 

                             activation='linear',

                             kernel_regularizer='l2', name="box_output")(output)



# output_2 = roi_layer([output, preprocessing_for_roi(output_1)])([output, output_1])

output_2 = keras.layers.Flatten()(output)

output_2 = keras.layers.BatchNormalization()(output_2)

output_2 = keras.layers.Dropout(0.3)(output_2)

output_2 = keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l1(1e-4),

                            kernel_initializer=kernek_initializer)(output_2)

output_2 = keras.layers.BatchNormalization()(output_2)

output_2 = keras.layers.Dropout(0.3)(output_2)

output_2 = keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l1(1e-4),

                            kernel_initializer=kernek_initializer)(output_2)

output_2 = keras.layers.BatchNormalization()(output_2)

output_2 = keras.layers.Dense(7, activation='softmax', kernel_regularizer=keras.regularizers.l1(1e-4),

                            kernel_initializer=kernek_initializer, name="category_output")(output_2)





model = keras.models.Model(inputs=features.inputs, outputs=[output_1, output_2])

model.summary()
losses = {"box_output": total_loss, "category_output": "categorical_crossentropy"}



adam = keras.optimizers.Adam(lr=1e-4, decay=1e-6)

model.compile(optimizer=adam, 

              loss={"box_output": total_loss, "category_output": "categorical_crossentropy"},

              metrics={'box_output': confidence_loss, 'category_output': 'accuracy'})
def scheduler(epoch):

  if epoch < 5:

    return 1e-4

  else:

    return 1e-4 * tf.math.exp(0.1 * (5 - epoch))

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
batch_size = 10

steps_per_epoch = sum(map(len, boxes.values()), 0) / batch_size



gen = data_generator(boxes, batch_size=batch_size)



checkpoint = keras.callbacks.ModelCheckpoint(

    'weights.{epoch:02d}-{loss:.3f}.hdf5',

    monitor='loss',

    verbose=1,  

    save_best_only=True, 

    save_weights_only=False,

    mode='auto', period=1)



model.fit_generator(generator=gen, 

                    steps_per_epoch=steps_per_epoch,

                    epochs=12, callbacks=[callback])
example = boxes['lag'][10]



_, sample_img = load_img(example['filename'])

pred = model.predict(np.array([sample_img,]))[0]
pred_2 = pred.reshape(23, 37, 5)

pred_2.shape
decoded = decode_prediction(pred_2, conf_thr=0.1)

decoded = sorted(decoded, key=lambda e: -e['conf'])



plt.figure(figsize=(6, 6), dpi=120)

img = draw_boxes(example, decoded[:3], (IMG_WIDTH, IMG_HEIGHT))

plt.imshow(img)

plt.title('{}x{}'.format(*img.shape));
from glob import glob

test_files = sorted(glob('../input/test-fish/test_stg1/test_stg1/*.jpg'))

test_files_2 = sorted(glob('../input/fullfish/test/test/*.jpg'))

len(test_files) + len(test_files_2)
train_files = glob('../input/the-nature-conservancy-fisheries-monitoring/train/YFT/*.jpg')

len(train_files)
from keras.preprocessing.image import load_img, img_to_array

from keras.applications.vgg16 import preprocess_input



def load_image(path, target_size=(IMG_HEIGHT, IMG_WIDTH)):

    img = load_img(path, target_size=target_size)  # загрузка и масштабирование изображения

    array = img_to_array(img)

    return preprocess_input(array)

# генератор последовательного чтения тестовых данных с диска

def predict_generator(files):

    while True:

        for path in files:

            yield np.array([load_image(path)])
pred = model.predict_generator(predict_generator(test_files), len(test_files))
pred_full = model.predict_generator(predict_generator(test_files_2), len(test_files_2))
import pandas as pd

my_submit= pd.DataFrame(columns=['image', 'ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])

my_submit.image = np.arange(0,13153).astype(int)
for i in range(1000):

    my_submit['image'].loc[my_submit.index==i] = os.path.basename(test_files[i])

    my_submit['ALB'].loc[my_submit.index==i] = pred[1][i][0]

    my_submit['BET'].loc[my_submit.index==i] = pred[1][i][1]

    my_submit['DOL'].loc[my_submit.index==i] = pred[1][i][2]

    my_submit['LAG'].loc[my_submit.index==i] = pred[1][i][3]

    my_submit['NoF'].loc[my_submit.index==i] = pred[1][i][4]

    my_submit['OTHER'].loc[my_submit.index==i] = 0. # other не размечена, поэтому забиваю нулями

    my_submit['SHARK'].loc[my_submit.index==i] = pred[1][i][5]

    my_submit['YFT'].loc[my_submit.index==i] = pred[1][i][6]
for i in range(1000, 13153):

    my_submit['image'].loc[my_submit.index==i] = 'test_stg2/' + os.path.basename(test_files_2[i-1000])

    my_submit['ALB'].loc[my_submit.index==i] = pred_full[1][i-1000][0]

    my_submit['BET'].loc[my_submit.index==i] = pred_full[1][i-1000][1]

    my_submit['DOL'].loc[my_submit.index==i] = pred_full[1][i-1000][2]

    my_submit['LAG'].loc[my_submit.index==i] = pred_full[1][i-1000][3]

    my_submit['NoF'].loc[my_submit.index==i] = pred_full[1][i-1000][4]

    my_submit['OTHER'].loc[my_submit.index==i] = 0. # other не размечена, поэтому забиваю нулями

    my_submit['SHARK'].loc[my_submit.index==i] = pred_full[1][i-1000][5]

    my_submit['YFT'].loc[my_submit.index==i] = pred_full[1][i-1000][6]
my_submit.shape
my_submit[980:1010]
my_submit.to_csv('submit.csv', index=False)
# TODO: предскажите класс рыбы для фотографии из тестовой выборки

#

# Подготовьте файл с предсказаниями вероятностей для каждой фотографии:

# image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT

# img_00001.jpg,1,0,0,0,0,...,0

# img_00002.jpg,0.3,0.1,0.6,0,...,0