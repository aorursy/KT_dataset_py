import os
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plot
import xml.etree.ElementTree as ET
from skimage.measure import regionprops

import keras
from keras.applications.mobilenet import MobileNet
from keras.layers import Flatten, Dense, Reshape, MaxPooling2D, Conv2D, Input, UpSampling2D, Add
from keras.models import Model
from keras.optimizers import adagrad
from keras.preprocessing import image
import tensorflow as tf

%matplotlib inline
model = MobileNet(weights='imagenet')
model_fpn = Model(inputs=model.input, outputs=model.get_layer('conv_pw_13_relu').output)
Add()

feature_scope  = 256
end = model.get_layer('conv_pw_13_relu').output
end = Conv2D(feature_scope, (1,1), strides=(1, 1), padding='same')(end)
end_1 = UpSampling2D(size=(2, 2), data_format=None , interpolation= 'bilinear')(end)
#conv_pw_11_relu
# do 1x1 conv to get same depth
conv_1x1_1 = model.get_layer('conv_pw_11_relu').output
conv_1x1_1 = Conv2D(feature_scope, (1,1), strides=(1, 1), padding='same')(conv_1x1_1)
Added_1 = Add()([end_1,conv_1x1_1])

end_2 = UpSampling2D(size=(2, 2), data_format=None , interpolation= 'bilinear')(Added_1)
#conv_pw_5_relu
conv_1x1_2 = model.get_layer('conv_pw_5_relu').output
conv_1x1_2 = Conv2D(feature_scope, (1,1), strides=(1, 1), padding='same')(conv_1x1_2)
Added_2 = Add()([end_2,conv_1x1_2])

end_3 = UpSampling2D(size=(2, 2), data_format=None , interpolation= 'bilinear')(Added_2)
#conv_pw_5_relu
conv_1x1_3 = model.get_layer('conv_pw_3_relu').output
conv_1x1_3 = Conv2D(feature_scope, (1,1), strides=(1, 1), padding='same')(conv_1x1_3)
Added_3 = Add()([end_3,conv_1x1_3])

end_4 = UpSampling2D(size=(2, 2), data_format=None , interpolation= 'bilinear')(Added_3)
#conv_pw_5_relu
conv_1x1_4 = model.get_layer('conv_pw_1_relu').output
conv_1x1_4 = Conv2D(feature_scope, (1,1), strides=(1, 1), padding='same')(conv_1x1_4)
Added_4 = Add()([end_4,conv_1x1_4])

model_fpn_test_2 = Model(inputs=model.input, outputs=[Added_1,Added_2,Added_3,Added_4])
#optimizer = optimizers.Adam(lr=0.001)
#model_fpn_test_2.compile(loss='mean_squared_error', optimizer=optimizer)
model_fpn_test_2.summary()
GRID_Y, GRID_X = 7,7
IMAGE_H, IMAGE_W = 224, 224
GRID_H, GRID_W = IMAGE_H/GRID_Y, IMAGE_W/GRID_X
CLASS_NUM = 2
SCALE_NOOB, SCALE_CONF, SCALE_COOR, SCALE_PROB = 0.5, 5.0, 5.0, 1.0
BATCH_SIZE = 16
BOX = 2
ANCHORS = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52] # w, h scaling pairs
classes = {'pothole': (1,0),
           'no_pothole': (0,0)}


def load_image(image_path):
    
    img = cv.imread(image_path)
    size = img.shape
    img = cv.resize(img,(IMAGE_H, IMAGE_W))
    return (img, size)

def get_feature_map(image_size, objects):
    
    '''
        #1) change xmin, ymin ..... as per image resizing
        #2) find center and responsible grid
        #3) change x,y as per grid
        #4) change w, h as per image
    '''
    #Feature map for describing the boxes.
    #Predictions of the model will be compared against this.
    feature = np.zeros((GRID_Y,GRID_X,BOX,5 + CLASS_NUM), dtype=float)
    x_ratio = IMAGE_W/image_size[1]
    y_ratio = IMAGE_H/image_size[0]
    for bbox_label in objects:
        xmin, ymin, xmax, ymax, classname = bbox_label
        if classname == 'p': 
            classname = 'pothole'
        else:
            classname = 'no_pothole'
        # Change the bbox labels to accomodate image resizing.
        xmin = xmin*x_ratio
        ymin = ymin*y_ratio
        xmax = xmax*x_ratio
        ymax = ymax*y_ratio
        # Find center and grid responsible for the object.
        cx = (xmax-xmin)/2
        cy = (ymax-ymin)/2
        # Find the row and column of grid containing center.
        gridx = int(np.floor(cx/(GRID_W)))
        gridy = int(np.floor(cy/(GRID_H)))
        # Normalize center.
        cx = (cx - (GRID_W*(1+gridx)))/GRID_W
        cy = (cy - (GRID_H*(1+gridy)))/GRID_H
        w = xmax-xmin
        h = ymax-ymin
        w = w/IMAGE_W
        h = h/IMAGE_H
        feature[gridx, gridy, :, :5] = cx, cy, w, h, 1
        feature[gridx, gridy, :, 5:] = classes[classname]      
    return feature

def get_data(data_folder):
    '''Loads image and tensor data.'''
    
    dataX = []
    dataY = []
    # Parse the input folder.
    for file in os.listdir(data_folder):
        if file.split('.')[1] == 'xml':
            tree = ET.parse(os.path.join(data_folder, file))
            root = tree.getroot()
            filename = root.find('filename').text
            objects = []
            for member in root.findall('object'):
                classname = member[0].text
                xmin = int(member[4][0].text)
                ymin = int(member[4][1].text)
                xmax = int(member[4][2].text)
                ymax = int(member[4][3].text)
                objects.append([xmin, ymin, xmax, ymax, classname])
            # Load the image into an array.
            image, size = load_image(os.path.join(data_folder, filename))
            # Load the output feature into an array.
            feature_map = get_feature_map(size, objects)
            dataX.append(image)
            dataY.append(feature_map)
    return np.array(dataX), np.array(dataY)
images, labels = get_data("../input/data/train")
labels.shape
def get_model(shape):
    '''Returns the CNN Model.'''
    
    base_model = model_fpn_test_2
    x = base_model.output[0]  
    x = Conv2D(128, (2,2),strides=(2,2))(x)
    x = Conv2D(BOX*(5+CLASS_NUM), (1,1), strides=(1,1))(x)
    x = Reshape((GRID_X, GRID_Y, BOX, (5 + CLASS_NUM)))(x)
    
    #x = Conv2D(9, (1,1))(x)
    model = Model(input=base_model.input, output=x)
    return(model)
model = get_model((IMAGE_W, IMAGE_H,3))
model.summary()
def custom_loss(y_true, y_pred, use_anchor = False):
    ### Adjust prediction
    # adjust x and y      
    pred_box_xy = tf.sigmoid(y_pred[:,:,:,:,:2])
    
    # adjust w and h
    if use_anchor:
        anch_weights = np.reshape(ANCHORS, [1,1,1,BOX,2])
    else:
        anch_weights = np.ones((1,1,1,BOX,2))
    pred_box_wh = tf.exp(y_pred[:,:,:,:,2:4]) * anch_weights
    pred_box_wh = tf.sqrt(pred_box_wh / np.reshape([float(GRID_X), float(GRID_Y)], [1,1,1,1,2]))
    
    # adjust confidence
    pred_box_conf = tf.expand_dims(tf.sigmoid(y_pred[:, :, :, :, 4]), -1)
    
    # adjust probability
    pred_box_prob = tf.nn.softmax(y_pred[:, :, :, :, 5:])
    
    y_pred = tf.concat([pred_box_xy, pred_box_wh, pred_box_conf, pred_box_prob], 4)
    
    ### Adjust ground truth
    # adjust x and y
    center_xy = .5*(y_true[:,:,:,:,0:2] + y_true[:,:,:,:,2:4])
    center_xy = center_xy / np.reshape([(float(IMAGE_W)/GRID_X), (float(IMAGE_H)/GRID_Y)], [1,1,1,1,2])
    true_box_xy = center_xy - tf.floor(center_xy)
    
    # adjust w and h
    true_box_wh = (y_true[:,:,:,:,2:4] - y_true[:,:,:,:,0:2])
    true_box_wh = tf.sqrt(true_box_wh / np.reshape([float(IMAGE_W), float(IMAGE_H)], [1,1,1,1,2]))
    
    # adjust confidence
    pred_tem_wh = tf.pow(pred_box_wh, 2) * np.reshape([GRID_X, GRID_Y], [1,1,1,1,2])
    pred_box_area = pred_tem_wh[:,:,:,:,0] * pred_tem_wh[:,:,:,:,1]
    pred_box_ul = pred_box_xy - 0.5 * pred_tem_wh
    pred_box_bd = pred_box_xy + 0.5 * pred_tem_wh
    
    true_tem_wh = tf.pow(true_box_wh, 2) * np.reshape([GRID_X, GRID_Y], [1,1,1,1,2])
    true_box_area = true_tem_wh[:,:,:,:,0] * true_tem_wh[:,:,:,:,1]
    true_box_ul = true_box_xy - 0.5 * true_tem_wh
    true_box_bd = true_box_xy + 0.5 * true_tem_wh
    
    intersect_ul = tf.maximum(pred_box_ul, true_box_ul) 
    intersect_br = tf.minimum(pred_box_bd, true_box_bd)
    intersect_wh = intersect_br - intersect_ul
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect_area = intersect_wh[:,:,:,:,0] * intersect_wh[:,:,:,:,1]
    
    iou = tf.truediv(intersect_area, true_box_area + pred_box_area - intersect_area)
    best_box = tf.equal(iou, tf.reduce_max(iou, [3], True)) 
    best_box = tf.to_float(best_box)
    true_box_conf = tf.expand_dims(best_box * y_true[:,:,:,:,4], -1)
    
    # adjust confidence
    true_box_prob = y_true[:,:,:,:,5:]
    
    y_true = tf.concat([true_box_xy, true_box_wh, true_box_conf, true_box_prob], 4)
    #y_true = tf.Print(y_true, [true_box_wh], message='DEBUG', summarize=30000)    
    
    ### Compute the weights
    weight_coor = tf.concat(4 * [true_box_conf], 4)
    weight_coor = SCALE_COOR * weight_coor
    
    weight_conf = SCALE_NOOB * (1. - true_box_conf) + SCALE_CONF * true_box_conf
    
    weight_prob = tf.concat(CLASS_NUM * [true_box_conf], 4) 
    weight_prob = SCALE_PROB * weight_prob 
    
    weight = tf.concat([weight_coor, weight_conf, weight_prob], 4)
    
    ### Finalize the loss
    loss = tf.pow(y_pred - y_true, 2)
    loss = loss * weight
    loss = tf.reshape(loss, [-1, GRID_X*GRID_Y*BOX*(4 + 1 + CLASS_NUM)])
    loss = tf.reduce_sum(loss, 1)
    loss = .5 * tf.reduce_mean(loss)
    
    return loss
finetune_rate = 1e-5
train_rate = finetune_rate * 100
adg = adagrad(lr=train_rate, decay=0.0005)

model.compile(loss=custom_loss, optimizer=adg)
model.fit(x=images, y=labels, epochs=20, verbose=True)
class BoundBox:
    def __init__(self, class_num, x=0., y=0., w = 0., h = 0., c = None):
        self.x, self.y, self.w, self.h, self.c = x,y,w,h,c if c is not None else 0.
        self.probs = np.zeros((class_num,))
        if c is not None:
            self.probs[c] = 1
        self.normed = False
        
    def norm_dims(self, image):
        if not self.normed:
            self.x, self.y, self.w, self.h = self.x/image.shape[1], self.y/image.shape[0], self.w/image.shape[1], self.h/image.shape[0]
        self.normed = True
    
    def unnorm_dims(self, image):
        if self.normed:
            self.x, self.y, self.w, self.h = self.x*image.shape[1], self.y*image.shape[0], self.w*image.shape[1], self.h*image.shape[0]
        self.normed = False
    
    def iou(self, box):
        intersection = self.intersect(box)
        union = self.w*self.h + box.w*box.h - intersection
        return intersection/union
        
    def intersect(self, box):
        width  = self.__overlap([self.x-self.w/2, self.x+self.w/2], [box.x-box.w/2, box.x+box.w/2])
        height = self.__overlap([self.y-self.h/2, self.y+self.h/2], [box.y-box.h/2, box.y+box.h/2])
        return width * height
        
    def __overlap(self, interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2,x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2,x4) - x3
            

def sigmoid(x):
    return 1. / (1.  + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)           

class_color = lambda class_lab: plot.cm.nipy_spectral((class_lab+1)/(CLASS_NUM+1))[:3]

def interpret_netout(image, netout, threshold, use_anchor = False):
    boxes = []

    # interpret the output by the network
    for row in range(GRID_X):
        for col in range(GRID_X):
            for b in range(BOX):
                box = BoundBox(CLASS_NUM)

                # first 5 weights for x, y, w, h and confidence
                box.x, box.y, box.w, box.h, box.c = netout[row,col,b,:5]

                box.x = (col + sigmoid(box.x)) / GRID_X
                box.y = (row + sigmoid(box.y)) / GRID_Y
                if use_anchor:
                    anch_w = ANCHORS[2 * b + 0]
                    anch_h = ANCHORS[2 * b + 1]
                else:
                    anch_w, anch_h = 1, 1
                box.w = anch_w * np.exp(box.w) / GRID_X
                box.h = anch_h * np.exp(box.h) / GRID_Y
                box.c = sigmoid(box.c)

                # last 20 weights for class likelihoods
                classes = netout[row,col,b,5:]
                box.probs = softmax(classes) * box.c
                box.probs *= box.probs > threshold

                boxes.append(box)

    # suppress non-maximal boxes
    for c in range(CLASS_NUM):
        sorted_indices = list(reversed(np.argsort([box.probs[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            
            if boxes[index_i].probs[c] == 0: 
                continue
            else:
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    
                    if boxes[index_i].iou(boxes[index_j]) >= 0.4:
                        boxes[index_j].probs[c] = 0

    # draw the boxes using a threshold
    for box in boxes:
        max_indx = np.argmax(box.probs)
        max_prob = box.probs[max_indx]
        
        if max_prob > threshold:
            xmin  = int((box.x - box.w/2) * image.shape[1])
            xmax  = int((box.x + box.w/2) * image.shape[1])
            ymin  = int((box.y - box.h/2) * image.shape[0])
            ymax  = int((box.y + box.h/2) * image.shape[0])


            cv.rectangle(image, (xmin,ymin), (xmax,ymax), class_color(max_indx), 2)
            cv.putText(image, '%d' % max_indx, (xmin, ymin - 12), 0, 4e-3 * image.shape[0], (0,255,0), 2)
            
    return image
img_path = "../input/data/test/49.jpeg"
img = cv.imread(img_path)
img = cv.resize(img, (IMAGE_W, IMAGE_H))
img = np.expand_dims(img, axis=0)
yolo_output = model.predict(img)[0]
out_image = interpret_netout(img, yolo_output, 0)[0]
plot.imshow(out_image)
