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
import pandas as pd
import numpy as np
import os
import pydicom
import cv2
import skimage
from skimage import color
from skimage import exposure

import tensorflow as tf
import numpy as np
import tensorflow 


def bbToYoloFormat(bb):
    """
    converts (left, top, right, bottom) to
    (center_x, center_y, center_w, center_h)
    """
    x1, y1, x2, y2 = np.split(bb, 4, axis=1) 
    w = x2 - x1
    h = y2 - y1
    c_x = x1 + w / 2
    c_y = y1 + h / 2
    
    return np.concatenate([c_x, c_y, w, h], axis=-1)

def findBestPrior(bb, priors):
    """
    Given bounding boxes in yolo format and anchor priors
    compute the best anchor prior for each bounding box
    """
    w1, h1 = bb[:, 2], bb[:, 3]
    w2, h2 = priors[:, 0], priors[:, 1]
    
    # overlap, assumes top left corner of both at (0, 0)
    horizontal_overlap = np.minimum(w1[:, None], w2)
    vertical_overlap = np.minimum(h1[:, None], h2)
    
    intersection = horizontal_overlap * vertical_overlap
    union = (w1 * h1)[:, None] + (w2 * h2) - intersection
    iou = intersection / union
    return np.argmax(iou, axis=1)

def processGroundTruth(bb, labels, priors, network_output_shape):
    """
    Given bounding boxes in normal x1,y1,x2,y2 format, the relevant labels in one-hot form,
    the anchor priors and the yolo model's output shape
    build the y_true vector to be used in yolov2 loss calculation
    """
    bb = bbToYoloFormat(bb) / 32
    best_anchor_indices = findBestPrior(bb, priors)
    
    responsible_grid_coords = np.floor(bb).astype(np.uint32)[:, :2]
    
    values = np.concatenate((
        bb, np.ones((len(bb), 1)), labels
    ), axis=1)
    
    x, y = np.split(responsible_grid_coords, 2, axis=1)
    y = y.ravel()
    x = x.ravel()
    
    y_true = np.zeros(network_output_shape)    
    y_true[y, x, best_anchor_indices] = values
    
    return y_true




class det_gen(tensorflow.keras.utils.Sequence):
    'Generates data from a Dataframe'
    def __init__(self,csv_path,patientId , img_path ,batch_size=8, dim=(256,256), n_channels=3,
                 n_classes=1, shuffle=True,transform=None , only_positive=True, preprocess = None):
        
        self.df = pd.read_csv(csv_path)
        if only_positive:
            self.df= self.df[self.df["Target"]==1]
            
        self.img_path = img_path
        self.patient_ids = patientId
        
        self.batch_size = batch_size
        self.nb_iteration = int(len(self.patient_ids)/self.batch_size)
        self.dim = dim
        self.n_channels= n_channels
        
        self.preprocess =preprocess
        
        
        self.TINY_YOLOV2_ANCHOR_PRIORS = np.array([1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]).reshape(5, 2)
        self.network_output_shape = (8,8,5,6)
    
    
                    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.nb_iteration

    def __getitem__(self, index):
        'Generate one batch of data'
        
        indicies = range(index, min(index+self.batch_size ,len(self.patient_ids) ))
            
        patientIds = self.patient_ids[indicies]
        X = np.zeros((self.batch_size, self.dim[0], self.dim[1],self.n_channels))
        y_boxes = []
        y = np.zeros((self.batch_size,self.network_output_shape[0],self.network_output_shape[1],self.network_output_shape[2],self.network_output_shape[3]))
        output_labels = []
        
        
        for index , patientId in enumerate(patientIds):
            filtered_df = self.df[self.df["patientId"] == patientId]
            img_path = os.path.join(self.img_path,patientId+".dcm" )
            img = self.load_img(img_path)
            
            X[index]= img
            y_boxes = []
            labels = []
            
            for i, row in filtered_df.iterrows():
                xmin = int(row['x'])
                ymin = int(row['y'])
                xmax = int(xmin + row['width'])
                ymax = int(ymin + row['height'])
                
                xmin = int((xmin/1024)*self.dim[0])
                xmax = int((xmax/1024)*self.dim[0])
                
                ymin = int((ymin/1024)*self.dim[1])
                ymax = int((ymax/1024)*self.dim[1])
                y_boxes.append([xmin,ymin,xmax,ymax])
                
                labels.append([1])
                
            
            #run preprocess_bboxes
            y[index] = processGroundTruth(np.array(y_boxes),np.array(labels), self.TINY_YOLOV2_ANCHOR_PRIORS , self.network_output_shape)
            

        return X, y
    
    
    
    def load_img(self,img_path):
        dcm_data = pydicom.read_file(img_path)
        a = dcm_data.pixel_array

        a=cv2.resize(a,(self.dim))
        if self.n_channels == 3:
            a = skimage.color.gray2rgb(a)
        
        
        
        if self.preprocess != None:
            a= self.preprocess(a)
        
        a = exposure.equalize_adapthist(a)
        
        return a


import random

csv_path= "/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv"
images_path = "/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/"

df = pd.read_csv(csv_path)
df= df[df["Target"]==1]

random.seed(42)
patient_ids = df["patientId"].unique()
random.shuffle(patient_ids)

validation_split = 0.2
patient_ids_train = patient_ids[int(len(patient_ids)*(validation_split)):]
patient_ids_validation = patient_ids[:int(len(patient_ids)*(validation_split))]

from tensorflow.keras.applications.densenet import preprocess_input
train_generator = det_gen(csv_path,patient_ids_train, images_path ,preprocess=None )
validation_generator = det_gen(csv_path,patient_ids_validation, images_path , preprocess=None)
X,Y = next(enumerate(validation_generator))[1]
Y.shape , X.shape
import matplotlib.pyplot as plt
plt.imshow(X[2])
plt.show()
color = (255, 0, 0) 
  
# Line thickness of 2 px 
thickness = 2


for xx,yy in zip(X,Y):
    for bbox in yy:
        image = cv2.rectangle(xx, (bbox[0],bbox[1]), (bbox[2],bbox[3]), color, thickness)

    plt.imshow(image)
    plt.show()
    

!git clone https://github.com/ahmadelsallab/MultiCheXNet.git
from MultiCheXNet.utils.Encoder import Encoder
from MultiCheXNet.utils.Detector import Detector
from MultiCheXNet.utils.ModelBlock import ModelBlock
encoder = Encoder()
detector_head = Detector(encoder, 256,1)
model = ModelBlock.add_heads(encoder ,[detector_head])
model.summary()
def loss( y_true, y_pred):
        TINY_YOLOV2_ANCHOR_PRIORS = np.array([1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]).reshape(5, 2)
        TINY_YOLOV2_ANCHOR_PRIORS = tf.convert_to_tensor(TINY_YOLOV2_ANCHOR_PRIORS, dtype= tf.float32)
        
        n_cells = y_pred.get_shape().as_list()[1]
        y_true = tf.reshape(y_true, tf.shape(y_pred), name='y_true')
        y_pred = tf.identity(y_pred, name='y_pred')

        #### PROCESS PREDICTIONS ####
        # get x-y coords (for now they are with respect to cell)
        predicted_xy = tf.nn.sigmoid(y_pred[..., :2])

        # convert xy coords to be with respect to image
        cell_inds = tf.range(n_cells, dtype=tf.float32)
        predicted_xy = tf.stack((
            predicted_xy[..., 0] + tf.reshape(cell_inds, [1, -1, 1]),
            predicted_xy[..., 1] + tf.reshape(cell_inds, [-1, 1, 1])
        ), axis=-1)

        # compute bb width and height
        predicted_wh = TINY_YOLOV2_ANCHOR_PRIORS * tf.exp(y_pred[..., 2:4])

        # compute predicted bb center and width
        predicted_min = predicted_xy - predicted_wh / 2
        predicted_max = predicted_xy + predicted_wh / 2

        predicted_objectedness = tf.nn.sigmoid(y_pred[..., 4])
        predicted_logits = tf.nn.softmax(y_pred[..., 5:])

        #### PROCESS TRUE ####
        true_xy = y_true[..., :2]
        true_wh = y_true[..., 2:4]
        true_logits = y_true[..., 5:]

        true_min = true_xy - true_wh / 2
        true_max = true_xy + true_wh / 2

        #### compute iou between ground truth and predicted (used for objectedness) ####
        intersect_mins = tf.maximum(predicted_min, true_min)
        intersect_maxes = tf.minimum(predicted_max, true_max)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = predicted_wh[..., 0] * predicted_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = intersect_areas / union_areas

        #### Compute loss terms ####
        responsibility_selector = y_true[..., 4]

        xy_diff = tf.square(true_xy - predicted_xy) * responsibility_selector[..., None]
        xy_loss = tf.reduce_sum(xy_diff, axis=[1, 2, 3, 4])

        wh_diff = tf.square(tf.sqrt(true_wh) - tf.sqrt(predicted_wh)) * responsibility_selector[..., None]
        wh_loss = tf.reduce_sum(wh_diff, axis=[1, 2, 3, 4])

        obj_diff = tf.square(iou_scores - predicted_objectedness) * responsibility_selector
        obj_loss = tf.reduce_sum(obj_diff, axis=[1, 2, 3])

        best_iou = tf.reduce_max(iou_scores, axis=-1)
        no_obj_diff = tf.square(0 - predicted_objectedness) * tf.cast(best_iou < 0.6, dtype=tf.float32)[..., None] * (
                    1 - responsibility_selector)
        no_obj_loss = tf.reduce_sum(no_obj_diff, axis=[1, 2, 3])

        clf_diff = tf.square(true_logits - predicted_logits) * responsibility_selector[..., None]
        clf_loss = tf.reduce_sum(clf_diff, axis=[1, 2, 3, 4])

        object_coord_scale = 5
        object_conf_scale = 1
        noobject_conf_scale = 1
        object_class_scale = 1

        loss = object_coord_scale * (xy_loss + wh_loss) + \
               object_conf_scale * obj_loss + noobject_conf_scale * no_obj_loss + \
               object_class_scale * clf_loss
        
        if np.isnan(loss):
            import ipdb;ipdb.set_trace()
        
        
        return loss
import tensorflow as tf
def calls(model_name):
  checkpoint = tf.keras.callbacks.ModelCheckpoint(model_name,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='min',
                             period=1)
  
  early = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                        min_delta=0,
                        patience=10,
                        verbose=1,
                        mode='min')
  
  class myCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
      if (logs.get('val_accuracy') > 0.998):
        print ('\nReached 0.998 Validation accuracy!')
        self.model.stop_training = True

  my_call = myCallBack()

  reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=5,
                              verbose=1, mode='min', min_delta=0,
                              cooldown=0, min_lr=0)
  
  lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-5 * 10**(epoch / 10))
  
  return [checkpoint, early, reduce_lr]
monitor = calls('model.h5')
from tensorflow.keras.optimizers import Adam
model.compile(loss= detector_head.loss, optimizer= Adam(1e-4))
num_epochs = 10
model.fit_generator(train_generator, validation_data= validation_generator ,epochs=num_epochs ,
                   callbacks=monitor)


