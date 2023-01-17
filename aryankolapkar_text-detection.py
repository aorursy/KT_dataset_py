# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

# Any results you write to the current directory are saved as output.
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
 
# imageio library will be used for image input/output
import imageio
import pandas as pd
import numpy as np
import re
import os
import glob
# this library is needed to read XML files for converting it into CSV
import xml.etree.ElementTree as ET
import shutil
data=pd.read_csv("/kaggle/input/new-csv/train_sort_img.csv")
#read csv file as pd dataframe
for i in range(len(data)):
    data['imageName'][i]='/kaggle/input/resimg/iml'+data['imageName'][i][3:]
data['imageName']
# rename image name to their image paths
data.head()
data=data.rename(columns={'h':'ymax','w':'xmax','height':'width','width':'height'})
#rename columns
data['xmax']=data['xmax']+data['xmin']
data['ymax']=data['ymax']+data['ymin']
# change format of bbox coordinates
#change format of bbox coordinates
x_scale=416/data['width'] 
y_scale=416/data['height']
data['xmin'] = (np.round(data['xmin'] *x_scale))
data['ymin'] = (np.round(data['ymin'] * y_scale))
data['xmax'] = (np.round(data['xmax']*x_scale ))
data['ymax'] = (np.round(data['ymax']* y_scale))
data['height']=416
data['width']=416
data

     


# defining global variable path
image_path = "/kaggle/input/resimg/iml"

 
def loadImages(path):
    # Put files into lists and return them as one list of size 4

    image_files = sorted([os.path.join(path, '', file)
         for i in range(len(data))
                          for file in os.listdir(path) if      file.endswith('{}'.format(data['imageName'][i][35:]))])
 
    return image_files
image_file=loadImages(image_path)
image_file
#return a np array of resized images 
def processing(data):
    img = [cv2.imread(i, cv2.IMREAD_UNCHANGED) for i in data[:]]
    height = 416
    width = 416
    dim = (width, height)
    res_img = []
    for i in range(len(img)):
        res = cv2.resize(img[i], dim, interpolation=cv2.INTER_LINEAR)
        res_img.append(res)

     

    
 
    #no_noise = []
    
    #for i in range(len(res_img)):
        #blur = cv2.GaussianBlur(res_img[i], (5, 5), 0)
        #no_noise.append(blur)
 


    
    
    return res_img
image_file=np.asarray(image_file)
# converts list to np array
# makes np array of unique images and gets rid of repititions
image_file=np.unique(image_file)
image_file

no_noise_images=processing(image_file)
no_noise_images=np.asarray(no_noise_images)
#makes dictionary of image path assigned with its np array
processed_images={}
for A, B in zip(image_file, no_noise_images):
    processed_images[A] = B

print(processed_images)

# try out on one image if our operations are correct
cv2.rectangle(processed_images['/kaggle/input/resimg/iml/00_00.jpg'],(int(data['xmin'][0]),int(data['ymin'][0])),(int(data['xmax'][0]),int(data['ymax'][0])),(0,255,0),2)
plt.imshow(processed_images['/kaggle/input/resimg/iml/00_00.jpg'])
# shows image

labels=np.zeros((len(image_file),9,5))
n=0
for i in image_file:
    df= data[(data['imageName'] == i)]
    row_list=[]
    for rows in df.itertuples(): 
        
    # Create list for the current row 
        my_list =[1, rows.xmin, rows.ymin,rows.xmax,rows.ymax] 
        row_list.append(my_list) 
    # append the list to the final list 
        p=0
        for m in row_list:
            labels[n,p]=m
            p=p+1
    n=n+1
  
 
    

     
data1=data.copy()
# makes a variable data1 which is the copy of data since we want to modify the format and keep the original one intact
data1['tag']='text'
data1.drop(['width','height','lex','address'],inplace=True,axis=1)
data1=data1[['imageName','xmin','ymin','xmax','ymax','tag']]
#performing some format modifications so as to make image files in format of pretrained detection model
data1 = data1.astype({"xmin": int, "ymin": int,'xmax':int,'ymax':int}) # assinging datatypes to columns
data1
import keras
keras.__version__
ANNOTATIONS_FILE='annotations.csv'
CLASSES_FILE='classes.csv'
# file names are stored as variables
data1.to_csv(ANNOTATIONS_FILE,index=False,header=None) # storing annotations from data1 as csv file
classes = set(['text'])

with open(CLASSES_FILE, 'w') as f:
    for i, line in enumerate(sorted(classes)):
        f.write('{},{}\n'.format(line,i))
!head annotations.csv
!git clone https://github.com/fizyr/keras-retinanet.git # cloning github repository
%cd keras-retinanet/

!pip install .
!python setup.py build_ext --inplace
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
from sklearn.model_selection import train_test_split
import urllib
import os
import csv
import cv2
import time
from PIL import Image

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

%matplotlib inline
%config InlineBackend.figure_format='retina'

register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 22, 10

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
os.makedirs("snapshots", exist_ok=True)
PRETRAINED_MODEL = './snapshots/_pretrained_model.h5'

URL_MODEL = 'https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5'
urllib.request.urlretrieve(URL_MODEL, PRETRAINED_MODEL)

print('Downloaded pretrained model to ' + PRETRAINED_MODEL)
!ls snapshots # viewing contents of snapshot folder created
model_path = '/kaggle/input/resnet50-weights/resnet50_csv_10.h5' # downloaded model
#model_path = os.path.join('snapshots', sorted(os.listdir('snapshots'), reverse=True)[0]) 
print(model_path)
model = models.load_model(model_path, backbone_name='resnet50') #loading the model so that it can be used
model = models.convert_model(model)
labels_to_names = pd.read_csv('/kaggle/input/anotoo/classes.csv', header=None).T.loc[0].to_dict()
labels_to_names
def predict(image):
    image = preprocess_image(image.copy())
    image, scale = resize_image(image)

    boxes, scores, labels = model.predict_on_batch(
    np.expand_dims(image, axis=0)
    )

    boxes /= scale

    return boxes, scores, labels
#THRES_SCORE = 1.0

def draw_detections(image, boxes, scores, labels):
    THRES_SCORE = 0.2
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < THRES_SCORE:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(image, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(image, b, caption)

def show_detected_objects(image_row,processed_images):
    img_path = image_row.imageName
  
    image = processed_images[img_path]

    boxes, scores, labels = predict(image)

    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    true_box = [
    image_row.xmin, image_row.ymin, image_row.xmax, image_row.ymax
    ]
    #draw_box(draw, true_box, color=(255, 255, 0))

    #draw_detections(draw, boxes, scores, labels)

    plt.axis('off')
    #plt.imshow(draw[boxes[:,0,2]:boxes[:,0,3],boxes[:,0,0]:boxes[:,0,1]])
    plt.imshow(draw)
    plt.show()
    print(boxes[:,0,:])
    print(scores[0][0])
    print(draw.shape)
    #return draw[135:192,164:250]
img=show_detected_objects(data1.iloc[0],processed_images)
# will show the bounding box proposals after applying a certain thrshold to their scores 
# functon to find iou b/w two rectangular bounding boxes

def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou
