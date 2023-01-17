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
!git clone https://github.com/fizyr/keras-retinanet.git
%cd keras-retinanet/
!pip install .
!python setup.py build_ext --inplace
import numpy as np
import tensorflow as tf
from tensorflow import keras
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
%cd ..
dataset=pd.read_csv("../input/face-mask-detection-dataset/train.csv", header=None)
dataset = dataset.iloc[1:]
dataset[0] = '../input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images/' + dataset[0].astype(str)
dataset.head()
train_df, test_df = train_test_split(
  dataset, 
  test_size=0.15, 
  shuffle=False
)
train_df.iloc[8131:8132,:]
train_df=train_df.drop([8132], axis=0)
train_df.to_csv(r'train_annot.csv', index = False, header=None)
test_df.to_csv(r'test_annot.csv', index = False, header=None)
clas=pd.read_csv("../input/class-obj-det/clas.csv", header=None)
clas.head()
clas.to_csv(r'clas.csv', index = False, header=None)
def show_image_objects(image_row):

    img_path = image_row["name"]
    box = [
    image_row["x1"], image_row["x2"], image_row["y1"], image_row["y2"]
    ]
    img_dir='../input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images'
    x=os.path.join(img_dir, img_path)
    image = read_image_bgr(x)

    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    draw_box(draw, box, color=(255, 255, 0))

    plt.axis('off')
    plt.imshow(draw)
    plt.show()
df=pd.read_csv("../input/face-mask-detection-dataset/train.csv")
df.head()
show_image_objects(df.iloc[0])
PRETRAINED_MODEL = 'pretrained_model.h5'

URL_MODEL = 'https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5'
urllib.request.urlretrieve(URL_MODEL, PRETRAINED_MODEL)

print('Downloaded pretrained model to ' + PRETRAINED_MODEL)
ANNOTATIONS_FILE = 'train_annot.csv'
CLASSES_FILE = 'clas.csv'
!keras-retinanet/keras_retinanet/bin/train.py --freeze-backbone --random-transform --weights {PRETRAINED_MODEL} --batch-size 8 --steps 500 --epochs 10 csv train_annot.csv clas.csv
!ls snapshots
import pickle
filename = 'model_raw.sav'
pickle.dump(model, open(filename, 'wb'))
model_path = os.path.join('snapshots', sorted(os.listdir('snapshots'), reverse=True)[0])
print(model_path)

model = models.load_model(model_path, backbone_name='resnet50')
model = models.convert_model(model)

labels_to_names = pd.read_csv(CLASSES_FILE, header=None).T.loc[0].to_dict()
def predict(image):
  image = preprocess_image(image.copy())
  image, scale = resize_image(image)

  boxes, scores, labels = model.predict_on_batch(
    np.expand_dims(image, axis=0)
  )

  boxes /= scale

  return boxes, scores, labels
THRES_SCORE = 0.5

def draw_detections(image, boxes, scores, labels):
  for box, score, label in zip(boxes[0], scores[0], labels[0]):
    if score < THRES_SCORE:
        break

    color = label_color(label)

    b = box.astype(int)
    draw_box(image, b, color=color)

    caption = "{} {:.3f}".format(labels_to_names[label], score)
    draw_caption(image, b, caption)
    print(box,score,caption)
def show_detected_objects(image_row):
  img_path = image_row["name"]

  image = read_image_bgr(img_path)

  boxes, scores, labels = predict(image)

  draw = image.copy()
  draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

  true_box = [
    image_row["x1"], image_row["x2"], image_row["y1"], image_row["y2"]
  ]
  draw_box(draw, true_box, color=(255, 255, 0))

  draw_detections(draw, boxes, scores, labels)

  plt.axis('off')
  plt.imshow(draw)
  plt.show()
test_df.columns = ['name', 'x1', 'x2', 'y1','y2','classname']
show_detected_objects(test_df.iloc[10])
show_detected_objects(test_df.iloc[20])
show_detected_objects(test_df.iloc[30])
show_detected_objects(test_df.iloc[40])
model.save_weights("model.h5")
submit=pd.read_csv("../input/face-mask-detection-dataset/submission.csv")
submit = submit.drop_duplicates()
submit.head()
def show_detected_objects_new(image_row):
  img_path = image_row["name"]
  img_dir='../input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images'
  img_path=os.path.join(img_dir, img_path)

  image = read_image_bgr(img_path)

  boxes, scores, labels = predict(image)

  draw = image.copy()
  draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

  draw_detections(draw, boxes, scores, labels)
  
  plt.axis('off')
  plt.imshow(draw)
  plt.show()
show_detected_objects_new(submit.iloc[5])
show_detected_objects_new(submit.iloc[6])
show_detected_objects_new(submit.iloc[9])
show_detected_objects_new(submit.iloc[12])
def predict(image):
  image = preprocess_image(image.copy())
  image, scale = resize_image(image)

  boxes, scores, labels = model.predict_on_batch(
    np.expand_dims(image, axis=0)
  )

  boxes /= scale

  return boxes, scores, labels
THRES_SCORE = 0.5

def draw_detections(image, boxes, scores, labels):
  dimension=[]
  classify=[]

  for box, score, label in zip(boxes[0], scores[0], labels[0]):

    if score < THRES_SCORE:
        break

    color = label_color(label)

    b = box.astype(int)
    draw_box(image, b, color=color)

    caption = "{} {:.3f}".format(labels_to_names[label], score)
    draw_caption(image, b, caption)
   
    classify.append(labels_to_names[label])
    dimension.append(box)
    
  return dimension,classify
def show_detected_objects_fin(image_row):
  img_path = image_row["name"]
  img_dir='../input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images'
  img_path=os.path.join(img_dir, img_path)

  image = read_image_bgr(img_path)

  boxes, scores, labels = predict(image)

  draw = image.copy()
  draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

  draw_detections(draw, boxes, scores, labels)
  
  #plt.axis('off')
  #plt.imshow(draw)
  #plt.show()  

  dimension, classify = draw_detections(draw, boxes, scores, labels)

  dfObj = pd.DataFrame(dimension,columns = ['x1' , 'x2', 'y1', 'y2'])
  dfObj['label'] = classify
  dfObj['name'] = image_row["name"]
  dfObj = dfObj[['name','x1' , 'x2', 'y1', 'y2','label']]
  return dfObj
 
final=pd.DataFrame(columns=['name','x1' , 'x2', 'y1', 'y2','label'])
for i in range(0,len(submit)):
    b=show_detected_objects_fin(submit.iloc[i])
    final=final.append(b,ignore_index = True)
final.head()
final.to_csv(r'submit_8.csv')
