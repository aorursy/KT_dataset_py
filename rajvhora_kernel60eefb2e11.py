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
!nvidia-smi
!git clone https://github.com/fizyr/keras-retinanet.git
!pip install --upgrade keras
%cd /kaggle/working/keras-retinanet/



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
wheat_df = pd.read_csv('/kaggle/input/global-wheat-detection/train.csv')

wheat_df.head()
xmin = []

ymin = []

width1 = []

height1 = []

xmax = []

ymax = []

imagepath = []

classname = []

for index, row in wheat_df.iterrows():

    imagepath.append('/kaggle/input/global-wheat-detection/train/'+row['image_id']+'.jpg')

    i = row['bbox']

    i = i[1:-1]

    i = i.split(',')

    newI = []

    for j in i:

        newI.append(int(round(float(j))))

    xmin.append(newI[0])

    ymin.append(newI[1])

    xmax.append(newI[0]+newI[2])

    ymax.append(newI[1]+newI[3])

    classname.append('wheat')

print("Listed all.")
dataset = pd.DataFrame()

dataset["image_name"] = imagepath

dataset["x_min"] = xmin

dataset["y_min"] = ymin

dataset["x_max"] = xmax

dataset["y_max"] = ymax

dataset["class_name"] = classname



dataset.head()
def show_image_objects(image_row):



  img_path = image_row.image_name

  box = [

    image_row.x_min, image_row.y_min, image_row.x_max, image_row.y_max

  ]



  image = read_image_bgr(img_path)



  draw = image.copy()

  draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)



  draw_box(draw, box, color=(255, 255, 0))



  plt.axis('off')

  plt.imshow(draw)

  plt.show()
show_image_objects(dataset.iloc[1])
train_df, test_df = train_test_split(

  dataset, 

  test_size=0.2, 

  random_state=RANDOM_SEED

)
ANNOTATIONS_FILE = '/kaggle/working/annotations.csv'

CLASSES_FILE = '/kaggle/working/classes.csv'
train_df.to_csv(ANNOTATIONS_FILE, index=False, header=None)
classes = set(['wheat'])



with open(CLASSES_FILE, 'w') as f:

  for i, line in enumerate(sorted(classes)):

    f.write('{},{}\n'.format(line,i))


!head /kaggle/working/classes.csv
!head /kaggle/working/annotations.csv
!ls
PRETRAINED_MODEL = './snapshots/_pretrained_model.h5'



URL_MODEL = 'https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5'

urllib.request.urlretrieve(URL_MODEL, PRETRAINED_MODEL)



print('Downloaded pretrained model to ' + PRETRAINED_MODEL)
!keras_retinanet/bin/train.py --freeze-backbone --random-transform --weights {PRETRAINED_MODEL} --batch-size 8 --steps 100 --epochs 10 csv /kaggle/working/annotations.csv /kaggle/working/classes.csv
!ls snapshots
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
THRES_SCORE = 0.3



def draw_detections(image, boxes, scores, labels):

  for box, score, label in zip(boxes[0], scores[0], labels[0]):

    if score < THRES_SCORE:

        break



    color = label_color(label)



    b = box.astype(int)

    draw_box(image, b, color=color)



    caption = "{} {:.3f}".format(labels_to_names[label], score)

    draw_caption(image, b, caption)
def show_detected_objects(image_row):

  img_path = image_row.image_name

  

  image = read_image_bgr(img_path)



  boxes, scores, labels = predict(image)



  draw = image.copy()

  draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)



  true_box = [

    image_row.x_min, image_row.y_min, image_row.x_max, image_row.y_max

  ]

  draw_box(draw, true_box, color=(255, 255, 0))



  draw_detections(draw, boxes, scores, labels)



  plt.axis('off')

  plt.imshow(draw)

  plt.show()
test_df.head(n=10)
show_detected_objects(test_df.iloc[3])
def test_image_bbox(imagepath):

  image = read_image_bgr(imagepath)

  boxes, scores, labels = predict(image)



  draw = image.copy()

  draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

  draw_detections(draw, boxes, scores, labels)



  plt.axis('off')

  plt.imshow(draw)

  plt.show()
test_image_bbox('/kaggle/input/global-wheat-detection/test/796707dd7.jpg')
ss = pd.read_csv('/kaggle/input/global-wheat-detection/sample_submission.csv')

PATH = '/kaggle/input/global-wheat-detection/test/'

listofboxes = []

listofimgname = []

for i in range (0,10):

    row = ss.iloc[i]

    img = row['image_id']

    listofimgname.append(img)

    image_ext = img + '.jpg'

    path = os.path.join(PATH,image_ext)

    image = read_image_bgr(path)

    boxes, scores, labels = predict(image)

    finalstring = ""

    for box, score, label in zip(boxes[0], scores[0], labels[0]):

        finalstring = finalstring + str(score)+ " "

        for j in box:

            finalstring = finalstring + str(int(round(j))) + " "

    print(finalstring)

    finalstring = finalstring[:-1]

    listofboxes.append(finalstring)

finaldata = pd.DataFrame()

finaldata['image_id'] = listofimgname

finaldata['PredictionString'] = listofboxes

finaldata.head()
finaldata.to_csv('/kaggle/working/submission.csv',index=False)