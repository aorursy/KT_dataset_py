!git clone https://github.com/fizyr/keras-retinanet.git
%cd keras-retinanet
!pip install .
!python setup.py build_ext --inplace
%cd ..
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
import json
f = open('../input/face-mask-detection-dataset/Medical mask/Medical mask/meta.json')
classes_colors = json.load(f)
#print(len(classes_colors["classes"]))

z = []
for i in range(len(classes_colors["classes"])):
    z.append([classes_colors["classes"][i]["title"],i+1,classes_colors["classes"][i]["color"]])
#print(z)

from pandas import DataFrame
class_color_df = DataFrame(z,columns=['classname','id','color'])
class_map_df = class_color_df[['classname','id']]
class_map_df=pd.DataFrame.from_records(class_map_df.values)
class_map_df.to_csv(r'train_class.csv',index=False)

class_color_df.head()
df=pd.read_csv("../input/face-mask-detection-dataset/train.csv",header= None)
df[0] = "../input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images/"+df[0].astype(str)
df=df[1:]
df=df.drop([8132], axis=0)
df.to_csv(r'train_new.csv',index=False,header=None)
df.head(7)
!retinanet-debug csv train_new.csv train_class.csv
def show_objects(record_image):

    path = record_image[0]
    box = [record_image[1], record_image[2], record_image[3], record_image[4]]
    class_n = record_image[5]
    image = read_image_bgr(path)
    
    colours = ((class_color_df[class_color_df.classname == class_n].color).iloc[0]).lstrip('#')
    colours1 = tuple(int(colours[i:i+2], 16) for i in (0, 2, 4))

    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    draw_box(draw, box, color = colours1)
    plt.imshow(draw)
    plt.show()
show_objects(df.iloc[119])
df_train, df_test = train_test_split(df, test_size=0.15, shuffle=False)
df_train.to_csv(r'train_data.csv', index = False, header=None)
df_test.to_csv(r'test_data.csv', index = False, header=None)
PRETRAINED_MODEL = 'pretrained_model.h5'

URL_MODEL = 'https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5'
urllib.request.urlretrieve(URL_MODEL, PRETRAINED_MODEL)
ANNOTATIONS_FILE = 'train_data.csv'
CLASSES_FILE = 'train_class.csv'
!keras-retinanet/keras_retinanet/bin/train.py --freeze-backbone --random-transform --weights {PRETRAINED_MODEL} --batch-size 8 --steps 500 --epochs 10 csv train_data.csv train_class.csv
!ls snapshots
model_path = os.path.join('snapshots', sorted(os.listdir('snapshots'), reverse=True)[0])
print(model_path)

model = models.load_model(model_path, backbone_name='resnet50')
model = models.convert_model(model)

labels_to_names = pd.read_csv(CLASSES_FILE, header=None).T.loc[0].to_dict()
import pickle
file_name = 'model_raw.sav'
pickle.dump(model, open(file_name, 'wb'))
model.save_weights("model_weights.h5")
def predict(image):
    image = preprocess_image(image.copy())
    image, scale = resize_image(image)

    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

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

    plt.imshow(draw)
    plt.show()
df_test.columns = ['name', 'x1', 'x2', 'y1','y2','classname']
show_detected_objects(df_test.iloc[27])
show_detected_objects(df_test.iloc[49])
show_detected_objects(df_test.iloc[30])
show_detected_objects(df_test.iloc[39])
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
show_detected_objects_new(submit.iloc[12])
show_detected_objects_new(submit.iloc[34])
show_detected_objects_new(submit.iloc[13])
def predict(image):
    image = preprocess_image(image.copy())
    image, scale = resize_image(image)

    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

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
nt = final.to_csv(r'submit_1.csv')