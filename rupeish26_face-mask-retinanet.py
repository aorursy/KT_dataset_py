!git clone https://github.com/fizyr/keras-retinanet.git
%cd keras-retinanet/
!pip install .
!python setup.py build_ext --inplace
%cd ..
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import urllib
%matplotlib inline
%config InlineBackend.figure_format='retina'

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
images=os.path.join("/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images")
annotations=os.path.join("/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/annotations")
train=pd.read_csv(os.path.join("/kaggle/input/face-mask-detection-dataset/train.csv"),header=None)
submission=pd.read_csv(os.path.join("/kaggle/input/face-mask-detection-dataset/submission.csv"))
submission_file=pd.DataFrame(columns=['name','x1' , 'x2', 'y1', 'y2','label'])
train = train.iloc[1:]
print(len(train))
print(train.head())

print(len(submission))
submission.head()
len(os.listdir(images))
a=os.listdir(images)
b=os.listdir(annotations)
a.sort()
b.sort()
print(len(b),len(a))
train_images=a[1698:]
test_images=a[:1698]
test_images[0]
type(test_images)
train_images[0]
img=plt.imread(os.path.join(images,test_images[0]))
plt.imshow(img)
plt.show()
img=plt.imread(os.path.join(images,train_images[1]))
plt.imshow(img)
plt.show()
train.head()
len(train)
options=['face_with_mask','face_no_mask']
train= train[train[5].isin(options)]
train.sort_values(0,axis=0,inplace=True)
train.head()
len(train)
def show_image_objects(image_row):

    img_path = image_row[0]
    box = [
    image_row[1], image_row[2], image_row[3], image_row[4]
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
train.head()
train[0][13382]
img=plt.imread(os.path.join(images,train[0][13382]))
plt.imshow(img)
plt.show()
show_image_objects(train.iloc[0])
train.head()
train[0] = '../input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images/' + train[0].astype(str)
train.head()
train_df, test_df = train_test_split(
  train, 
  test_size=0.15, 
  shuffle=False
)
train_df.head()
len(train_df)
PRETRAINED_MODEL = 'pretrained_model.h5'

URL_MODEL = 'https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5'
urllib.request.urlretrieve(URL_MODEL, PRETRAINED_MODEL)

print('Downloaded pretrained model to ' + PRETRAINED_MODEL)
train_df.to_csv(r'train_annot.csv', index = False, header=None)
test_df.to_csv(r'test_annot.csv', index = False, header=None)
data = [['face_with_mask',0],['face_no_mask',1]]
df = pd.DataFrame(data)
df.head()
df.to_csv(r'clas.csv', index = False, header=None)
ANNOTATIONS_FILE = 'train_annot.csv'
CLASSES_FILE = 'clas.csv'
!ls
!keras-retinanet/keras_retinanet/bin/train.py --freeze-backbone --random-transform --weights {PRETRAINED_MODEL} --batch-size 8 --steps 500 --epochs 10 csv train_annot.csv clas.csv
!ls snapshots
import pickle
filename = 'model_raw.sav'
pickle.dump(model, open(filename, 'wb'))
model_path = os.path.join('snapshots', sorted(os.listdir('snapshots'), reverse=True)[0])
print(model_path)
model = models.load_model(model_path, backbone_name='resnet50')
model = models.convert_model(model)
model.save_weights("model.h5")
labels_to_names = pd.read_csv(CLASSES_FILE, header=None).T.loc[0].to_dict()
labels_to_names
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
show_detected_objects(test_df.iloc[430])
test_df[test_df['classname']=='face_with_mask']
submission.head()
len(submission)
submit = submission.drop_duplicates()
submit.head()
len(submit)
THRES_SCORE = 0.5

def draw_detections(image, boxes, scores, labels):
    boxes_list=[]
    labels_list=[]
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < THRES_SCORE:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(image, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(image, b, caption)
        print(box,score,caption)
        boxes_list.append(box)
        labels_list.append(labels_to_names[label])
    return boxes_list,labels_list
def show_detected_objects_fin(image_row):  
  img_path = image_row["name"]
  img_dir='../input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images'
  img_path=os.path.join(img_dir, img_path)

  image = read_image_bgr(img_path)

  boxes, scores, labels = predict(image)

  draw = image.copy()
  draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

  draw_detections(draw, boxes, scores, labels) 

  dimension, classify = draw_detections(draw, boxes, scores, labels)

  temp = pd.DataFrame(dimension,columns = ['x1' , 'x2', 'y1', 'y2'])
  temp['label'] = classify
  temp['name'] = image_row["name"]
  temp = temp[['name','x1' , 'x2', 'y1', 'y2','label']]
  return temp
for i in range(0,len(submit)):
    detected=show_detected_objects_fin(submit.iloc[i])
    submission_file = submission_file.append(detected,ignore_index=True)
submission_file.head()
submission_file.to_csv('submit_this.csv')
