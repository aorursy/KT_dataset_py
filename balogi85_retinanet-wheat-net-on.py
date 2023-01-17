import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_path = '/kaggle/input/global-wheat-detection/train.csv'
path = '/kaggle/input/global-wheat-detection/train/'
test_path = '/kaggle/input/global-wheat-detection/test/'
temp_path = '/kaggle/working/'
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import urllib
import cv2
from matplotlib import pyplot as plt

%matplotlib inline

data_table = pd.read_csv(train_path)
data_table.head()
annotations = pd.DataFrame(columns=['id_path', 'x1','y1','w','h','class'])
annotations.head()
a = 0
for id in data_table['image_id']:
  jpg_path = os.path.join(path, id + '.jpg')
  data_table.iat[a,0]=jpg_path
  a+=1

annotations['id_path']=data_table['image_id']
annotations.head()
annotations['id_path'][0]
a = 0
for i in data_table['bbox']:
  item = i.split(',')
  annotations.iat[a,1] = item[0]
  annotations.iat[a,2] = item[1]
  annotations.iat[a,3] = item[2]
  annotations.iat[a,4] = item[3]
  a+=1
annotations.head() 
annotations['x1'].replace({'\[':''}, regex=True, inplace=True)
annotations['h'].replace({']':''}, regex=True, inplace=True)
annotations['class']='wheat'
annotations['x1']=pd.to_numeric(annotations['x1'])
annotations['y1']=pd.to_numeric(annotations['y1'])
annotations['w']=pd.to_numeric(annotations['w'])
annotations['h']=pd.to_numeric(annotations['h'])
annotations['x1']=annotations['x1'].astype(int)
annotations['y1']=annotations['y1'].astype(int)
annotations['w']=annotations['w'].astype(int)
annotations['h']=annotations['h'].astype(int)
for i in range(len(annotations['w'])):
    annotations['w'][i]=annotations['x1'][i]+annotations['w'][i]
for i in range(len(annotations['h'])):
    annotations['h'][i]=annotations['y1'][i]+annotations['h'][i]
annotations.head()
annotations.info()
annotations.to_csv(temp_path + 'annot.csv', header=False, index=False)
class_df = pd.DataFrame(columns=['class','class_id'],index=[0])
class_df['class']='wheat'
class_df['class_id']=0
class_df
class_df.to_csv(temp_path + 'class.csv', header=False, index=False)
im_path = annotations['id_path'][0]
im_path
x1 = annotations['x1'][1]
w = annotations['w'][1]
y1 = annotations['y1'][1]
h = annotations['h'][1]
im= cv2.imread(im_path)
im = cv2.rectangle(im, (x1, y1), (w, h), (255,0,0), 2)
plt.imshow(im)
plt.show()
!git clone https://github.com/fizyr/keras-retinanet.git
%cd keras-retinanet/
!pip install .
!python setup.py build_ext --inplace
PRETRAINED_MODEL = './snapshots/_pretrained_model.h5'
PRETRAINED_MODEL1 = '/kaggle/working/keras-retinanet/snapshots/resnet50_csv_06.h5'

URL_MODEL = 'https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5'
urllib.request.urlretrieve(URL_MODEL, PRETRAINED_MODEL)

print('Downloaded pretrained model to ' + PRETRAINED_MODEL)
EPOCHS = 6
BATCH_SIZE=8
STEPS = 100 
LR=0.0001
!keras_retinanet/bin/train.py --random-transform --weights {PRETRAINED_MODEL1} --lr {LR} --batch-size {BATCH_SIZE} --steps {STEPS} --epochs {EPOCHS} --no-resize csv /kaggle/working/annot.csv /kaggle/working/class.csv
test_images = os.listdir(test_path)
import tensorflow as tf
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
!ls snapshots
removing_path = '/kaggle/working/keras-retinanet/snapshots/'
os.remove(removing_path+'resnet50_csv_01.h5')
os.remove(removing_path+'resnet50_csv_06.h5')
os.remove(removing_path+'resnet50_csv_03.h5')
os.remove(removing_path+'resnet50_csv_04.h5')
os.remove(removing_path+'resnet50_csv_05.h5')
model_path = os.path.join('snapshots', 'resnet50_csv_02.h5')

model = models.load_model(model_path, backbone_name='resnet50')
model = models.convert_model(model)
def prediction(image):
    image = preprocess_image(image.copy())
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    return boxes, scores, labels
thres = 0.5
def draw_detections(image, boxes, scores, labels):
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < thres:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(image, b, color=color)

        caption = "{:.3f}".format(score)
        draw_caption(image, b, caption)
def show_detected_objects(image_name):
    img_path = test_path+'/'+image_name
  
    image = read_image_bgr(img_path)

    boxes, scores, labels = prediction(image)
    print(boxes[0,0].shape)
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    draw_detections(draw, boxes, scores, labels)
    plt.figure(figsize=(15,10))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()
for img in test_images:
  
    show_detected_objects(img)
preds=[]
imgid=[]
for img in test_images:
    img_path = test_path+'/'+img
    image = read_image_bgr(img_path)
    boxes, scores, labels = prediction(image)
    boxes=boxes[0]
    scores=scores[0]
    for idx in range(boxes.shape[0]):
        if scores[idx]>thres:
            box,score=boxes[idx],scores[idx]
            imgid.append(img.split(".")[0])
            preds.append("{} {} {} {} {}".format(score, int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1])))
    
preds[0]
sub={"image_id":imgid, "PredictionString":preds}
sub=pd.DataFrame(sub)
sub.head()
sub_=sub.groupby(["image_id"])['PredictionString'].apply(lambda x: ' '.join(x)).reset_index()
sub_
submiss=pd.read_csv("/kaggle/input/global-wheat-detection/sample_submission.csv")
submiss.head()
for idx,imgid in enumerate(submiss['image_id']):
    submiss.iloc[idx,1]=sub_[sub_['image_id']==imgid].values[0,1]
    
submiss.head()
submiss.to_csv('/kaggle/working/submission.csv',index=False)
annot = '/kaggle/working/annot.csv'
class_cs = '/kaggle/working/class.csv'
retina = '/kaggle/working/keras-retinanet'

os.remove(annot)
os.remove(class_cs)
import shutil
shutil.rmtree(retina)
