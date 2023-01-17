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

!wget https://pjreddie.com/media/files/yolov3-tiny.weights
!git clone https://github.com/larsien/keras-yolo3
!ls keras-yolo3
import os 
os.chdir('keras-yolo3')
!ls 
!python convert.py yolov3-tiny.cfg ../yolov3-tiny.weights model_data/yolo_tiny.h5
!unzip /kaggle/input/2019-3rd-ml-month-with-kakr/train.zip -d /kaggle/working/data/train/
!ls ../input
#현재 위치는 ./kaggle/working
from PIL import Image

def get_resized_img(file, file_name='test.jpg', 
                    save_path='./', 
                    image_size=(320,320)):
    Image.open(file).resize(image_size).save(save_path+file_name)
    return Image.open(save_path+file_name)

test_img = get_resized_img('/kaggle/working/data/train/train_00036.jpg')
test_img
os.chdir('./keras-yolo3')
!ls ../
#에러 난다면 tensorflow 버전 확인 필요
from yolo import YOLO
def objectDetection(path, model_path, class_path):
    yolo = YOLO(model_path = model_path, classes_path = class_path, anchors_path = 'model_data/tiny_yolo_anchors.txt')
    result_image, label = yolo.detect_image(Image.open(path))
    display(result_image)
    return result_image
    
# os.chdir('./keras-yolo3')
test_result = objectDetection('../test.jpg', 'model_data/yolo_tiny.h5','model_data/coco_classes.txt')
test_result
# 환경마다 삭제해야 할 라이브러리가 조금 씩 다름. 에러나는 것들 지우고 텐서플로우 설치 
!pip uninstall tensorflow tensorflow-estimator tensorflow-probability keras fancyimpute catalyst --yes
!pip uninstall tensorflow-gpu tensorflow-tensorboard --yes 

!pip install tensorflow-gpu==1.13.1
!pip install keras==2.1.5

import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import PIL
from sklearn.model_selection import StratifiedKFold, KFold
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import multi_gpu_model
import keras 
import warnings 
import tensorflow as tf
warnings.filterwarnings('ignore')

print(K.image_data_format()) # 이미지 데이터 포맷 방식 확인. first, last 2가지. 이미지를 처음부터 읽을지 마지막부터 읽을지 
print(keras.__version__)
print(tf.__version__)
import gc
import os
import glob
import zipfile
import warnings
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import cv2
import PIL
from PIL import ImageOps, ImageFilter, ImageDraw
import pandas as pd
import os

DATA_PATH = os.getcwd()+'/data/'
TRAIN_IMG_PATH = os.path.join(DATA_PATH, 'train')
TEST_IMG_PATH = os.path.join(DATA_PATH, 'test')

# CSV 파일 경로
df_train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
df_test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
df_class = pd.read_csv(os.path.join(DATA_PATH, 'class.csv'))

df_train.head()

IMAGE_SIZE = 320

def crop_resize_boxing_img(img_name, margin=16, size=(288,288)):
    PATH = ''
    if 'train_' in img_name:
        PATH = TRAIN_IMG_PATH
        data = df_train
    elif 'test_' in img_name:
        PATH = TEST_IMG_PATH
        data = df_test
    else:
        print(f'wrong file input {img_name}')
        return
    img = PIL.Image.open(os.path.join(PATH, img_name))
#     print(data['img_file'] )
    pos = data.loc[data['img_file'] == img_name, ['bbox_x1','bbox_y1','bbox_x2','bbox_y2']].values.reshape(-1)
#     print(pos[:10])
    width, height =  img.size
    x1 = max(0, pos[0] - margin)
    y1 = max(0, pos[1] - margin)
    x2 = min(pos[2] + margin, width)
    y2 = min(pos[3] + margin, height)
    return img.crop((x1, y1, x2, y2)).resize(size)

TRAIN_CROP_PATH = './data/cropped_train'
TEST_CROP_PATH = './data/cropped_test'

if (os.path.isdir(TRAIN_CROP_PATH) == False):
    os.mkdir(TRAIN_CROP_PATH)
    for i, row in df_train.iterrows():
        cropped = crop_resize_boxing_img(row['img_file'])
        cropped.save(f"{TRAIN_CROP_PATH}/{row['img_file']}")

if (os.path.isdir(TEST_CROP_PATH) == False):
    os.mkdir(TEST_CROP_PATH)
    for i, row in df_test.iterrows():
        cropped = crop_resize_boxing_img(row['img_file'])
        cropped.save(f"{TEST_CROP_PATH}/{row['img_file']}")




display(PIL.Image.open(os.path.join(TRAIN_IMG_PATH, df_train.loc[0,'img_file'])))
display(PIL.Image.open(os.path.join(TRAIN_CROP_PATH,df_train.loc[0, 'img_file'])))
df_train.to_csv('sample.txt', index=False, header=None, sep=',')

with open('sample.txt', 'r') as f, open('./keras-yolo3/train.txt', 'w') as w:
    # train_00001.jpg,1,80,641,461,108
    lines = f.readlines()
    for line in lines:
        w.write(f"{line.split(',')[0].replace('train_',home+'/data/cropped_train/train_')} 0,0,320,320,{int(line.split(',')[-1]) -1}\n")

class_df = pd.read_csv('./data/class.csv')
class_df = class_df.loc[:, 'name']
class_df.to_frame().to_csv('./keras-yolo3/class.txt',index=None, header=False)
def is_history_exist(log_dir):
    history_files = []
    for file in os.listdir(log_dir):
        if '.h5' in file:
            history_files.append(file)
    
    if len(history_files) == 0:
        return False
    else : 
        return True
    
def get_minimum_loss_weight(log_dir):
    history_files = []
    for file in os.listdir(log_dir):
        if '.h5' in file:
            history_files.append(file)
    minimum_loss = float(9999999)
    minimum_loss_file = ''
    for file in history_files:
        loss = float(file.split('-')[1].replace('loss',''))
        if loss < minimum_loss:
            minimum_loss = loss
            minimum_loss_file = file
    print(f'Minimum loss file : {minimum_loss_file}, minimum loss : {minimum_loss}, ')
    epoch = int(minimum_loss_file.split('-')[0].replace('ep', ''))
    return os.path.join(log_dir,minimum_loss_file), epoch

log_dir = 'logs/000'
print(is_history_exist('./test'))
print(get_minimum_loss_weight(log_dir))
#현재 위치는 keras-yolo 
!python ./train.py
# print(os.getcwd())
import os
# os.chdir('./keras-yolo3')
from PIL import Image
from yolo import  YOLO 
img, i2 = objectDetection('../data/cropped_test/test_00002.jpg', 'logs/000/ep100-loss8.058-val_loss8.383.h5', 'model_data/voc_classes.txt')

def objects_detection(path, model_path, class_path):
    yolo = YOLO(model_path = model_path, classes_path = class_path, anchors_path = 'model_data/tiny_yolo_anchors.txt')
    result_image, score = yolo.detect_image(Image.open(path))
    print(score)
    display(result_image)
submission_df = df_test.loc[:,'img_file'].to_frame()
for i, img_path in enumerate(submission_df.loc[:,'img_file']):
    objectDetection(os.path.join(TEST_CROP_PATH, img_path), 'model_data/yolo_tiny.h5', 'class.txt')

submission_df['class'] = pred
submission_df.to_csv('suvmission.csv', index = False)
submission_df.head()


