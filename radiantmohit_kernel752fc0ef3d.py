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
import os
print(os.getcwd())
import pandas as pd
df = pd.read_csv('train.csv')
df
import ast
df.bbox = df.bbox.apply(ast.literal_eval)
df
df = df.groupby('image_id')['bbox'].apply(list).reset_index()
DATA_PATH = '/kaggle/input/global-wheat-detection/'
df
from sklearn import model_selection

df_train , df_valid = model_selection.train_test_split(
df,test_size = 0.1 ,random_state = 42 ,shuffle = True)

df_train  = df_train.reset_index(drop = True)
df_valid  = df_valid.reset_index(drop = True)

df_train
os.mkdir('/kaagle/wheat_data')
os.mkdir('/kaagle/wheat_data/images')
os.mkdir('/kaagle/wheat_data/labels')
os.mkdir('/kaagle/wheat_data/images/train')
os.mkdir('/kaagle/wheat_data/images/validation')
os.mkdir('/kaagle/wheat_data/labels/train')
os.mkdir('/kaagle/wheat_data/labels/validation')
OUTPUT_PATH = '/kaagle/wheat_data'
from tqdm import tqdm
import shutil
import numpy as np
def process_data(data,data_type = 'train'):
    for _, row in tqdm(data.iterrows() , total = len(data)):
        image_name = row['image_id']
        bounding_boxes = row['bbox']
        yolo_data = []
        for bobox in bounding_boxes:
            x = bobox[0]
            y = bobox[1]
            w = bobox[2]
            h = bobox[3]
            x_center = x + w / 2
            y_center = y + h / 2
            x_center /= 1024
            y_center /= 1024
            w /= 1024
            h /= 1024
            yolo_data.append([0 , x_center , y_center , w, h  ])
        yolo_data = np.array(yolo_data)
        np.savetxt(os.path.join(OUTPUT_PATH , f"labels/{data_type}/{image_name}.txt"),
                   yolo_data,
                  fmt=["%d" , "%f", "%f" ,"%f" , "%f"])
        shutil.copyfile(os.path.join(DATA_PATH, f"train/{image_name}.jpg")
                       ,os.path.join(OUTPUT_PATH, f"images/{data_type}/{image_name}.jpg"))
        
            
            
            
    
df_train
process_data(df_valid ,data_type = 'validation')
process_data(df_train ,data_type = 'train')
import os
os.chdir('/kaggle/input/yolov5/yolov5')
!python train.py --img 1024 --batch 10 --epochs 100 --data /kaggle/input/wheat123/wheat.yaml --cfg /kaggle/input/yolov5/yolov5/models/yolov5s.yaml --name wm