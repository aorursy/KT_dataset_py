#Need to get working with your pretrained weights though

#And also need to make sure it works without internet (how can I do that with the torch version?)
!pip install ../input/torchoffline/torch-1.6.0cu101-cp37-cp37m-linux_x86_64.whl

!pip install ../input/torchoffline/torchvision-0.7.0cu101-cp37-cp37m-linux_x86_64.whl
import torch

import os

import glob

import pandas as pd

import sys

import numpy as np

from PIL import Image

#1: Run prediction of testimages using pretrained weights



yolo_dir = '/kaggle/input/yolosource-2/yolov5_to_zip'

result_dir = '/kaggle/working/inference/output'

submission_dir = '/kaggle/working'

print(torch.__version__)
#Latest version

!mkdir utils

!cp -r ../input/latestyolo/yolov5-master/utils/* ./utils/

!mkdir models

!cp -r ../input/latestyolo/yolov5-master/models/* ./models/

!mkdir weights

!cp -r ../input/latestyolo/yolov5-master/weights/* ./weights/

!cp -r ../input/latestyolo/yolov5-master/detect.py ./

!cp ../input/best-weights/full_best.pt ./weights/
# !rm detect.py

# !rm -r utils

# !rm -r models

# !rm -r weights
# %run detect.py --weights yolov5x.pt --img 1024 --conf 0.4 --source /kaggle/input/global-wheat-detection/test --save-txt --device '0'
!mkdir test_resized

width = 1024

height = 1024

for filename in glob.glob('../input/global-wheat-detection/test/*.jpg'):

    prefix = filename.split("/")[-1].split(".")[0]

    img = Image.open(filename)

    new_img = img.resize((width,height))

    new_img.save('./test_resized/'+prefix + '.jpg')
%run detect.py --weights weights/full_best.pt --img 1024 --conf 0.5 --source ./test_resized --save-txt --device '0'


# image_id,PredictionString

# ce4833752,1.0 0 0 50 50

# adcfa13da,1.0 0 0 50 50

# 6ca7b2650,

# 1da9078c1,0.3 0 0 50 50 0.5 10 10 30 30

# 7640b4963,0.5 0 0 50 50

#Have to be careful about what width and height should be depending on the size of data you test on

# id,conf x y w h conf2 x2 y2 w2 h2. id is column under image_id, rest is column under PredictionString

#Open all txt results

width = 1024

height = 1024

results = []

for filename in glob.glob(os.path.join(result_dir,'*.txt')):

    prefix = filename.split("/")[-1].split(".")[0]

    predict_str = ""

    with open(filename, 'r') as fp:

        for cnt, line in enumerate(fp):

            entries = np.array(line.strip('\n').split()).astype(np.float)

            bbox = np.round([(entries[1]-entries[3]/2.0)*width,(entries[2]-entries[4]/2.0)*height,entries[3]*width,entries[4]*height],3).tolist()

            conf = entries[-1]

            predict_str = predict_str + str(conf) + " " + " ".join(map(str,bbox)) + " " 

            

            

        results.append({'image_id':prefix, 'PredictionString':predict_str.rstrip()}) 

resultFrame = pd.DataFrame(results)
temp = []

for filename in glob.glob('../input/global-wheat-detection/test/*.jpg'):

    

    prefix = filename.split("/")[-1].split(".")[0]

    if resultFrame['image_id'].str.contains(prefix).any():

        pass

    else:

        temp.append({'image_id':prefix, 'PredictionString':""})

        

df = pd.DataFrame(temp)

resultFrame = resultFrame.append(df)

        

   

   

    
resultFrame.to_csv(submission_dir + "/submission.csv", index = False)

resultFrame.head()