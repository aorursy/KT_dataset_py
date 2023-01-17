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
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
x1 = pd.read_csv('../input/test-face-detection-challenge-analytic-vidhya/test_Rj9YEaI.csv')
print(x1)
x2=[]
facedetector=cv2.CascadeClassifier("../input/haar-cascades-for-face-detection/haarcascade_frontalface_alt.xml")
for i in tqdm(range(x1.shape[0])):
    z1 = '../input/face-counting-challenge-analytics-vidhya/image_data/'+ x1['Name'][i]
    #print(z1)
    img = cv2.imread(z1 ,0)
    face=facedetector.detectMultiScale(img,
                                 scaleFactor=1.1, 
                                 minNeighbors=3, 
                                 minSize=(20, 20)
                                 )
    x2.append(len(face))
    
print(x2)
submissions = pd.DataFrame({'Name':x1['Name'],'HeadCount':x2})
submissions.to_csv("/kaggle/working/submission_1.csv",index = False)

# my=cv2.imread("../input/face-counting-challenge-analytics-vidhya/image_data/10022.jpg")
# my2=cv2.imread("../input/face-counting-challenge-analytics-vidhya/image_data/10022.jpg",0)
# facedetector=cv2.CascadeClassifier("../input/haar-cascades-for-face-detection/haarcascade_frontalface_alt.xml")
# face=facedetector.detectMultiScale(my2,
#                                  scaleFactor=1.1, 
#                                  minNeighbors=3, 
#                                  minSize=(20, 20)
#                                  )
# print('number of faces:')
# print(len(face))
# fig,ax = plt.subplots(1)
# im = cv2.cvtColor(my, cv2.COLOR_BGR2RGB)
# ax.imshow(im)
# for x,y,z,h in face:
#     rect = patches.Rectangle((x,y),z,h,linewidth=1,edgecolor='r',facecolor='none')
#     print(rect)
#     ax.add_patch(rect)
# plt.show()

%matplotlib inline
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

import os
import json
import time
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import cv2
import sys
sys.path.append('/kaggle/input/retinafacetorch')
from retina import retinaface_model, detect_images
device = 'cuda' if torch.cuda.is_available() else 'cpu'
retinaface_model = retinaface_model(model_path='../input/retinafacetorch/Resnet50_Final.pth',device=device)
import pandas as pd
x1 = pd.read_csv('../input/test-face-detection-challenge-analytic-vidhya/test_Rj9YEaI.csv')
my=cv2.imread("../input/face-counting-challenge-analytics-vidhya/image_data/10022.jpg")
my = cv2.cvtColor(my, cv2.COLOR_BGR2RGB)
bboxes = detect_images(imgs=[np.float32(my)], net=retinaface_model, thresh=0.94, device=device, batch_run=False , rescale_factor = 1.1)
print(bboxes)
print("Total no. of faces: ",len(bboxes[0]))
fig,ax = plt.subplots(1)
im = my
ax.imshow(im)
for b in bboxes[0]:
    rect = patches.Rectangle((b[0],b[1]),(b[2]-b[0]),(b[3]-b[1]),linewidth=1,edgecolor='r',facecolor='none')
    print(rect)
    ax.add_patch(rect)
plt.show()
x3 =[]
for i in tqdm(range(x1.shape[0])):
    z1 = '../input/face-counting-challenge-analytics-vidhya/image_data/'+ x1['Name'][i]
    #print(z1)
    img = cv2.imread(z1 ,0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxes = detect_images(imgs=[np.float32(img)], net=retinaface_model, thresh=0.58, device=device, batch_run=False , rescale_factor = 1.3)
    x3.append(len(bboxes[0]))
# bboxes = detect_images(imgs=[np.float32(img) for img in imgs], net=retinaface_model, thresh=0.94, device=device)
submissions = pd.DataFrame({'Name':x1['Name'],'HeadCount':x3})
submissions.to_csv("/kaggle/working/retinaface_2_thres0.58_scale1.3.csv",index = False)
fig,ax = plt.subplots(1)
im = cv2.imread("../input/face-counting-challenge-analytics-vidhya/image_data/12744.jpg")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
bboxes = detect_images(imgs=[np.float32(im)], net=retinaface_model, thresh=0.58, device=device, batch_run=False , rescale_factor = 1.3)
ax.imshow(im)
for b in bboxes[0]:
    rect = patches.Rectangle((b[0],b[1]),(b[2]-b[0]),(b[3]-b[1]),linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
plt.show()
print("Total no. of faces: ",len(bboxes[0]))
train = pd.read_csv("../input/face-counting-challenge-analytics-vidhya/train.csv")
