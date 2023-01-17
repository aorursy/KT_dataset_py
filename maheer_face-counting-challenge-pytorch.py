import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np 
import pandas as pd 
from tqdm import tqdm
train=pd.read_csv('../input/facecounting/train_fc/train.csv')
box_train=pd.read_csv('../input/facecounting/train_fc/bbox_train.csv')
test=pd.read_csv('../input/facecounting/test_fc.csv')
img_dir="../input/facecounting/train_fc/image_data/"

train.head()
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

test1=[]
facedetector=cv2.CascadeClassifier("../input/haar-cascades-for-face-detection/haarcascade_frontalface_alt.xml")
for i in tqdm(range(test.shape[0])):
    z1 = '../input/facecounting/train_fc/image_data/'+ test['Name'][i]
    #print(z1)
    img = cv2.imread(z1 ,0)
    face=facedetector.detectMultiScale(img,
                                 scaleFactor=1.1, 
                                 minNeighbors=3, 
                                 minSize=(20, 20)
                                 )
    test1.append(len(face))
test1
import sys
sys.path.append('../input/retinafacetorch')
from retina import retinaface_model, detect_images
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = retinaface_model(model_path='../input/retinafacetorch/Resnet50_Final.pth',device=device)
my=cv2.imread("../input/facecounting/train_fc/image_data/10022.jpg")
my = cv2.cvtColor(my, cv2.COLOR_BGR2RGB)
bboxes = detect_images(imgs=[np.float32(my)], net=model, thresh=0.94, device=device, batch_run=False , rescale_factor = 1.1)
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
for i in tqdm(range(test.shape[0])):
    z1 = '../input/facecounting/train_fc/image_data/'+ test['Name'][i]
    #print(z1)
    img = cv2.imread(z1 ,0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxes = detect_images(imgs=[np.float32(img)], net=model, thresh=0.58, device=device, batch_run=False , rescale_factor = 1.3)
    x3.append(len(bboxes[0]))
submissions = pd.DataFrame({'Name':test['Name'],'HeadCount':x3})
submissions.to_csv("submission.csv",index = False)
