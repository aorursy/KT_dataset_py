import os

from fastai.vision import *

from fastai import *

import matplotlib.pyplot as plt

import seaborn as sns

from functools import partial

from tqdm.notebook import tqdm

import gc

from pylab import imread,subplot,imshow,show

%matplotlib inline
path = "/kaggle/input/kermany2018/OCT2017 /"
data

size = 224

bs=6
size = 224

bs = 64

data = ImageDataBunch.from_folder(path, 

                                  ds_tfms=get_transforms(max_rotate=0.1,max_lighting=0.15),

                                  valid_pct=0.2, 

                                  size=size, 

                                  bs=bs)
data.show_batch(rows=4)
fb = FBeta()

fb.average='macro'
learner = cnn_learner(data, models.resnet18, metrics=[accuracy])
learner.fit_one_cycle(1,1e-3)
learner.save('model_retina2')
learner.model_dir='/kaggle/working/'
learner.export('/kaggle/working/export2.pkl')
l=load_learner('/kaggle/working/')
learner=load_learner('/kaggle/input/modelx/')
learner
learner.load('/kaggle/input/modelx/model_retina')
interp= ClassificationInterpretation.from_learner(learner)
preds,y,losses = learner.get_preds(with_loss=True)
interp.plot_confusion_matrix()
import cv2

import numpy as np

img = cv2.imread('/kaggle/input/kermany2018/OCT2017 /val/DME/DME-9721607-2.jpeg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img2 = np.zeros_like(img)

img2[:,:,0] = gray

img2[:,:,1] = gray

img2[:,:,2] = gray

path2='/kaggle/input/kermany2018/OCT2017 /test/'
img=pil2tensor(img2,dtype=np.float32)
cl=['CNV','DME','DRUSEN','NORMAL']
for i in range(len(cl)):

    for file in os.listdir(path2+str(cl[i])):

        path3='/kaggle/input/kermany2018/OCT2017 /test/'+str(cl[i])+"/"+str(file)

        

        img=cv2.imread(str(path3))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img2 = np.zeros_like(img)

        img2[:,:,0] = gray

        img2[:,:,1] = gray

        img2[:,:,2] = gray

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img2 = np.zeros_like(img)

        img2[:,:,0] = gray

        img2[:,:,1] = gray

        img2[:,:,2] = gray

        print(cl[i])

        img=pil2tensor(img2,dtype=np.float32)

        print(learner.predict(Image(img)))

        



        

    

    
pre=[]

for i in preds:

    a=max(i)

    pre.append(a)

pre = np.array(pre, dtype=np.float32)
path2='/kaggle/input/kermany2018/OCT2017 /val/'

img=cv2.imread('/kaggle/input/kermany2018/OCT2017 /val/CNV/CNV-6294785-1.jpeg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img2 = np.zeros_like(img)

img2[:,:,0] = gray

img2[:,:,1] = gray

img2[:,:,2] = gray

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img2 = np.zeros_like(img)

img2[:,:,0] = gray

img2[:,:,1] = gray

img2[:,:,2] = gray



img=pil2tensor(img2,dtype=np.float32)

        

print(learner.predict(Image(img)))