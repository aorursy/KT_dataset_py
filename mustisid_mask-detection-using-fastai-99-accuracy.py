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
%reload_ext autoreload
%autoreload 2
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
from fastai import *
from fastai.vision import *
path = Path('../input/withwithout-mask/maskdata/maskdata')
path.ls()
data = (ImageList.from_folder(path)
       .split_by_folder(train='train',valid='test')
       .label_from_folder()
       .transform(get_transforms(),size=224)
       .databunch(bs=10))
data.classes
data.show_batch()
learn = cnn_learner(data,models.resnet34,metrics=[accuracy])
learn.fit_one_cycle(4)
m=learn.model.eval()
idx=0
x,y = data.train_ds[idx]
x.show()
data.valid_ds.y[idx]
xb,_ = data.one_item(x)
xb_im = Image(xb.view(3,224,224))
xb = xb.cuda()
xb_im.shape
from fastai.callbacks.hooks import *
def hooked_backward(cat=y):
    with hook_output(m[0]) as hook_a: 
        with hook_output(m[0], grad=True) as hook_g:
            preds = m(xb)
            preds[0,int(cat)].backward()
    return hook_a,hook_g
hook_a,hook_g = hooked_backward()
acts  = hook_a.stored[0].cpu()
acts.shape
avg_acts = acts.mean(0)
avg_acts.shape
def show_heatmap(hm):
    _,ax = plt.subplots()
    xb_im.show(ax)
    ax.imshow(hm, alpha=0.6, extent=(0,224,224,0),
              interpolation='bilinear', cmap='magma');
show_heatmap(avg_acts)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
learn.model_dir='/kaggle/working'
img=open_image('../input/withwithout-mask/masks2.0/masks/test/0/1.jpg')
learn.predict(img)
import cv2
import torch
torch.__version__

color_dict ={0:(0,255,0),1:(255,0,0)}
labels_dict=['With Mask','Without Mask']
face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
source=cv2.VideoCapture(0)
# learn =load_learner('','export.pkl')
while True:
    ret,img=source.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_clsfr.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:

        face_img=gray[y:y+w,x:x+w]
        # resized=cv2.resize(face_img,(100,100))
        # normalized=resized/255.0
        # reshaped=np.reshape(normalized,(1,100,100,1))
        # result=model.predict(reshaped)
        img_t = pil2tensor(face_img, np.float32)
        image = Image(img_t)
        label = learn.predict(image)[0]
        # label=np.argmax(result,axis=1)[0]

        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)


    cv2.imshow('LIVE',img)
    key=cv2.waitKey(1)

    if(key=='q'):
        break

cv2.destroyAllWindows()
source.release()
learn.export(Path('/kaggle/working/export.pkl'))
