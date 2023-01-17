# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from fastai import *
from fastai.vision import *
os.listdir("../input")
train_imgs=os.listdir("../input/train/train/")
train_imgs[:5]
import pathlib
path=pathlib.Path("../input/train/train/")
fnames=get_image_files(path)
fnames[:5]
def return_label(path):
    if str(path).find('dog')==-1:
        return "cat"
    else:
        return "dog"
return_label(pathlib.Path("../input/train/train/cat.1.jpg"))
data = ImageDataBunch.from_name_func(path, fnames, return_label, ds_tfms=get_transforms(), size=224,bs=8)
data.normalize(imagenet_stats)
data.train_ds.y.classes
TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"

learn=create_cnn(data,models.resnet34,metrics=[accuracy,error_rate],path=TMP_PATH,model_dir=MODEL_PATH)
learn.fit_one_cycle(1)
learn.save('model_224')
os.listdir("../../tmp/model/")
learn.lr_find()
learn.recorder.plot()
interp=ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
#dog=1 and cat=0
test_path=pathlib.Path("../input/test1/test1/")
test_images=get_image_files(test_path)
test_images[:5]
id_list=[]
labels=[]
import re
for img_path in test_images:
    #print(img)
    img_id=re.findall('\d+',str(img_path))
    id_list.append(int(img_id[2]))
    img=open_image(img_path)
    label=learn.predict(img)
    if str(label[0])=='cat':
        labels.append(0)
    else:
        labels.append(1)
    print(img_id,"  ",label)
len(labels)
len(id_list)
os.listdir("../input")
my_submission=pd.DataFrame({"id":id_list,"label":labels})
my_submission.to_csv('submission.csv', index=False)
os.listdir()
sub=pd.read_csv("submission.csv")
sub.head()
