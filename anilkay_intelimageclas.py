# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

i=0

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        i=i+1

        if i>5:

            break



# Any results you write to the current directory are saved as output.
import cv2

import skimage.io as sio
image=cv2.imread("/kaggle/input/intel-image-classification/seg_train/seg_train/buildings/11054.jpg")

sio.imshow(image)
image=cv2.imread("/kaggle/input/intel-image-classification/seg_test/seg_test/forest/23876.jpg")

sio.imshow(image)
image=cv2.imread("/kaggle/input/intel-image-classification/seg_train/seg_train/street/11424.jpg")

sio.imshow(image)
from fastai.vision import *
!mkdir ./images/

!cp -a ../input/intel-image-classification/seg_train/seg_train ./images/train

!cp -a ../input//intel-image-classification/seg_test/seg_test ./images/valid

!cp -a ../input/intel-image-classification/seg_pred/seg_pred ./images/test

data3=ImageDataBunch.from_folder(path="/kaggle/input/intel-image-classification/seg_train/seg_train/",train="/kaggle/input/intel-image-classification/seg_train/seg_train/",test="/kaggle/input/intel-image-classification/seg_test/seg_test/",

                                 valid_pct=0.2,size=224)

data = ImageDataBunch.from_folder(path="./images/", train="train", valid="valid", test="test", ds_tfms=get_transforms(), size=224, bs=64)

learn3 = cnn_learner(data, 

             models.resnet152, 

                    metrics=accuracy)

learn3.fit(20)
interp = ClassificationInterpretation.from_learner(learn3, ds_type=DatasetType.Valid)

interp.plot_confusion_matrix()
interp.plot_multi_top_losses()
interp3 = ClassificationInterpretation.from_learner(learn3, ds_type=DatasetType.Test)

learn3.validate()
for liste in interp3.preds:

    print(liste.argmax())
list(interp3.pred_class)
import matplotlib.pyplot as plt

predictions, targets = learn3.get_preds(ds_type=DatasetType.Test)

classes = predictions.argmax(1)

class_dict = dict(enumerate(learn3.data.classes))

labels = [class_dict[i] for i in list(classes[:9].tolist())]

test_images = [i.name for i in learn3.data.test_ds.items][:9]

plt.figure(figsize=(10,8))



for i, fn in enumerate(test_images):

    img = plt.imread("./images/test/" + fn, 0)

    plt.subplot(3, 3, i+1)

    plt.imshow(img)

    plt.title(labels[i])

    plt.axis("off")
test_images = [i.name for i in learn3.data.test_ds.items][9:18]

plt.figure(figsize=(10,8))



for i, fn in enumerate(test_images):

    img = plt.imread("./images/test/" + fn, 0)

    plt.subplot(3, 3, i+1)

    plt.imshow(img)

    plt.title(labels[i])

    plt.axis("off")
test_images = [i.name for i in learn3.data.test_ds.items][27:36]

plt.figure(figsize=(10,8))



for i, fn in enumerate(test_images):

    img = plt.imread("./images/test/" + fn, 0)

    plt.subplot(3, 3, i+1)

    plt.imshow(img)

    plt.title(labels[i])

    plt.axis("off")
!rm -rf ./images/