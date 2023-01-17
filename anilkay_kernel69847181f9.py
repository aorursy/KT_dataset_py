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
import skimage.io as sio

img1=sio.imread("/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL/NORMAL2-IM-1349-0001.jpeg")

sio.imshow(img1)
img1=sio.imread("/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/PNEUMONIA/person1044_bacteria_2978.jpeg")

sio.imshow(img1)
path="/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/"

from fastai.vision import *

data = ImageDataBunch.from_folder(path=path, train="train", valid="val", test="test",ds_tfms=get_transforms(), size=224, bs=64)
learn3 = cnn_learner(data, 

             models.resnet152, 

                    metrics=accuracy)

learn3.fit(15)
interp = ClassificationInterpretation.from_learner(learn3, ds_type=DatasetType.Valid)

interp.plot_confusion_matrix()
interp_train = ClassificationInterpretation.from_learner(learn3, ds_type=DatasetType.Train)

interp_train.plot_confusion_matrix()
interp_train = ClassificationInterpretation.from_learner(learn3, ds_type=DatasetType.Test)

interp_train.plot_confusion_matrix()
data2 = ImageDataBunch.from_folder(path=path, train="train", valid="test", test="val",ds_tfms=get_transforms(), size=224, bs=64)

learn3.data=data2

learn3.validate(data.valid_dl)
interp_test = ClassificationInterpretation.from_learner(learn3, ds_type=DatasetType.Valid)

interp_test.plot_confusion_matrix()
interp_test.plot_multi_top_losses()
learn3.show_results()
learn3.summary()