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

from fastai import *

from fastai.vision import *

import numpy as np

import warnings

warnings.filterwarnings('ignore')
os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/')
path = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/'

# create a data bunch

np.random.seed(42)

data = ImageDataBunch.from_folder(path,train='train',valid='test',test='val',

                                  ds_tfms=get_transforms(do_flip=False),

                                  size=224,bs=64,

                                  num_workers=1).normalize(imagenet_stats)
data
data.classes
data.c
data.show_batch(rows=6,figsize=(12,8))
learn = create_cnn(data,models.resnet34,metrics=error_rate,model_dir="/tmp/model/")

learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5,1e-3,moms=(0.8,0.7))
learn.fit_one_cycle(4)
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

# plot the top losses

interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(5,5), dpi=60)
learn.save('stage-1')
learn.unfreeze()

learn.fit_one_cycle(2)
learn.lr_find()

learn.recorder.plot()
# we select the slice with the steepest slope

learn.unfreeze()

learn.fit_one_cycle(1, max_lr=slice(1e-5,1e-4))
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(figsize=(5,5), dpi=60)
preds,y =learn.get_preds(DatasetType.Test)
labels = preds.numpy()
labels.shape
data.test_ds
predictions = [i.argmax().item() for i in labels]
predictions
os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/val/NORMAL/')
images = []

labels = []

for image in os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/val/NORMAL/'):

    if image != '.DS_Store':

        images.append(image)

        labels.append(0)
for image in os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/val/PNEUMONIA/'):

     if image != '.DS_Store':

        images.append(image)

        labels.append(1)
import pandas as pd

results = pd.DataFrame({'Image_Name':images,'Actual_Label':labels})
results.shape
results['Predicted_Label'] = predictions
results
from sklearn.metrics import accuracy_score

accuracy_score(results['Predicted_Label'].values,results['Actual_Label'].values)
import PIL

PIL.Image.open('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/val/PNEUMONIA/person1951_bacteria_4882.jpeg')