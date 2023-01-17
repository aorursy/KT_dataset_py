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
os.listdir('/kaggle/input/histopathologic-cancer-detection')
os.listdir('/kaggle/input/histopathologic-cancer-detection/train')[0]
import pandas as pd
import numpy as np
import seaborn as sns
from fastai import *
from fastai.vision import *
labels = pd.read_csv('/kaggle/input/histopathologic-cancer-detection/train_labels.csv')
labels.head()
sns.countplot(x='label',data=labels)
plt.show()
path=Path('/kaggle/input/histopathologic-cancer-detection/')
tfms = get_transforms(do_flip=True,flip_vert=True, max_warp=0, max_rotate=10, max_lighting=0.05)
path
data = ImageDataBunch.from_csv(path, 
                             csv_labels='train_labels.csv', 
                             folder='train', 
                             suffix='.tif',
                             num_workers=2,
                             ds_tfms=tfms,
                             bs=64,
                             size=72,
                             test=path/'test').normalize(imagenet_stats)
data.classes
data.c
data
data.show_batch(rows=3,figsize=(8,10))
import warnings
warnings.filterwarnings('ignore')
from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *
learn= create_cnn(data, models.resnet34, metrics=[accuracy],model_dir='/kaggle/working/')
learn.lr_find()
learn.recorder.plot()
#wd = weight decay
learn.fit_one_cycle(6, max_lr=(1e-4, 1e-3, 1e-2), wd=(1e-4, 1e-4, 1e-1))
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(15, slice(1e-5,1e-4,1e-3))
learn.save('stage-1') 
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(10,8))
preds,y = learn.get_preds(DatasetType.Valid)
accuracy(preds,y)
label,y=learn.get_preds(DatasetType.Test)
sub = pd.read_csv(f'{path}/sample_submission.csv').set_index('id')
sub.head()
data.test_ds.items[0]
names=np.vectorize(lambda img_name: str(img_name).split('/')[-1][:-4]) 

file_names= names(data.test_ds.items).astype(str)
label.numpy()[:,1]
sub.loc[file_names,'label']=label.numpy()[:,1]
sub.to_csv(f'submission_resnet34.csv')
sub.head()
predictions = []
for i in label:
    predictions.append(i.argmax().item())
sns.countplot(predictions)
learn.show_results()
_,axs = plt.subplots(3,5,figsize=(11,8))
for i,ax in enumerate(axs.flatten()): 
  img = data.test_ds[i][0]
  img.show(ax=ax,y=learn.predict(img)[0])