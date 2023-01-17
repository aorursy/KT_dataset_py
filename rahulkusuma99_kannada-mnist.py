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
from fastai.vision import *

from fastai.tabular import *

import imageio
train = pd.read_csv('../input/Kannada-MNIST/train.csv')

train.head()

tfms = get_transforms(flip_vert=False, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
path='../input/Kannada-MNIST'

test = pd.read_csv('../input/Kannada-MNIST/test.csv')
def to_img_shape(data_X, data_y=[]):

    data_X = np.array(data_X).reshape(-1,28,28)

    data_X = np.stack((data_X,)*3, axis=-1)

    data_y = np.array(data_y)

    return data_X,data_y
data_X, data_y = train.loc[:,'pixel0':'pixel783'], train['label']



from sklearn.model_selection import train_test_split



train_X, val_X, train_y, val_y = train_test_split(data_X, data_y, test_size=0.1,random_state=42,stratify=data_y)



test_X = test.loc[:,'pixel0':'pixel783']
train_X,train_y = to_img_shape(train_X, train_y)

val_X,val_y = to_img_shape(val_X,val_y)

test_X, _ = to_img_shape(test_X)
def save_imgs(path:Path, data, labels):

    path.mkdir(parents=True,exist_ok=True)

    for label in np.unique(labels):

        (path/str(label)).mkdir(parents=True,exist_ok=True)

    for i in range(len(data)):

        if(len(labels)!=0):

            imageio.imsave( str( path/str(labels[i])/(str(i)+'.jpg') ), data[i] )

        else:

            imageio.imsave( str( path/(str(i)+'.jpg') ), data[i] )



save_imgs(Path('/data/train'),train_X,train_y)

save_imgs(Path('/data/valid'),val_X,val_y)

save_imgs(Path('/data/test'),test_X, [])
#np.random.seed(42)

src = (ImageList.from_folder('/data/')

       .split_by_folder()

       .label_from_folder()

       .add_test_folder()          

       .transform(tfms, size=64)   

       .databunch())
src.show_batch(rows=3, figsize=(12,9))
arch = models.resnet50
acc_02 = partial(accuracy_thresh, thresh=0.2)

f_score = partial(fbeta, thresh=0.2)

learn = cnn_learner(src, arch, metrics=accuracy)
learn.lr_find()
learn.recorder.plot()
lr=1e-02

learn.fit_one_cycle(3)


learn152 = cnn_learner(src, models.resnet152, metrics=accuracy)
learn152.lr_find()

learn152.recorder.plot()
lr = 1e-2

learn152.fit_one_cycle(4)
learn152.unfreeze()
preds50, _ = learn.get_preds(DatasetType.Test)



preds152, _ = learn152.get_preds(DatasetType.Test)




y = torch.argmax(preds50, dim=1)
preds = 0.50*preds50  + 0.50*preds152



y = torch.argmax(preds, dim=1)
num = len(learn.data.test_ds)

indexes = {}



for i in range(num):

    filename = str(learn.data.test_ds.items[i]).split('/')[-1]

    filename = filename[:-4] # get rid of .jpg

    indexes[(int)(filename)] = i
submission = pd.DataFrame({ 'id': range(0, num),'label': [y[indexes[x]].item() for x in range(0, num)] })

submission.to_csv(path_or_buf ="submission.csv", index=False)