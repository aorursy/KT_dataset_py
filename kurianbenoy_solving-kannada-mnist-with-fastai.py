import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *

import imageio

import numpy as np 

import pandas as pd 

import seaborn as sns
path = Path('../input/Kannada-MNIST')

train = pd.read_csv('../input/Kannada-MNIST/train.csv')

test  =pd.read_csv('../input/Kannada-MNIST/test.csv')
train.head(5)
train.describe()
y=train.label.value_counts()

sns.barplot(y.index,y)

def to_img_shape(data_X, data_y=[]):

    data_X = np.array(data_X).reshape(-1,28,28)

    data_X = np.stack((data_X,)*3, axis=-1)

    data_y = np.array(data_y)

    return data_X,data_y
data_X, data_y = train.loc[:,'pixel0':'pixel783'], train['label']

from sklearn.model_selection import train_test_split



train_X, val_X, train_y, val_y = train_test_split(data_X, data_y, test_size=0.10,random_state=7,stratify=data_y)
train_X,train_y = to_img_shape(train_X, train_y)

val_X,val_y = to_img_shape(val_X,val_y)
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
tfms = get_transforms(do_flip=False )
data = (ImageList.from_folder('/data/') 

        .split_by_folder()          

        .label_from_folder()        

        .add_test_folder()          

        .transform(tfms, size=64)   

        .databunch())
data
data.show_batch(3,figsize=(6,6))
# Copying pretrained models from fastai-pretrained models in data to adequate folder

!mkdir -p /tmp/.cache/torch/checkpoints

!cp /kaggle/input/fastai-pretrained-models/resnet101-5d3b4d8f.pth /tmp/.cache/torch/checkpoints/resnet101-5d3b4d8f.pth

learn = cnn_learner(data, models.resnet101, metrics=[error_rate, accuracy], model_dir = Path('../kaggle/working'),path = Path("."))
learn.fit_one_cycle(3)
learn.lr_find()
learn.recorder.plot()
lr = slice(2e-05)
learn.save('stage-1')
learn.unfreeze()
learn.fit_one_cycle(2,lr)
learn.save('stage-2')
test_csv = pd.read_csv('../input/Kannada-MNIST/test.csv')

test_csv.drop('id',axis = 'columns',inplace = True)

sub_df = pd.DataFrame(columns=['id','label'])
test_data = np.array(test_csv)
# Handy function to get the image from the tensor data

def get_img(data):

    t1 = data.reshape(28,28)/255

    t1 = np.stack([t1]*3,axis=0)

    img = Image(FloatTensor(t1))

    return img
from fastprogress import progress_bar

mb=progress_bar(range(test_data.shape[0]))
for i in mb:

    timg=test_data[i]

    img = get_img(timg)

    sub_df.loc[i]=[i+1,int(learn.predict(img)[1])]
def decr(ido):

    return ido-1



sub_df['id'] = sub_df['id'].map(decr)

sub_df.to_csv('submission.csv',index=False)
# Displaying the submission file

sub_df.head()
# interfering the learner class with one image



img = data.train_ds[0][0]

learn.predict(img)