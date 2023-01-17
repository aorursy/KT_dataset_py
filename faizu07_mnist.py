# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *

import imageio
path = Path('../input/digit-recognizer')

train = pd.read_csv('../input/digit-recognizer/train.csv')

test  =pd.read_csv('../input/digit-recognizer/sample_submission.csv')
train.head()
def to_img_shape(data_X, data_y=[]):

    data_X = np.array(data_X).reshape(-1,28,28)

    data_X = np.stack((data_X,)*3, axis=-1)

    data_y = np.array(data_y)

    return data_X,data_y
data_X, data_y = train.loc[:,'pixel0':'pixel783'], train['label']



from sklearn.model_selection import train_test_split



train_X, val_X, train_y, val_y = train_test_split(data_X, data_y, test_size=0.01,random_state=7,stratify=data_y)
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
data.show_batch(3,figsize=(6,6))
learn = cnn_learner(data, models.resnet50, metrics = accuracy)
learn.fit_one_cycle(4)
learn.save('s1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
lr = slice(1e-06,6e-06)

learn.fit_one_cycle(5,lr)
learn.fit_one_cycle(2,lr)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(8,8))
learn.save('s2')
learn.load('s2')
test_csv = pd.read_csv('../input/digit-recognizer/test.csv')

sub_df = pd.DataFrame(columns=['ImageId','Label'])

test_data = np.array(test_csv)

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
sub_df.to_csv('submission.csv',index=False)