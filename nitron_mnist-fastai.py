import numpy as np

import pandas as pd



from fastai import *

from fastai.vision import *



import imageio



import os



import matplotlib.pyplot as plt

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
root = Path('../input')

train_path = Path('train')

rseed = 7

val_size = 0.05
def save_imgs(path:Path, data, labels):

    path.mkdir(parents=True,exist_ok=True)

    for label in np.unique(labels):

        (path/str(label)).mkdir(parents=True,exist_ok=True)

    for i in range(len(data)):

        if(len(labels)!=0):

            imageio.imsave( str( path/str(labels[i])/(str(i)+'.jpg') ), data[i] )

        else:

            imageio.imsave( str( path/(str(i)+'.jpg') ), data[i] )
train_csv = pd.read_csv(root/'train.csv')
test_csv = pd.read_csv(root/'test.csv')
train_csv.head()
data_X, data_y = train_csv.loc[:,'pixel0':'pixel783'], train_csv['label']
from sklearn.model_selection import train_test_split



train_X, val_X, train_y, val_y = train_test_split(data_X, data_y, test_size=val_size,random_state=rseed,stratify=data_y)
def to_img_shape(data_X, data_y=[]):

    data_X = np.array(data_X).reshape(-1,28,28)

    data_X = np.stack((data_X,)*3, axis=-1)

    data_y = np.array(data_y)

    return data_X,data_y
train_X,train_y = to_img_shape(train_X, train_y)
val_X,val_y = to_img_shape(val_X,val_y)
save_imgs(Path('/data/train'),train_X,train_y)
save_imgs(Path('/data/valid'),val_X,val_y)
data = ImageDataBunch.from_folder('/data/',bs=256,size=28,ds_tfms=get_transforms(do_flip=False),num_workers=0).normalize(imagenet_stats)
data.show_batch(3,figsize=(6,6))
learn = cnn_learner(data,models.resnet18,metrics=accuracy,path='.')
learn.fit_one_cycle(4,0.01)
learn.save('s1')
learn.load('s1');
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(30,1e-4)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(8,8))
interp.plot_top_losses(9,figsize=(9,9))
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
sub_df.head()
sub_df.to_csv('submission.csv',index=False)