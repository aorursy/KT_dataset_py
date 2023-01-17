from numpy.random import seed

seed(101)



import pandas as pd

import numpy as np

from glob import glob 



from fastai.vision import *

from fastai.metrics import error_rate

import os

import cv2



import imageio

import skimage

import skimage.io

import skimage.transform



from sklearn.utils import shuffle

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

import itertools

import shutil

import matplotlib.pyplot as plt

%matplotlib inline
os.listdir('../input')
print(len(os.listdir('/kaggle/input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/')))

print(len(os.listdir('/kaggle/input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/')))

path_train = "/kaggle/input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/"

path_test = "/kaggle/input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/"

meta_path = "/kaggle/input/coronahack-chest-xraydataset/"
meta_data = pd.read_csv(meta_path+'Chest_xray_Corona_Metadata.csv',index_col=[0])

meta_data.head()
plt.figure(1, figsize = (15 , 7))

plt.subplot(1 , 2 , 1)

img = glob(path_train+"/*.jpeg") 

img = np.asarray(plt.imread(img[0]))

plt.title('COVID X-RAY')

plt.imshow(img)
os.mkdir("/kaggle/corona_check")

os.mkdir("/kaggle/corona_check/train")

os.mkdir("/kaggle/corona_check/test")

os.mkdir("/kaggle/corona_check/train/Normal/")

os.mkdir("/kaggle/corona_check/train/COVID19/")

os.mkdir("/kaggle/corona_check/test/Normal/")

os.mkdir("/kaggle/corona_check/test/COVID19/")
!cd /kaggle/corona_check/train/

!rm /kaggle/corona_check/train/*.jpeg



!cd /kaggle/corona_check/test/

!rm /kaggle/corona_check/test/*.jpeg
def copy_img(src_path,dst_path):

    try:

        shutil.copy(src_path, dst_path)

        stmt ='File Copied'

    except IOError as e:

        print('Unable to copy file {} to {}'

              .format(src_path, dst_path))

        stmt ='Copy Failed - IO Error'

    except:

        print('When try copy file {} to {}, unexpected error: {}'

              .format(src_path, dst_path, sys.exc_info()))

        stmt ='Copy Failed - other Error'+ sys.exc_info()

        

    return stmt
data_dir="../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/"

metadata_path="../input/coronahack-chest-xraydataset/"
train_dir = os.path.join(data_dir, 'train')

test_dir = os.path.join(data_dir, 'test')
meta_data['File_path']=''

meta_data.loc[meta_data['Dataset_type']=='TRAIN','File_path']=train_dir+'/'

meta_data.loc[meta_data['Dataset_type']=='TEST','File_path']=test_dir+'/'

meta_data['X_ray_img_nm_path']=meta_data['File_path']+meta_data['X_ray_image_name']
meta_data.head()
meta_COVID_19_train = meta_data[(meta_data['Dataset_type']=='TRAIN') & 

                        ((meta_data['Label']=='Normal')|(meta_data['Label']=='Pnemonia') & (meta_data['Label_2_Virus_category']=='COVID-19'))]





meta_COVID_19_test = meta_data[(meta_data['Dataset_type']=='TEST') & 

                        ((meta_data['Label']=='Normal')|(meta_data['Label']=='Pnemonia') & (meta_data['Label_2_Virus_category']=='COVID-19'))]





## Moving the 10 Corona Infected dataset to Test



meta_data_covid_test = meta_data[meta_data['Label_2_Virus_category']=='COVID-19'].sample(12)

meta_COVID_19_train = meta_COVID_19_train[~meta_COVID_19_train['X_ray_image_name'].isin(meta_data_covid_test['X_ray_image_name'])]

meta_COVID_19_test_fnl = pd.concat([meta_data_covid_test,meta_COVID_19_test],ignore_index=False)
meta_COVID_19_train.loc[meta_COVID_19_train['Label'] =='Pnemonia','Label']='COVID19'

meta_COVID_19_test_fnl.loc[meta_COVID_19_test_fnl['Label'] =='Pnemonia','Label']='COVID19'
meta_COVID_19_train['Img_tgt_path']="/kaggle/corona_check/train/"

meta_COVID_19_test_fnl['Img_tgt_path']="/kaggle/corona_check/test/"
meta_COVID_19_train.loc[meta_COVID_19_train['Label']=='Normal','Img_tgt_path']=meta_COVID_19_train['Img_tgt_path']+'Normal/'

meta_COVID_19_train.loc[meta_COVID_19_train['Label']=='COVID19','Img_tgt_path']=meta_COVID_19_train['Img_tgt_path']+'COVID19/'



meta_COVID_19_test_fnl.loc[meta_COVID_19_test_fnl['Label']=='Normal','Img_tgt_path']=meta_COVID_19_test_fnl['Img_tgt_path']+'Normal/'

meta_COVID_19_test_fnl.loc[meta_COVID_19_test_fnl['Label']=='COVID19','Img_tgt_path']=meta_COVID_19_test_fnl['Img_tgt_path']+'COVID19/'
meta_COVID_19_train['Move_status'] = np.vectorize(copy_img)(meta_COVID_19_train['X_ray_img_nm_path'],meta_COVID_19_train['Img_tgt_path'])

meta_COVID_19_test_fnl['Move_status'] = np.vectorize(copy_img)(meta_COVID_19_test_fnl['X_ray_img_nm_path'],meta_COVID_19_test_fnl['Img_tgt_path'])
dirname = '/kaggle/corona_check/'

train_path = os.path.join(dirname, 'train/')

train_nrml_pth = os.path.join(train_path, 'Normal/')

train_covid19_pth = os.path.join(train_path, 'COVID19/')



test_path = os.path.join(dirname, 'test/')

test_nrml_pth = os.path.join(train_path, 'Normal/')

test_covid19_pth = os.path.join(train_path, 'COVID19/')
plt.figure(1, figsize = (15 , 7))

plt.subplot(1 , 2 , 1)

img = glob(train_nrml_pth+"/*.jpeg")

img = np.asarray(plt.imread(img[0]))

plt.title('Normal Chest X-RAY')

plt.imshow(img)



plt.subplot(1 , 2 , 2)

img = glob(train_covid19_pth+"/*.jpeg") 

img = np.asarray(plt.imread(img[0]))

plt.title('COVID CHEST X-RAY')

plt.imshow(img)



plt.show()
src = (ImageList.from_folder(dirname)

       .split_by_rand_pct(valid_pct=0.2)

       .label_from_folder()

       .transform(get_transforms(), size=256)

       .add_test_folder(test_path))
data = (src.databunch(bs=32)

        .normalize(imagenet_stats))



data.show_batch(rows=3,figsize=(8,8))
#lets recheck the classes

data.classes
print(data.c, len(data.train_ds), len(data.valid_ds))
learner34=cnn_learner(data,models.resnet34,metrics=accuracy)
#Model Training

learner34.fit_one_cycle(4)
#find the learning rate and plot the learning rate

lr=1e-2

learner34.fit_one_cycle(4,slice(lr))

learner34.recorder.plot()

learner34.save('stage-1')
#Plot the losses of the model

interp=ClassificationInterpretation.from_learner(learner34)

losses,idxs = interp.top_losses()

interp.plot_top_losses(9,figsize=(12,12))

len(data.valid_ds)==len(losses)==len(idxs)
#Plot a confusion matrix

interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
learner34.export(file = Path("/kaggle/working/export.pkl"))
learner34.save('stage-1')
learner50=cnn_learner(data,models.resnet50,metrics=accuracy)
#Model Training

learner50.lr_find()

learner50.recorder.plot()
#find the learning rate and plot the learning rate

lr=1e-2

learner50.fit_one_cycle(4,slice(lr))

learner50.recorder.plot()

learner50.save('stage-2')
#Plot the losses of the model

interp=ClassificationInterpretation.from_learner(learner50)

interp.plot_top_losses(9,figsize=(12,12))

len(data.valid_ds)==len(losses)==len(idxs)
#Plot a confusion matrix

interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
learner50.show_results()