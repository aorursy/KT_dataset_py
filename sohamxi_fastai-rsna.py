# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai.imports import *

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from matplotlib.patches import Rectangle

import os

import seaborn as sns

import pydicom

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# # Any results you write to the current directory are saved as output.
%matplotlib inline

%reload_ext autoreload

%autoreload 2
import pandas as pd

stage_2_detailed_class_info = pd.read_csv("../input/rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv")

stage_2_train_labels = pd.read_csv("../input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv")
stage_2_detailed_class_info.head()
train_class_df = stage_2_train_labels.merge(stage_2_detailed_class_info, left_on='patientId', right_on='patientId', how='inner')
torch.cuda.is_available()
torch.backends.cudnn.enabled
# Looking at the CSV

stage_2_train_labels.head()
# Setting the path

path = "/kaggle/input/rsna-pneumonia-detection-challenge/"
f, ax = plt.subplots(1,1, figsize=(6,4))

total = float(len(stage_2_detailed_class_info))

sns.countplot(stage_2_detailed_class_info['class'],order = stage_2_detailed_class_info['class'].value_counts().index, palette='Set3').set_title('Class Distribution')

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(100*height/total),

            ha="center") 

plt.show()
def show_dicom_images_with_boxes(data):

    img_data = list(data.T.to_dict().values())

    f, ax = plt.subplots(3,3, figsize=(16,18))

    for i,data_row in enumerate(img_data):

        patientImage = data_row['patientId']+'.dcm'

        imagePath = os.path.join(path,"stage_2_train_images/",patientImage)

        data_row_img_data = pydicom.read_file(imagePath)

        modality = data_row_img_data.Modality

        age = data_row_img_data.PatientAge

        sex = data_row_img_data.PatientSex

        data_row_img = pydicom.dcmread(imagePath)

        ax[i//3, i%3].imshow(data_row_img.pixel_array, cmap=plt.cm.bone) 

        ax[i//3, i%3].axis('off')

        ax[i//3, i%3].set_title('ID: {}\nModality: {} Age: {} Sex: {} Target: {}\nClass: {}'.format(

                data_row['patientId'],modality, age, sex, data_row['Target'], data_row['class']))

        rows = train_class_df[train_class_df['patientId']==data_row['patientId']]

        box_data = list(rows.T.to_dict().values())

        for j, row in enumerate(box_data):

            ax[i//3, i%3].add_patch(Rectangle(xy=(row['x'], row['y']),

                        width=row['width'],height=row['height'], 

                        color="yellow",alpha = 0.1))   

    plt.show()
show_dicom_images_with_boxes(train_class_df[train_class_df['Target']==1].sample(9))
from fastai.vision import *

from fastai.basics import *

from fastai.metrics import error_rate

import pydicom

import imageio

import PIL

import json, pdb
def open_dcm_image(fn:PathOrStr,convert_mode:str='RGB',after_open:Callable=None)->Image:

    "Return `Image` object created from image in file `fn`."

    array = pydicom.dcmread(fn).pixel_array

    x = PIL.Image.fromarray(array).convert('RGB')

    return Image(pil2tensor(x,np.float32).div_(255))

# Updating the default method to the custom method

vision.data.open_image = open_dcm_image
# Setting Transforms

tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0)

# Creating Image DataBunch

data = ImageDataBunch.from_csv(path,folder='stage_2_train_images',csv_labels='stage_2_train_labels.csv',ds_tfms=tfms,fn_col='patientId',label_col='Target',suffix='.dcm',seed=47,size=224)
data.show_batch(rows=3, figsize=(9,9))
learn = cnn_learner(data, models.resnet34, metrics=[accuracy,error_rate])
learn.fit_one_cycle(2)
learn.model_dir = '/kaggle/working'

learn.lr_find()
learn.load('stage-1-34')
learn.recorder.plot()
learn.unfreeze()

defaults.device = torch.device('cuda')

learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-2))
learn.save('stage-2-34')
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,8))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
learn2 = cnn_learner(data, models.resnet50, metrics=[accuracy,error_rate])
learn2.model_dir = '/kaggle/working'

learn2.lr_find()
learn2.recorder.plot()
learn2.unfreeze()

# It's mostly overfitting after 4 epochs

learn2.fit_one_cycle(6, max_lr=slice(1e-4,4*1e-2))
learn2.save('stage1-50')
interp = ClassificationInterpretation.from_learner(learn2)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,8))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)