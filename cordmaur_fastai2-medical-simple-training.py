# Importing our dependencies

import torch

import fastai

from fastai.medical import *

from fastai.medical.imaging import *

from fastai.torch_core import *

import matplotlib.pyplot as plt

import pandas as pd

from fastai.vision.all import *

from pathlib import Path

import pydicom



!conda install -c conda-forge gdcm -y

import gdcm
# Just for visualization purpose, we will grab the files contained in just 1 study (each study has many images/slices)

# For that, we will use Fastai 2 get_dicom_files, that just maps all dcm files within recursive subdirs

sample_files = get_dicom_files('../input/rsna-str-pulmonary-embolism-detection/train/0003b3d648eb')
# Let's grab the first file of this study and display its metadata

dicom = dcmread(sample_files[0])

dicom
# using a snippet from fastai medical tutorial, we will display the images with different scales

scales = False, True, dicom_windows.brain, dicom_windows.subdural

titles = 'raw','normalized','brain windowed','subdural windowed'

for s,a,t in zip(scales, subplots(2,2,imsize=4)[1].flat, titles):

    dicom.show(scale=s, ax=a, title=t)
dicom.show(cmap=plt.cm.gist_ncar, figsize=(6,6))
# Initially, we will create our Pandas Dataframes with the CSV diles.

# The train dataframe contains all the information to get to the images, so it will be passed 

# as the source of our dataset and the datablock will be in charge of transforming it into 

# inputs and targets (x, y)



train = pd.read_csv('../input/rsna-str-pulmonary-embolism-detection/train.csv', low_memory=False)

test = pd.read_csv('../input/rsna-str-pulmonary-embolism-detection/test.csv')
# We will separate in train_pos and train_neg. Afterwards, we will grab 100k images from each dataframe and join then to do our training

negatives = train['negative_exam_for_pe'] == 1

train_neg = train[negatives]

train_pos = train[~negatives]
balanced = pd.concat([train_neg[:250000], train_pos[:250000]], axis=0)

balanced
vocab = ['negative_exam_for_pe', 'pe_present_on_image', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1',

         'leftsided_pe', 'chronic_pe', 'rightsided_pe', 'acute_and_chronic_pe',

         'central_pe', 'indeterminate']

vocab.sort()



def get_x(row):

    base_path = Path('../input/rsna-str-pulmonary-embolism-detection/train')

    file_path = f"{row['StudyInstanceUID']}/{row['SeriesInstanceUID']}/{row['SOPInstanceUID']}.dcm"

    return base_path/file_path



# def get_y(row):

#     labels = row[vocab]

    

#     return list(labels.index[labels==1])



def get_encoded_y(row):

    return row[vocab].values.squeeze().astype('long')
# we will test our functions by passing an arbitrary row

r = train.iloc[3]

get_x(r), get_encoded_y(r)
dblock = DataBlock(#blocks=(ImageBlock(cls=PILDicom), MultiCategoryBlock(encoded=True, vocab=vocab)),

                   blocks=(TransformBlock([PILDicom.create, ToTensor]), MultiCategoryBlock(encoded=True, vocab=vocab)),

                   get_x=get_x,

                   get_y=get_encoded_y,

                  )

dsets = dblock.datasets(balanced, verbose=False)
# If we index the dataset, we get x and y as return

dsets[150000]
# to check the sanity of our dblock, we could also call the `.summary()` function

# dblock.summary(train.iloc[:100])
# dls = dblock.dataloaders(train.iloc[:20000], bs=16, num_workers=0)

# To check our dataloader we can either create an item or a full batch

dls = dsets.dataloaders(bs=64, num_workers=0)

dls.create_item(1)
dls.show_batch()
# Will create a multicategorical accuracy

def accuracy_multi(inp, targ, thresh=0.5, sigmoid=True):

    "Compute accuracy when `inp` and `targ` are the same size."

    if sigmoid: inp = inp.sigmoid()

    return ((inp>thresh)==targ.bool()).float().mean()



# accuracy_multi(y, activs, thresh=0.5)
learn = cnn_learner(dls, resnet18, n_in=1, metrics=accuracy_multi)

learn.model_dir = '.'



try:

    learn.load('../input/fastai2-medical-simple-training/resnet18-v3')

    print('Model Loaded Successfully')

except:

    print('Could not load model. Content of added data:')



    !ls ../input/fastai2-medical-simple-training/

    print('Content of working directory')

    !ls ../working
#testing one pass through the model

# x,y = to_cpu(dls.train.one_batch())

# activs = learn.model(x)

# activs.shape
# apply the new metrics and look for the best learning rate

# learn.metrics=accuracy_multi

# learn.lr_find()



# I noticed a problem when training with more data. 

# I also noticed that some dcm cannot be opened correctly, have to check further.

# It seems to be a good idea to iterate through all dcms and check if thay are 

# opening correctly and that they can be cast to PILDicom
learn.fine_tune(1, base_lr=2e-2, freeze_epochs=1)
learn.model_dir = '.'

learn.save('./resnet18-v4')
item = dsets.valid[1500]
learn.predict(item[0])
item[1]
# interp = ClassificationInterpretation.from_learner(learn)

# interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
# interp.plot_top_losses(6)