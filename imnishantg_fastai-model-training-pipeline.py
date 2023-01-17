!cp ../input/gdcm-conda-install/gdcm.tar .

!tar -xvzf gdcm.tar

!conda install --offline ./gdcm/gdcm-2.8.9-py37h71b2a6d_0.tar.bz2
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



## Loading Libraries

import numpy as np

import pandas as pd



import fastai

from fastai.basics import *

from fastai.callback.all import *

from fastai.vision.all import *

from fastai.medical.imaging import *

import torchvision.models as models

import pydicom



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
source = '../input/rsna-str-pulmonary-embolism-detection'

files = os.listdir(source)

files
df = pd.read_csv(f'{source}/train.csv')

print(df.shape)

df.head()
get_x = lambda x:f'{source}/train/{x.StudyInstanceUID}/{x.SeriesInstanceUID}/{x.SOPInstanceUID}.dcm'



vocab = ['pe_present_on_image', 'negative_exam_for_pe', 'indeterminate', 

         'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1', # Only one label should be true at a time

         'chronic_pe', 'acute_and_chronic_pe', # Only one label can be true at a time

         'leftsided_pe', 'central_pe', 'rightsided_pe', # More than one label can be true at a time

         'qa_motion', 'qa_contrast', 'flow_artifact', 'true_filling_defect_not_pe'] # These are only informational. Maybe use it for study level inferences



get_y = ColReader(vocab) 
tfms = [IntToFloatTensor(div=1000.0, div_mask=1), 

        *aug_transforms(size=224)]
block = DataBlock(blocks=(ImageBlock(cls=PILDicom), MultiCategoryBlock(vocab=vocab, encoded=True)), 

                  get_x=get_x,

                  get_y=get_y,

                  batch_tfms=tfms)
dls = block.dataloaders(df[:1600], bs=32, num_workers=0)  # Change df[:1600] to df to train on complete data. 

dls.show_batch(max_n=9, nrows=3, ncols=3, figsize=(20,20))
def accuracy_multi(inp, targ, thresh=0.5, sigmoid=True):

    "Compute accuracy when `inp` and `targ` are the same size."

    if sigmoid: inp = inp.sigmoid()

    return ((inp>thresh)==targ.bool()).float().mean()
head = create_head(nf=1024, n_out=14, lin_ftrs=[512, 128], concat_pool=True)

config = cnn_config(custom_head=head)



learn = cnn_learner(dls, resnet34, config=config, metrics=accuracy_multi)
lr_good = learn.lr_find()
lr_good # pick the learning rate from here to fit method below
learn.fit_one_cycle(3, lr_max=0.03)
learn.model_dir = '.'

learn.save(file='../working/resnet34')
interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()

len(dls.valid_ds)==len(losses)==len(idxs)

interp.plot_confusion_matrix(figsize=(7,7))