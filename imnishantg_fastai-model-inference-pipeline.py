!cp ../input/fastai-model-training-pipeline/gdcm.tar .

!tar -xvzf gdcm.tar

!conda install --offline ./gdcm/gdcm-2.8.9-py37h71b2a6d_0.tar.bz2

print("done")
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

import shutil



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
source = '../input/rsna-str-pulmonary-embolism-detection'

upload = '../input/fastai-model-training-pipeline/'
files = os.listdir(source)

files
upl = os.listdir(upload)

upl
df = pd.read_csv(f'{source}/test.csv')

print(df.shape)

df.head()
# getter functions



get_x = lambda x:f'{source}/test/{x.StudyInstanceUID}/{x.SeriesInstanceUID}/{x.SOPInstanceUID}.dcm'



vocab = ['pe_present_on_image', 'negative_exam_for_pe', 'indeterminate', 

         'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1', # Only one label should be true at a time

         'chronic_pe', 'acute_and_chronic_pe', # Only one label can be true at a time

         'leftsided_pe', 'central_pe', 'rightsided_pe', # More than one label can be true at a time

         'qa_motion', 'qa_contrast', 'flow_artifact', 'true_filling_defect_not_pe'] # These are only informational. Maybe use it for study level inferences



# get_y = ColReader(vocab)
tfms = [IntToFloatTensor(div=1000.0, div_mask=1), 

        *aug_transforms(size=224), 

        Normalize.from_stats(*imagenet_stats)]
block = DataBlock(blocks=(ImageBlock(cls=PILDicom)),

                  get_x=get_x,

                  batch_tfms=tfms)
dls = block.dataloaders(df, bs=512, num_workers=0)

dls.show_batch(max_n=9, nrows=3, ncols=3, figsize=(20,20))
head = create_head(nf=1024, n_out=14, lin_ftrs=[512, 128], concat_pool=True)

config = cnn_config(custom_head=head)



learn = cnn_learner(dls, resnet34, config=config, n_out=14, pretrained=False, loss_func=nn.BCEWithLogitsLoss())
learn.model_dir = '.'

learn.load('../input/fastai-model-training-pipeline/resnet34')
test_data = dls.test_dl(df)   # df[:1000]
preds = learn.get_preds(dl=test_data, act=sigmoid)
test_filepaths = get_dicom_files(f'{source}/test/')
test_filepaths[0]
parse_StudyInstanceUID = lambda x: x.__str__().split('/')[-3]

parse_SeriesInstanceUID = lambda x: x.__str__().split('/')[-2]

parse_SOPInstanceUID = lambda x: x.__str__().split('/')[-1].split('.')[0]
sub = pd.DataFrame(preds[0])

sub.columns = vocab

sub['StudyInstanceUID'] = [parse_StudyInstanceUID(x) for x in test_filepaths]

sub['SeriesInstanceUID'] = [parse_SeriesInstanceUID(x) for x in test_filepaths]

sub['SOPInstanceUID'] = [parse_SOPInstanceUID(x) for x in test_filepaths]



print(sub.shape)

sub.head()
sub_p1 = sub.loc[:, ['SOPInstanceUID', 'pe_present_on_image']]

sub_p1.columns = ['id', 'label']



print(sub_p1.shape)

sub_p1.head()
sel_cols = ['negative_exam_for_pe', 'indeterminate', 

            'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1', # Only one label should be true at a time

            'chronic_pe', 'acute_and_chronic_pe', # Only one label can be true at a time

            'leftsided_pe', 'central_pe', 'rightsided_pe'] # More than one label can be true at a time



sub_p2 = sub.loc[:, ['StudyInstanceUID']+sel_cols]



# Temporary summary for study level predictions

agg_func = {'negative_exam_for_pe' : ['min'], 

            'indeterminate' : ['min'], 

            'rv_lv_ratio_gte_1' : ['mean'], 

            'rv_lv_ratio_lt_1' : ['mean'], 

            'chronic_pe' : ['mean'], 

            'acute_and_chronic_pe' : ['mean'], 

            'leftsided_pe' : ['max'], 

            'central_pe' : ['max'], 

            'rightsided_pe' : ['max']}



sub_p2 = sub_p2.groupby(['StudyInstanceUID']).agg(agg_func)

sub_p2.columns = sub_p2.columns.droplevel(1)

sub_p2 = sub_p2.reset_index()



# Data Reshaping

sub_p2 = pd.melt(sub_p2, id_vars=['StudyInstanceUID'], value_vars=sel_cols)

sub_p2['id'] = sub_p2['StudyInstanceUID']+'_'+sub_p2['variable']

sub_p2.drop(['StudyInstanceUID', 'variable'], inplace=True, axis=1)

sub_p2 = sub_p2.loc[:, ['id', 'value']]

sub_p2.columns = ['id', 'label']



print(sub_p2.shape)

sub_p2.head()
finsub = pd.concat([sub_p2, sub_p1])



print(finsub.shape)

finsub.head()
finsub.to_csv('../working/submission.csv', index=False)