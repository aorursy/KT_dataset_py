import dicom 

import os 

import pandas as pd 



data_dir = '../input/sample_images/'

patients = os.listdir(data_dir)

labels_df = pd.read_csv('/home/mar/db/kaggle/Bowl2017/stage1_labels.csv', index_col=0)



labels_df.head()
