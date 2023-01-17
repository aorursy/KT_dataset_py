import dicom

import os

import pandas as pd



data_dir = '../input/sample_images/'

patients = os.listdir(data_dir)

labels_df = pd.read_csv('../input/stage1_labels.csv')
!ls -la /kaggle/input/sample_images/