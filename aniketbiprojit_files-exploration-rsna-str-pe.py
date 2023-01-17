import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import os
base_path = '/kaggle/input/rsna-str-pulmonary-embolism-detection/'
df_train = pd.read_csv('/kaggle/input/rsna-str-pulmonary-embolism-detection/train.csv')
df_train.head()
df_train.info()
df_train['qa_motion'].unique()
df_train['qa_contrast'].unique()
df_train['qa_motion_contrast']=df_train['qa_motion']+df_train['qa_contrast']
df_train['qa_motion_contrast'].unique()
# k = df_train['qa_motion_contrast']
# df_train[k==0]
# df_train[k==1]
# df_train[k==2]
df_train['negative_exam_for_pe'].describe()
df_train['negative_exam_for_pe'].unique()
len(df_train.StudyInstanceUID.unique()),len(df_train)
len(df_train.SeriesInstanceUID.unique()),len(df_train)
len(df_train.SOPInstanceUID.unique()),len(df_train)
import os
training_files = os.listdir(base_path+'train')
training_files.index('6897fa9de148')
len(training_files)
df_train[df_train['StudyInstanceUID']=='6897fa9de148']
series_instances = base_path + 'train/6897fa9de148'
os.listdir(series_instances)
study_instances = base_path + 'train/6897fa9de148' + '/' + '2bfbb7fd2e8b'
len(os.listdir(study_instances))
# pd.set_option('display.max_columns', None)  # or 1000
# pd.set_option('display.max_rows', None)  # or 1000
# pd.set_option('display.max_colwidth', -1)  # or 199

sample_submission = pd.read_csv(base_path + 'sample_submission.csv')
# sample_submission = sample_submission.sort_values('id')
sample_submission.tail(10)
sample_submission.head(10)
sample_submission[sample_submission.id.str.contains('df06fad17bc3', regex= True, na=False)]
sample_submission[sample_submission.id.str.contains('41cb110f177e', regex= True, na=False)]
sample_submission[sample_submission.id.str.contains('4f6f2387f9ba', regex= True, na=False)]
sample_submission[sample_submission.id.str.contains('012c12fe09c3', regex= True, na=False)]
%matplotlib inline
import pydicom
data=pydicom.dcmread('../input/rsna-str-pulmonary-embolism-detection/test/00268ff88746/75d23269adbd/012c12fe09c3.dcm')
data
import matplotlib.pyplot as plt

plt.imshow(data.pixel_array, cmap=plt.cm.bone)
# pltshow()ss
