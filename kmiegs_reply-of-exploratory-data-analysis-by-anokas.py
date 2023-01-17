import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

import glob

%matplotlib inline

p = sns.color_palette()



os.listdir('../input')
for d in os.listdir('../input/sample_images'):

    print("Patient '{}' has {} scans".format(d, len(os.listdir('../input/sample_images/' + d))))

print('----')

print('Total patients {} Total DCM files {}'.format(len(os.listdir('../input/sample_images')), 

                                                      len(glob.glob('../input/sample_images/*/*.dcm'))))
patient_sizes = [len(os.listdir('../input/sample_images/' + d)) for d in os.listdir('../input/sample_images')]

plt.hist(patient_sizes, color=p[2])

plt.ylabel('Number of patients')

plt.xlabel('DICOM files')

plt.title('Histogram of DICOM count per patient')
sizes = [os.path.getsize(dcm)/1000000 for dcm in glob.glob('../input/sample_images/*/*.dcm')]

print('DCM file sizes: min {:.3}MB max {:.3}MB avg {:.3}MB std {:.3}MB'.format(np.min(sizes), 

                                                       np.max(sizes), np.mean(sizes), np.std(sizes)))
df_train = pd.read_csv('../input/stage1_labels.csv')

df_train.head()
print('Number of training patients: {}'.format(len(df_train)))

print('Cancer rate: {:.4}%'.format(df_train.cancer.mean()*100))
from sklearn.metrics import log_loss

logloss = log_loss(df_train.cancer, np.zeros_like(df_train.cancer) + df_train.cancer.mean())

print('Training logloss is {}'.format(logloss))
sample = pd.read_csv('../input/stage1_sample_submission.csv')

sample['cancer'] = df_train.cancer.mean()

sample.to_csv('naive_submission.csv', index=False)