# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/rsna-str-pulmonary-embolism-detection/train.csv')

test = pd.read_csv('../input/rsna-str-pulmonary-embolism-detection/test.csv')



print('train.shape', train.shape, 'test.shape', test.shape)
train.head(5)
train.groupby('StudyInstanceUID')['SeriesInstanceUID'].nunique().max(), test.groupby('StudyInstanceUID')['SeriesInstanceUID'].nunique().max()
np.intersect1d(train.StudyInstanceUID.unique(), test.StudyInstanceUID.unique())
train_image_num_per_patient = train.groupby('StudyInstanceUID')['SOPInstanceUID'].nunique()

test_image_num_per_patient = test.groupby('StudyInstanceUID')['SOPInstanceUID'].nunique()
train_image_num_per_patient.describe()
test_image_num_per_patient.describe()
import matplotlib.pyplot as plt

plt.title('image_num_per_patient')

plt.hist(train_image_num_per_patient, bins=100, label='train', density=True)

plt.hist(test_image_num_per_patient, bins=100, label='test', density=True)

plt.legend()

plt.show()
FOLD_NUM = 20

target_cols = [c for i, c in enumerate(train.columns) if i > 2]
# build summary of image num and target variables for each patient

train_per_patient_char = pd.DataFrame(index=train_image_num_per_patient.index, columns=['image_per_patient'], data=train_image_num_per_patient.values.copy())

for t in target_cols:

    train_per_patient_char[t] = train_per_patient_char.index.map(train.groupby('StudyInstanceUID')[t].mean())



train_per_patient_char.head(10)
# make image_per_patient and pe_present_on_image into bins

bin_counts = [40] #, 20]

digitize_cols = ['image_per_patient'] #, 'pe_present_on_image']

non_digitize_cols = [c for c in train_per_patient_char.columns if c not in digitize_cols]
for i, c in enumerate(digitize_cols):

    bin_count = bin_counts[i]

    percentiles = np.percentile(train_per_patient_char[c], q=np.arange(bin_count)/bin_count*100.)

    #print(percentiles)

    print(train_per_patient_char[c].value_counts())

    train_per_patient_char[c+'_digitize'] = np.digitize(train_per_patient_char[c], percentiles, right=False)

    print(train_per_patient_char[c+'_digitize'].value_counts())

    plt.hist(train_per_patient_char[c+'_digitize'], bins=bin_count)

    plt.show()
train_per_patient_char['key'] = train_per_patient_char[digitize_cols[0]+'_digitize'].apply(str)

for c in digitize_cols[1:]:

    train_per_patient_char['key'] = train_per_patient_char['key']+'_'+train_per_patient_char[c+'_digitize'].apply(str)



train_per_patient_char['key'].value_counts()
from sklearn.model_selection import StratifiedKFold

folds = FOLD_NUM

kfolder = StratifiedKFold(n_splits=folds, shuffle=True, random_state=719)

val_indices = [val_indices for _, val_indices in kfolder.split(train_per_patient_char['key'], train_per_patient_char['key'])]



train_per_patient_char['fold'] = -1

for i, vi in enumerate(val_indices):

    patients = train_per_patient_char.index[vi]

    train_per_patient_char.loc[patients, 'fold'] = i

train_per_patient_char['fold'].value_counts()
# check each fold for the distribution of the number of images per patients

for col in digitize_cols:

    fig, axs = plt.subplots(nrows=4, ncols=int(np.floor(folds/4)), constrained_layout=False, sharex=True, sharey=True)

    fig.set_figheight(10)

    fig.set_figwidth(20)

    axs = axs.flat

    for i, vi in enumerate(val_indices):

        patients = train_per_patient_char.index[vi]

        axs[i].set_title(col+' fold_'+str(i))

        axs[i].hist(train_per_patient_char.loc[patients, col], bins=20, range=(train_per_patient_char[col].min(), train_per_patient_char[col].max()))

    plt.show()
# check each fold for the target distribution

for col in non_digitize_cols:

    fig, axs = plt.subplots(nrows=4, ncols=int(np.floor(folds/4)), constrained_layout=False, sharex=True, sharey=True)

    fig.set_figheight(10)

    fig.set_figwidth(20)

    axs = axs.flat

    for i, vi in enumerate(val_indices):

        patients = train_per_patient_char.index[vi]

        axs[i].set_title(col+' fold_'+str(i))

        axs[i].hist(train_per_patient_char.loc[patients, col], bins=20, range=(train_per_patient_char[col].min(), train_per_patient_char[col].max()))

    plt.show()
train_per_patient_char.to_csv('rsna_train_splits_fold_{}.csv'.format(FOLD_NUM))