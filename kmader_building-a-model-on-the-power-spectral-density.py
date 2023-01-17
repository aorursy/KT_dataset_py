import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from skimage.io import imread # read image

from PIL import Image 

# imread fails on some of the tiffs so we use PIL

pil_imread = lambda c_file: np.array(Image.open(c_file)) 

from skimage.exposure import equalize_adapthist

from glob import glob



%matplotlib inline

import matplotlib.pyplot as plt
import h5py

t_h5 = os.path.join('..','input', 'fourier-analysis-for-spatial-resolution-estimates', 'training_subset.h5')

with h5py.File(t_h5, 'r') as fd:

    for i in fd.keys():

        print(i, fd[i].shape)

    full_train_df = pd.DataFrame({c_lab: [x for x in fd[c_lab]] for c_lab in fd.keys()})

full_train_df['category'] = full_train_df['category'].map(lambda x: x.decode())

full_train_df['psd'] = full_train_df['psd'].map(lambda x: np.log10(np.mean(x, 1))[30:])

full_train_df.sample(3)
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

cat_enc = LabelEncoder()

cat_enc.fit(full_train_df['category'])

train_df, test_df = train_test_split(full_train_df, 

                                     test_size = 0.4,

                                    random_state = 2018,

                                    stratify = full_train_df['category'])

print('Train', train_df.shape[0], 

      'Test', test_df.shape[0])
from sklearn.ensemble import ExtraTreesClassifier



rfc = ExtraTreesClassifier(n_estimators = 25)

rfc.fit(np.stack(train_df['psd'], 0), 

        train_df['category'])
from sklearn.metrics import classification_report, confusion_matrix

out_pred = rfc.predict(np.stack(test_df['psd'], 0))

print(classification_report(test_df['category'], 

                            out_pred))

plt.matshow(confusion_matrix(test_df['category'], out_pred))
from tpot import TPOTClassifier

tpt = TPOTClassifier(generations = 3, population_size = 10, max_eval_time_mins = 1, verbosity=1)

tpt.fit(np.stack(train_df['psd'], 0), 

        train_df['category'])
from sklearn.metrics import classification_report, confusion_matrix

out_pred = tpt.predict(np.stack(test_df['psd'], 0))

print(classification_report(test_df['category'], 

                            out_pred))

plt.matshow(confusion_matrix(test_df['category'], out_pred))
list_train = glob(os.path.join('..', 'input', 'sp-society-camera-model-identification', 'train', '*', '*.jpg'))

print('Train Files found', len(list_train), 'first file:', list_train[0])

list_test = glob(os.path.join('..', 'input', 'sp-society-camera-model-identification', '*', '*.tif'))

print('Test Files found', len(list_test), 'first file:', list_test[0])