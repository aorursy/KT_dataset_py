import librosa

import numpy as np

import pandas as pd

from sklearn import *

from PIL import Image

from numba import njit

import soundfile as sf

from scipy.io import wavfile

import scipy, cv2, imagehash

import matplotlib.pyplot as plt

from multiprocessing import Pool, cpu_count

%matplotlib inline



label = {'FAKE':1, 'REAL':0}

path = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/'

train = pd.read_json(path + 'metadata.json').T.reset_index()

train.columns = ['filename', 'label', 'split', 'original']

train['path'] = path + train['filename']

train['label'] = train['label'].map(label)



path = '/kaggle/input/deepfake-detection-challenge/'

test = pd.read_csv(path + 'sample_submission.csv')

test['path'] = path + 'test_videos/' + test['filename']

print(train.shape, test.shape)
train['file_meta'] = train.path.map(lambda x: open(x,'rb').read(1000))

test['file_meta'] = test.path.map(lambda x: open(x,'rb').read(1000))
import xgboost as xgb

import joblib, pickle



ft_cv2 = pickle.load(open('/kaggle/input/deepfake/cv.pickle','rb'))

model2 = joblib.load('/kaggle/input/deepfake/model.xgb')



test['label'] = model2.predict(xgb.DMatrix(ft_cv2.transform(test.file_meta)), ntree_limit=model2.best_ntree_limit)

test[['filename', 'label']].to_csv('submission.csv', index=False)