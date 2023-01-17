import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

import plotly.express as px

import librosa



import pywt

from statsmodels.robust import mad

from warnings import filterwarnings; filterwarnings('ignore')



import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots

from scipy.io import wavfile

import subprocess



from tqdm.notebook import tqdm

import librosa



PATH = '../input/birdsong-recognition/'

TEST_FOLDER = '../input/birdsong-recognition/test_audio/'

model_path = '../input/xgb-model/XGB_Model.pickle'

train_path = '../input/train-pkl/train (1).pkl'

os.listdir(PATH)

RANDOM_SEED = 4444
def load_test_clip(path, start_time, duration=5):

    return librosa.load(path, offset=start_time, duration=duration)[0]



def make_prediction(y, le, model):

    feats = np.array([np.min(y), np.max(y), np.mean(y), np.std(y)]).reshape(1, -1)

    return le.inverse_transform(model.predict(feats))[0]
train = pd.read_pickle(train_path)
train.head()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

train['target'] = le.fit_transform(train['target_raw'].values)
from xgboost import XGBClassifier
try:

    preds = []

    for index, row in test.iterrows():

        # Get test row information

        site = row['site']

        start_time = row['seconds'] - 5

        row_id = row['row_id']

        audio_id = row['audio_id']



        # Get the test sound clip

        if site == 'site_1' or site == 'site_2':

            y = load_test_clip(TEST_FOLDER + audio_id + '.mp3', start_time)

        else:

            y = load_test_clip(TEST_FOLDER + audio_id + '.mp3', 0, duration=None)



        # Make the prediction

        pred = make_prediction(y, le, model)



        # Store prediction

        preds.append([row_id, pred])

except:

     preds = pd.read_csv('../input/birdsong-recognition/sample_submission.csv')

preds = pd.DataFrame(preds, columns=['row_id', 'birds'])
preds.to_csv('submission.csv', index = False)