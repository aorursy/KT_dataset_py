import numpy as np

import pandas as pd

import librosa



np.random.seed(0)
def load_test_clip(path, start_time, duration=5):

    return librosa.load(path, offset=start_time, duration=duration)[0]
TEST_FOLDER = '../input/birdsong-recognition/test_audio/'

test_info = pd.read_csv('../input/birdsong-recognition/test.csv')

test_info.head()
train = pd.read_csv('../input/birdsong-recognition/train.csv')

birds = train['ebird_code'].unique()

birds[0:20]
def make_prediction(sound_clip, birds):

    return np.random.choice(birds)
try:

    preds = []

    for index, row in test_info.iterrows():

        # Get test row information

        site = row['site']

        start_time = row['seconds'] - 5

        row_id = row['row_id']

        audio_id = row['audio_id']



        # Get the test sound clip

        if site == 'site_1' or site == 'site_2':

            sound_clip = load_test_clip(TEST_FOLDER + audio_id + '.mp3', start_time)

        else:

            sound_clip = load_test_clip(TEST_FOLDER + audio_id + '.mp3', 0, duration=None)



        # Make the prediction

        pred = make_prediction(sound_clip, birds)



        # Store prediction

        preds.append([row_id, pred])



    preds = pd.DataFrame(preds, columns=['row_id', 'birds'])

except:

    preds = pd.read_csv('../input/birdsong-recognition/sample_submission.csv')
preds.to_csv('submission.csv', index=False)