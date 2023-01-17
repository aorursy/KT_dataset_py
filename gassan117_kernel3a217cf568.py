# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import librosa
def load_test_clip(path, start_time, duration=5):

    return librosa.load(path, offset=start_time, duration=duration)[0]

TEST_FOLDER = '../input/birdsong-recognition/test_audio/'

test_info = pd.read_csv('../input/birdsong-recognition/test.csv')

test_info.head()
train = pd.read_csv('../input/birdsong-recognition/train.csv')

birds = train['ebird_code'].unique()

train.head()
test_info
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