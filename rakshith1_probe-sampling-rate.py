import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import librosa

import os
BASE_TEST_DIR = '../input/birdsong-recognition' if os.path.exists('../input/birdsong-recognition/test_audio') else '../input/birdcall-check'
df_test = pd.read_csv(f'{BASE_TEST_DIR}/test.csv')

df_train = pd.read_csv('../input/birdsong-recognition/train.csv')

sub_test_12 = df_test[df_test.site.isin(['site_1', 'site_2'])]

sub_test_3 = df_test[df_test.site.isin(['site_3'])]
%%capture

submission = {'row_id': [], 'birds': []}

st=set()

for audio_id, data in sub_test_12.groupby('audio_id'):

    wav,sr=librosa.load(BASE_TEST_DIR+'/test_audio/'+ audio_id+'.mp3',sr=None,duration=1)

    st.add(sr)

    submission['row_id'].extend(data['row_id'].values)

    submission['birds'].extend(['nocall' for i in range(data.shape[0])])

    del wav,sr

for _, row in sub_test_3.iterrows():

    row_id, audio_id = row['row_id'], row['audio_id']

    wav,sr=librosa.load(BASE_TEST_DIR+'/test_audio/'+ audio_id+'.mp3',sr=None,duration=1)

    st.add(sr)

    submission['row_id'].append(row_id)

    submission['birds'].append('nocall')

    del wav,sr
submission = pd.DataFrame(submission)

submission.head()
if len(st)==2 : pass

elif len(st)==3 : submission.birds='amecro'                 

elif len(st)==4 : submission.birds='amebit'

elif len(st)==5 : submission.birds='nocall amecro'

else : submission.birds='redhea'
submission.to_csv('submission.csv', index=False)
submission