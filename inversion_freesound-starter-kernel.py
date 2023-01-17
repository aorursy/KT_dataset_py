# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import IPython.display as ipd  # To play sound in the notebook
from tqdm import tqdm_notebook
import wave

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
audio_train_files = os.listdir('../input/audio_train')
audio_test_files = os.listdir('../input/audio_test')

train = pd.read_csv('../input/train.csv', index_col='fname')
submission = pd.read_csv('../input/sample_submission.csv', index_col='fname')
train.head()
submission.head()
ipd.Audio('../input/audio_train/' + '00044347.wav') # Hi-hat
ipd.Audio('../input/audio_train/' + '001ca53d.wav') # Saxophone
ipd.Audio('../input/audio_train/' + '00c82919.wav') # Can you guess?
train['nframes'] = 0

for e, fname in enumerate(tqdm_notebook(train.index)):
    try:
        w = wave.open('../input/audio_train/' + fname)
        p = w.getparams()
        train.loc[fname, 'nframes'] = p.nframes
    except:
        print(f'Failed: {e} - {fname}')
_, ax = plt.subplots(figsize=(16, 4))
sns.violinplot(ax=ax, x="label", y="nframes", data=train)
plt.xticks(rotation=90)
plt.title('Distribution of audio frames, per label', fontsize=16)
plt.show()
test = pd.DataFrame(index=submission.index, columns=['nframes'], data=0)

for e, fname in enumerate(tqdm_notebook(test.index)):
    try:
        w = wave.open('../input/audio_test/' + fname)
        p = w.getparams()
        test.loc[fname, 'nframes'] = p.nframes
    except:
        print(f'Failed: {e} - {fname}')
test.head()
clf = LogisticRegression()
clf.fit(train['nframes'].values.reshape(-1,1), train['label'].values)
preds = clf.predict_proba(test.values)
top_3 = clf.classes_[np.argsort(preds, axis=1)[:, -3:]]
submission['label'] = [' '.join(list(x)) for x in top_3]
submission.sample(10)
submission.to_csv('audio_frame_lr.csv')



