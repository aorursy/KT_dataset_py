# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import tensorflow as tf
import pickle
import librosa
classes = os.listdir("../input/train/audio/")
print(classes)
def create_data():
    x = []
    y = []
    for c in classes:
        try:
            tmpx = []
            tmpy = []
            print('processs-...', c)

            for file in os.listdir('../input/train/audio/' + c):
                wave,sr = librosa.load('../input/train/audio/' + c +'/' + file, mono=True)
                mfcc = librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=20)
                if mfcc.shape == (20, 44):
                    x.append(mfcc)
                    tmpx.append(mfcc)
                    y.append(classes.index(c))
                    tmpy.append(classes.index(c))

            print('write file pickle ', c)
            pickle.dump(np.array(tmpx), open('{}.pickle'.format(c), 'wb'))
            pickle.dump(np.array(tmpy), open('{}_y.pickle'.format(c), 'wb'))
        except:
            pass
            

    print('complete')
    return np.array(x), np.array(y)
        
x, y = create_data()
        

