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
mylist=os.listdir('../input/emotion-classification-speach/song/Actor_01')
mylist
import librosa

import librosa.display

data, sampling_rate=librosa.load("../input/emotion-classification-speach/song/Actor_01/03-02-01-01-01-01-01.wav")


import os

import pandas as pd

import librosa

import glob 

import numpy as np

import matplotlib.pyplot as plt

plt.figure(figsize=(15,5))

librosa.display.waveplot(data, sr=sampling_rate)
import scipy.io.wavfile

import numpy as np

import sys





sr,x = scipy.io.wavfile.read('../input/emotion-classification-speach/song/Actor_01/03-02-01-01-01-01-01.wav')



## Parameters: 10ms step, 30ms window

nstep = int(sr * 0.01)

nwin  = int(sr * 0.03)

nfft = nwin



window = np.hamming(nwin)



## will take windows x[n1:n2].  generate

## and loop over n2 such that all frames

## fit within the waveform



nn = range(nwin, len(x), nstep)



X = np.zeros( (len(nn), nfft//2) )



for i,n in enumerate(nn):

    xseg = x[n-nwin:n]

    z = np.fft.fft(window * xseg, nfft)

    X[i,:] = np.log(np.abs(z[:nfft//2]))



plt.imshow(X.T, interpolation='nearest',

    origin='lower',

    aspect='auto')



plt.show()
feeling_list=[]

for item in mylist:

    if item[6:-16]=='02' and int(item[18:-4])%2==0:

        feeling_list.append('female_calm')

    elif item[6:-16]=='02' and int(item[18:-4])%2==1:

        feeling_list.append('male_calm')

    elif item[6:-16]=='03' and int(item[18:-4])%2==0:

        feeling_list.append('female_happy')

    elif item[6:-16]=='03' and int(item[18:-4])%2==1:

        feeling_list.append('male_happy')

    elif item[6:-16]=='04' and int(item[18:-4])%2==0:

        feeling_list.append('female_sad')

    elif item[6:-16]=='04' and int(item[18:-4])%2==1:

        feeling_list.append('male_sad')

    elif item[6:-16]=='05' and int(item[18:-4])%2==0:

        feeling_list.append('female_angry')

    elif item[6:-16]=='05' and int(item[18:-4])%2==1:

        feeling_list.append('male_angry')

    elif item[6:-16]=='06' and int(item[18:-4])%2==0:

        feeling_list.append('female_fearful')

    elif item[6:-16]=='06' and int(item[18:-4])%2==1:

        feeling_list.append('male_fearful')

    elif item[:1]=='a':

        feeling_list.append('male_angry')

    elif item[:1]=='f':

        feeling_list.append('male_fearful')

    elif item[:1]=='h':

        feeling_list.append('male_happy')

    #elif item[:1]=='n':

        #feeling_list.append('neutral')

    elif item[:2]=='sa':

        feeling_list.append('male_sad')
labels = pd.DataFrame(feeling_list)



labels[:25]


df = pd.DataFrame(columns=['feature'])

bookmark=0

for index,y in enumerate(mylist):

    if mylist[index][6:-16]!='01' and mylist[index][6:-16]!='07' and mylist[index][6:-16]!='08' and mylist[index][:2]!='su' and mylist[index][:1]!='n' and mylist[index][:1]!='d':

        X, sample_rate = librosa.load('../input/emotion-classification-speach/song/Actor_01'+y, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)

        sample_rate = np.array(sample_rate)

        mfccs = np.mean(librosa.feature.mfcc(y=X, 

                                            sr=sample_rate, 

                                            n_mfcc=13),

                        axis=0)

        feature = mfccs

        #[float(i) for i in feature]

        #feature1=feature[:135]

        df.loc[bookmark] = [feature]

        bookmark=bookmark+1
