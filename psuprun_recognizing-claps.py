# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import librosa



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
path = "../input/"

files = os.listdir(path)

files = [path+x for x in files]

files
sample_rate = 22050 # librosa default

data = [x[0] for x in map(librosa.core.load, files)]

#data = [librosa.core.load(x, sr=sample_rate)[0] for x in files]
df = pd.DataFrame(data=files)

df['data'] = data

classes = df[0].str.extract("\/([a-z]*)[0-9]*.wav$",expand=False).str.strip()

df['class'] = classes

df['is_clap'] = (df['class']=='claps').map(int)

df.columns = ['file','data','class','is_clap']

df.head()
test = df[df['class']=='test'].iloc[0]

df = df[df['class']!='test']
import matplotlib.pyplot as plt

import numpy.fft as fft



fft_len = sample_rate // 10 # I will analyze 0.1s samples

half_fft_len = fft_len // 2 # fft is symmetric, so we do not need 2nd half

groups_num = 25 # I will use 25 frequency intervals



def get_coefs(data, minx): # will transform 0.1s sample of audiofile, starting from minx-th frame

    maxx = minx + fft_len

    data = data[minx:maxx]

    

    FFT = fft.fft(data)

    

    FFT = FFT[:half_fft_len] # throw away 2nd half, which is symmetric to the 1st

    coefs = [sum(abs(FFT[x : x+half_fft_len//groups_num])) # sum frequencies in intervals 

             for x in range(0,len(FFT), half_fft_len//groups_num )]

    

    return np.array(coefs)
x = np.array([np.array(x)[:len(x)-len(x)%fft_len] for x in df['data']]) # leave only whole 0.1s intervals

y = df['is_clap'].values
xs = []

ys = []



for i in range(len(x)): # for each audiofile

    xx = x[i] # i-th audiofile

    yy = y[i] # i-th class

    

    # for each 0.1ms interval

    for minx in range(0,len(xx),fft_len):

        

        coefs = get_coefs(data=xx, minx=minx) # extract features

        

        # add features to list

        xs += [coefs] 

        ys += [yy]

        

        

xs = np.array(xs)

ys = np.array(ys)
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical



ys = to_categorical(ys)

x_train,x_test,y_train,y_test = train_test_split(xs,ys,test_size=0.3,)
from keras import Sequential

from keras.layers import Dense

from keras.optimizers import Adam



# Regular MLP

classifier = Sequential([

    Dense(100, activation='relu'),

    Dense(2, activation='softmax')

])

classifier.compile(optimizer='sgd', loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(x_train, y_train)
y_test_num = np.argmax(y_test, axis=1)

print('total accuracy:',sum(classifier.predict_classes(x_test)==y_test_num)/len(y_test))

print('false negatives',sum(classifier.predict_classes(x_test[y_test_num==0]))/len(y_test))

print('true positives',sum(classifier.predict_classes(x_test[y_test_num==1]))/len(y_test))
classifier.save('recognize_clap_model_test.h5')

del classifier

from keras.models import load_model

classifier = load_model('recognize_clap_model_test.h5')

classifier.predict(x_test)
test_arr = test['data']

test_arr = test_arr[:len(test_arr)-len(test_arr)%fft_len]



# recognize all 0.1ms intervals with claps

claps = []

for minx in range(0,len(test_arr),fft_len):

        c = get_coefs(data=test_arr, minx=minx)

        p = classifier.predict_classes(np.array([c]))

        if (p==1):

            claps+=[minx//fft_len]

test_arr_claps = [x for x in range(len(test_arr)) if int(x/fft_len) in claps] # mark all indexes with claps





plt.figure(figsize=(14,8))



xticks = np.arange(0,len(test_arr)/sample_rate, 1/sample_rate) # xticks to seconds scale

line1 = plt.plot(xticks,test_arr, linewidth=1, c='grey', alpha=0.5)



line2 = plt.scatter([x/sample_rate for x in test_arr_claps], # xticks to seconds scale

            np.zeros(len(test_arr_claps)), 

            c='red', s=50)



plt.axes().xaxis.set_ticks(np.arange(0,len(test_arr)/sample_rate,1.0))

plt.grid(alpha=0.3)

plt.title(test['file'])

plt.legend((line1,line2),('audio','claps'))



plt.show()
model = Sequential([

    Dense(100, activation='relu'),

    Dense(2, activation='softmax')

])

model.compile(optimizer='sgd', loss='binary_crossentropy',metrics=['accuracy'])

model.fit(xs, ys)
model.save('recognize_clap_model.h5')