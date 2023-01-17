"""

Copyright 2020 Stanislav Dereka



Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:



The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import pylab

pylab.rcParams['figure.figsize'] = (20, 5)

from sklearn.linear_model import LinearRegression

from itertools import combinations

from seaborn import distplot

import seaborn as sns

sns.set()



batches_train = [

    slice(100000*0, 100000*5),

    slice(100000*5, 100000*10),

    slice(100000*10, 100000*15),

    slice(100000*15, 100000*20),

    slice(100000*20, 100000*25),

    slice(100000*25, 100000*30),

    slice(100000*30, 100000*35),

    slice(100000*35, 100000*40),

    slice(100000*40, 100000*45),

    slice(100000*45, 100000*50),

]



groups_train = [0, 0, 1, 2, 3, 4, 1, 2, 4, 3]



batches_test = [

    slice(100000*0, 100000*1),

    slice(100000*1, 100000*2),

    slice(100000*2, 100000*3),

    slice(100000*3, 100000*4),

    slice(100000*4, 100000*5),

    slice(100000*5, 100000*6),

    slice(100000*6, 100000*7),

    slice(100000*7, 100000*8),

    slice(100000*8, 100000*9),

    slice(100000*9, 100000*10),

    slice(100000*10, 100000*15),

    slice(100000*15, 100000*20),

]



groups_test = [0, 2, 4, 5, 1, 3, 4, 3, 5, 2, 5, 5]



def label_batches_and_groups(data, batches, groups):

    data.loc[:, 'batch'] = np.empty(len(data), dtype=np.int)

    data.loc[:, 'group'] = np.empty(len(data), dtype=np.int)



    for i in range(len(batches)):

        b = batches[i]

        g = groups[i]

        data.loc[b, 'batch'] = i

        data.loc[b, 'group'] = g



test = pd.read_csv("../input/data-without-drift/test_clean.csv")

train = pd.read_csv("../input/data-without-drift/train_clean.csv")

label_batches_and_groups(train, batches_train, groups_train)

label_batches_and_groups(test, batches_test, groups_test)



res = 1000

plt.figure(figsize=(20, 5))

plt.plot(train.time.values[::res], train.signal.values[::res])

plt.plot(train.time.values[::res], train.open_channels.values[::res])

for b in batches_train:

    t = train.time.values[b]

    plt.plot([t[0],t[0]],[-10,8],'g')

    plt.text(t[0]+4,6,str(batches_train.index(b)),size=13)

plt.show()



res = 1

plt.plot(test.time.values[::res], test.signal.values[::res])

for b in batches_test:

    t = test.time.values[b]

    plt.plot([t[0],t[0]],[-10,8],'g')

    plt.text(t[0]+4,6,str(batches_test.index(b)),size=13)

plt.show()
corrupted = slice(3_640_000, 3_840_000)

healthy = slice(1_500_000, 1_700_000)



cleaned = train.drop(train[corrupted].index)

signal = cleaned[cleaned.group != 3].signal.values

channels = cleaned[cleaned.group != 3].open_channels.values





# https://www.kaggle.com/kakoimasataka/remove-pick-up-electric-noise

c = 6

label = np.arange(len(signal))



channel_list = np.arange(c)

n_list = np.empty(c)

mean_list = np.empty(c)

std_list = np.empty(c)

stderr_list = np.empty(c)



for i in range(c):

    x = label[channels == i]

    y = signal[channels == i]

    n_list[i] = np.size(y)

    mean_list[i] = np.mean(y)

    std_list[i] = np.std(y)



stderr_list = std_list / np.sqrt(n_list)

plt.show()



w = 1 / stderr_list

channel_list = channel_list.reshape(-1, 1)

linreg_m = LinearRegression()

linreg_m.fit(channel_list, mean_list, sample_weight=w)



mean_predict = linreg_m.predict(np.arange(0, 11).reshape(-1, 1))



x = np.linspace(-0.5, 10, 5)

y = linreg_m.predict(x.reshape(-1, 1))

plt.figure(figsize = (10, 10))

plt.plot(x, y, label="regression")

plt.plot(channel_list, mean_list, ".", markersize=8, label="original")

plt.legend()

plt.show()



print("mean:", mean_predict)
noise = train[train.batch == 3].signal.values - mean_predict[train[train.batch == 3].open_channels.values]



# https://www.kaggle.com/kakoimasataka/remove-pick-up-electric-noise

fs=10000.

fig, ax = plt.subplots(nrows=1, ncols=1)

fig.subplots_adjust(hspace = .5)



fft = np.fft.fft(noise)

psd = np.abs(fft) ** 2

fftfreq = np.fft.fftfreq(len(psd),1/fs)



i = abs(fftfreq) < 200

ax.grid()

ax.plot(fftfreq[i], 20*np.log10(psd[i]), linewidth=.5)

ax.set_xlabel('Frequency (Hz)') 

ax.set_ylabel('PSD (dB)')

plt.show()
healthy_noise = train[healthy].signal.values - mean_predict[train[healthy].open_channels.values]

fixed = mean_predict[train[corrupted].open_channels.values] + healthy_noise

train.loc[train[corrupted].index, 'signal'] = fixed
def compose(data_1, data_2, means, noise_factor):

    ch_1 = data_1.open_channels.values

    ch_2 = data_2.open_channels.values

    comp_label = ch_1 + ch_2



    noise_1 = data_1.signal.values - means[ch_1]

    noise_2 = data_2.signal.values - means[ch_2]

    noise = (noise_1 + noise_2) / noise_factor



    comp = means[comp_label] + noise

    return comp, comp_label





def combinatorial_synthesis(data, n, flip, **params):

    assert len(data) % n == 0

    l_s = len(data) // n

    comb = combinations(list(range(n)), 2)

    for i, j in comb:

        sig, ch = compose(data[i*l_s:(i+1)*l_s], data[j*l_s:(j+1)*l_s], **params)

        yield sig, ch

        if flip:

            sig, ch = compose(data[i * l_s:(i + 1) * l_s], data[j * l_s:(j + 1) * l_s][::-1], **params)

            yield sig, ch





def append_dataset(data, signal, channels, group):

    t_0 = data.time.values[-1]

    b = data.batch.values[-1]

    tau = 0.0001

    time = np.arange(t_0 + tau, t_0 + tau * (len(signal) + 1), tau)

    new = pd.DataFrame()

    new['time'] = time

    new['signal'] = signal

    new['open_channels'] = channels

    new['batch'] = b + 1

    new['group'] = group

    return pd.concat([data, new], ignore_index=True, axis=0)
cs1 = combinatorial_synthesis(train[train.group == 0], 4, flip=False, means=mean_predict, noise_factor=2 ** 0.5)

for sig, ch in cs1:

    train = append_dataset(train, sig, ch, 5)



plt.plot(test[test.group == 5].signal.values[:10000])

plt.title('Group 5 original')

plt.show()



plt.plot(train[train.group == 5].signal.values[-10000:])

plt.title('Group 5 synthetic')

plt.show()

    

distplot(train[train.group == 5].signal.values, label='Group 5 synthetic')

distplot(test[test.group == 5].signal.values, label='Group 5 original')

plt.legend()

plt.show()
cs2 = combinatorial_synthesis(train[train.group == 4], 10, flip=False, means=mean_predict, noise_factor=1.0)

for sig, ch in cs2:

    train = append_dataset(train, sig, ch, 3)



new = train.batch >= len(batches_train)

mean_new = train[new & (train.group == 3)].signal.mean()



to_be_fixed = (train.batch == 4) | (train.batch == 9)



for b in [4, 9]:

    train.loc[train.batch == b, 'signal'] = train[train.batch == b].signal.values - train[

        train.batch == b].signal.values.mean() + mean_new



for b in [5, 7]:

    test.loc[test.batch == b, 'signal'] = test[test.batch == b].signal.values - test[

        test.batch == b].signal.values.mean() + mean_new



plt.plot(train[train.group == 3].signal.values[:10000])

plt.title('Group 3 original')

plt.show()



plt.plot(train[train.group == 3].signal.values[-10000:])

plt.title('Group 3 synthetic')

plt.show()



distplot(train[10_000_000:].signal.values, label='Group 3 original')

distplot(test[test.group == 3].signal.values, label='Group 3 synthetic')

plt.legend()

plt.show()
noise = train[train.batch == 28].signal.values - mean_predict[train[train.batch == 28].open_channels.values]



# https://www.kaggle.com/kakoimasataka/remove-pick-up-electric-noise

fs=10000.

fig, ax = plt.subplots(nrows=1, ncols=1)

fig.subplots_adjust(hspace = .5)



fft = np.fft.fft(noise)

psd = np.abs(fft) ** 2

fftfreq = np.fft.fftfreq(len(psd),1/fs)



i = abs(fftfreq) < 200

ax.grid()

ax.plot(fftfreq[i], 20*np.log10(psd[i]), linewidth=.5)

ax.set_xlabel('Frequency (Hz)') 

ax.set_ylabel('PSD (dB)')

plt.show()
train.signal.plot()

plt.title('Train')

plt.show()



test.signal.plot()

plt.title('Test')

plt.show()



train.to_csv("train_synthetic.csv", index=False, float_format='%.4f')

test.to_csv("test_synthetic.csv", index=False, float_format='%.4f')
def rescale_noise(data, means, scale_factor):

    sig = data.signal.values

    ch = data.open_channels.values

    noise = sig - means[ch]

    noise *= scale_factor

    sig_ = noise + means[ch]

    return sig_





def reduce_channels(data, res, means):

    residual = res.open_channels.values

    reduced_sig = data.signal.values - (means - means[0])[residual]

    return reduced_sig, residual
reduced_sig, residual = reduce_channels(train[train.batch == 35], train[train.group == 4][500_000:600_000], mean_predict)

sig = rescale_noise(train[train.group == 4], mean_predict, 2 ** 0.5)
distplot(sig, label='group 4 signal with rescaled noise')

distplot(reduced_sig, label='group 3 reduced signal')

plt.legend()

plt.show()