import pandas as pd
pd.options.display.precision = 15
train = pd.read_csv('../input/train.csv', dtype={
    'acoustic_data': 'int16',
    'time_to_failure': 'float64',
}, nrows=200000000)
train[::1000].plot(secondary_y=['time_to_failure'], figsize=(30,5))
train[:50000].plot(secondary_y=['time_to_failure'], figsize=(30,5))
train.iloc[:8000].plot(secondary_y=['time_to_failure'], figsize=(5, 30))
a, b = train.time_to_failure[4094:4096]
print('Interval between chunks: %.9f\nInterval between points: %.9f' % (a - b, train.time_to_failure[0] - a))
from IPython.display import display, Audio
Audio(train.acoustic_data[:150000], rate=2000)
failures = train.index[train.time_to_failure.shift(1) < train.time_to_failure]
spike = train[:failures[0]].acoustic_data.abs().argmax()
d = train[spike - 100000:failures[0]]
d.plot(figsize=(30, 5), secondary_y=['time_to_failure'])
display(Audio(d.acoustic_data, rate=4000))
import librosa
import librosa.display
X = librosa.stft(d.acoustic_data.astype('float16').values)
Xdb = librosa.amplitude_to_db(abs(X))
from matplotlib import pylab as plt
plt.figure(figsize=(28, 10))
librosa.display.specshow(Xdb, x_axis='time', y_axis='hz')
import os
test = []
for i in os.listdir('../input/test'):
    test.append((i, pd.read_csv('../input/test/' + i, dtype={'acoustic_data': 'int16'})))
for name, i in test:
    if i.acoustic_data.max() > 1000:
        display(name)
        display(Audio(i.acoustic_data, rate=4000))
        i.plot()
        plt.show()
for i in range(10):
    display(test[i][0])
    display(Audio(test[i][1].acoustic_data, rate=4000))
    test[i][1].plot()
    plt.show()
