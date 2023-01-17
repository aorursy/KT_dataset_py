import numpy as np

import pandas as pd

import wave, IPython

from sklearn import *

from scipy.io import wavfile

import gc; gc.enable()



train = pd.read_csv('../input/train_curated.csv')

trainn = pd.read_csv('../input/train_noisy.csv')

test = pd.read_csv('../input/sample_submission.csv')

train.shape, trainn.shape, test.shape
train['path'] = train['fname'].map(lambda x: '../input/train_curated/'+x)

trainn['path'] = trainn['fname'].map(lambda x: '../input/train_noisy/'+x)

test['path'] = test['fname'].map(lambda x: '../input/test/'+x)



train['noisy'] = 0; trainn['noisy'] = 1

train = pd.concat((train, trainn), sort=False).reset_index(drop=True)



labels = [c for c in test.columns if c not in ['path','fname']]

train = train[train['labels'].isin(labels)].reset_index(drop=True)

test = test[['path', 'fname']]

train.shape, test.shape
norm_labels = []

cut = 100

for l in train.labels.unique():

    norm_labels.append(train[train.labels == l][:cut])

train = pd.concat(norm_labels, sort=False).sample(frac=1).reset_index(drop=True)

train.shape
waves = """3a5b14ee.wav 404712 423984

7a9cf335.wav 501072 520344

c421d4a2.wav 289080 308352

aa28de21.wav 231264 250536

703ac398.wav 19272 38544

3cbb9c24.wav 57813 77084

7c20368d.wav 616672 635943

c6cb06d9.wav 481775 501046

7f0af3bb.wav 481775 501046

76caa793.wav 385420 404691

767b8f3a.wav 635943 655214

a98c3157.wav 231252 250523

8ddb4c26.wav 0 19271

3e1d0af4.wav 635943 655214

aca0ce49.wav 578130 597401""".split('\n')

wavesc = []

for w in waves:

    w1, c1, c2 = w.split(' ')

    c1 = int(c1); c2 = int(c2)

    nrate, ndata = wavfile.read('../input/train_noisy/'+w1)

    wavesc.append(ndata[c1:c2])

wavesc = np.concatenate(wavesc)

wavfile.write('one_step.wav', 44100, wavesc)

IPython.display.display(IPython.display.Audio('one_step.wav'))
def get_short_wave(w, size=1200):

    rate, data = wavfile.read(w)

    if len(data) > size:

        return data[:size]

    else:

        print(int((size / len(data))+1), len(data))

        return np.repeat(data, int((size / len(data))+1))[:size]
train['nframes'] = train['path'].map(lambda x: wave.open(x).getnframes())

train['short_wave'] = train['path'].map(lambda x: get_short_wave(x))



test['nframes'] = test['path'].map(lambda x: wave.open(x).getnframes())

test['short_wave'] = test['path'].map(lambda x: get_short_wave(x))
def features(df, col='short_wave'):

    for agg in ['min', 'max', 'sum', 'median', 'mean', 'std', 'skew', 'kurtosis']:

        df[col+agg] = df[col].map(lambda x: eval('pd.DataFrame(x).' + agg + '(axis=0)')[0])

        df[col+'a'+agg] = df[col].map(lambda x: eval('pd.DataFrame(x).abs().' + agg + '(axis=0)')[0])

        

    df[col+'max_diff'] = df[col+'max'] - df[col+'mean']

    df[col+'amax_diff'] = df[col+'amax'] - df[col+'amean']

    

    df[col+'min_diff'] = df[col+'mean'] - df[col+'min']

    df[col+'amin_diff'] = df[col+'amean'] - df[col+'amin']

    

    df[col+'max_diff2'] = df[col+'max'] - df[col+'median']

    df[col+'amax_diff2'] = df[col+'amax'] - df[col+'amedian']

    

    df[col+'min_diff2'] = df[col+'median'] - df[col+'min']

    df[col+'amin_diff2'] = df[col+'amedian'] - df[col+'amin']

    return df



train = features(train).fillna(-999)

test = features(test).fillna(-999)

print(train.shape, test.shape)
col = [c for c in train.columns if c not in ['path','fname', 'noisy', 'labels', 'short_wave']]

le = preprocessing.LabelEncoder()

train['labels'] = le.fit_transform(train['labels'])



clf1 = ensemble.ExtraTreesClassifier(n_jobs=-1, n_estimators=400, max_features=0.9, random_state=10)

clf2 = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=400, max_features=0.9, random_state=9)



split = 3000

clf1.fit(train[col][:split], train['labels'][:split])

clf2.fit(train[col][:split], train['labels'][:split])

def LOL_WRAP(y_true, y_score): return metrics.label_ranking_average_precision_score(y_true, y_score)



print('ETR LOL_WRAP', LOL_WRAP(pd.get_dummies(train['labels'])[split:], clf1.predict_proba(train[col][split:])))

print('RFR LOL_WRAP', LOL_WRAP(pd.get_dummies(train['labels'])[split:], clf2.predict_proba(train[col][split:])))



clf1.fit(train[col], train['labels'])

clf2.fit(train[col], train['labels'])



sub = clf1.predict_proba(test[col])

sub += clf2.predict_proba(test[col])

sub /= 2



sub = pd.DataFrame(sub, columns=le.classes_)

sub['fname'] = test['fname']

missing = [c for c in labels if c not in sub.columns]

for c in missing:

    sub[c] = 0.0

sub.to_csv('submission.csv', index=False)