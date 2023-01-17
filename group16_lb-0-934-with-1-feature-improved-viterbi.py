import os



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from tqdm.notebook import tqdm

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix



from scipy.stats import norm
train = pd.read_csv('../input/ghost-drift-and-outliers/train_clean_kalman.csv')

test  = pd.read_csv('../input/ghost-drift-and-outliers/test_clean_kalman.csv')
class ViterbiClassifier:

    def __init__(self):

        self._p_trans = None

        self._p_signal = None

        self._p_in = None

    

    def fit(self, x, y):

        self._p_trans = self.markov_p_trans(y)

        self._dists = []

        self._states = len(np.unique(y))

        for s in np.arange(y.min(), y.max() + 1):

            self._dists.append((np.mean(x[y == s]), np.std(x[y == s])))

        

        return self

        

    def predict(self, x):

        p_signal = self.markov_p_signal(x)

        return self.viterbi(self._p_trans, p_signal, x)

    

    def markov_p_signal(self, signal):

        p_signal = np.zeros((self._states, len(signal)))

        for k, dist in enumerate(self._dists):

            p_signal[k, :] = norm.pdf(signal, *dist)

            

        return p_signal

    

    def markov_p_trans(self, states):

        # https://www.kaggle.com/friedchips/the-viterbi-algorithm-a-complete-solution

        max_state = np.max(states)

        states_next = np.roll(states, -1)

        matrix = []

        for i in range(max_state + 1):

            current_row = np.histogram(states_next[states == i], bins=np.arange(max_state + 2))[0]

            if np.sum(current_row) == 0: # if a state doesn't appear in states...

                current_row = np.ones(max_state + 1) / (max_state + 1) # ...use uniform probability

            else:

                current_row = current_row / np.sum(current_row) # normalize to 1

            matrix.append(current_row)

        return np.array(matrix)

    

    def viterbi(self, p_trans, p_signal, signal):

        # https://www.kaggle.com/friedchips/the-viterbi-algorithm-a-complete-solution

        offset = 10**(-20) # added to values to avoid problems with log2(0)



        p_trans_tlog  = np.transpose(np.log2(p_trans  + offset)) # p_trans, logarithm + transposed

        p_signal_tlog = np.transpose(np.log2(p_signal + offset)) # p_signal, logarithm + transposed

        

        T1 = np.zeros(p_signal.shape)

        T2 = np.zeros(p_signal.shape)



        T1[:, 0] = p_signal_tlog[0, :]

        T2[:, 0] = 0



        for j in range(1, p_signal.shape[1]):

            for i in range(len(p_trans)):

                T1[i, j] = np.max(T1[:, j - 1] + p_trans_tlog[:, i] + p_signal_tlog[j, i])

                T2[i, j] = np.argmax(T1[:, j - 1] + p_trans_tlog[:, i] + p_signal_tlog[j, i])



        x = np.empty(p_signal.shape[1], 'B')

        x[-1] = np.argmax(T1[:, p_signal.shape[1] - 1])

        for i in reversed(range(1, p_signal.shape[1])):

            x[i - 1] = T2[x[i], i]

    

        return x
train['batch'] = (train['time'] - 0.0001) // 50

counts = train.groupby('batch').count()['time'].values

models = [0, 0, 1, 2, 4, 3, 1, 2, 3, 4]

blocks = [[], [], [], [], []]

total = 0

for model, count in zip(models, counts):

    blocks[model].extend(list(range(total, total + count)))

    total += count

print([len(x) for x in blocks])
true_state = train.open_channels.values

signal = train.signal.values
# Let's show the (gaussian) distributions of the signals

f, ax = plt.subplots(1, len(blocks), figsize=(20, 5))

for i, ix in enumerate(blocks):

    for label in set(true_state[ix]):

        pd.Series(signal[ix][true_state[ix] == label]).plot(kind='hist', ax=ax[i], 

                                                            alpha=0.5, label=label)

    ax[i].set_title('Data #{}'.format(i))

    ax[i].legend()



plt.show()
models = []

train_predictions = np.zeros(len(signal))

for i, ix in enumerate(blocks):

    sub_signal = signal[ix]

    viterbi = ViterbiClassifier().fit(sub_signal, true_state[ix])

    models.append(viterbi)

    

    train_predictions[ix] = viterbi.predict(sub_signal)

    print('[Model #{}] F1 (macro) = {}'.format(i, f1_score(y_pred=train_predictions[ix], y_true=true_state[ix], average='macro')))
print("Total Accuracy =", accuracy_score(y_pred=train_predictions, y_true=true_state))

print("Total F1 (macro) =", f1_score(y_pred=train_predictions, y_true=true_state, average='macro'))



# Total Accuracy = 0.9670930385544279

# Total F1 (macro) = 0.9359432322559637

test_blocks = [

    list(range(0, 100000)) + list(range(300000, 400000)) + list(range(800000, 900000)) + list(range(1000000, 2000000)),

    list(range(400000, 500000)),

    list(range(100000, 200000)) + list(range(900000, 1000000)),

    list(range(200000, 300000)) + list(range(600000, 700000)),

    list(range(500000, 600000)) + list(range(700000, 800000))

]



# Sanity check

assert sum([len(x) for x in test_blocks]) == 2000000
df_subm = pd.read_csv("../input/liverpool-ion-switching/sample_submission.csv")

for i, ix in enumerate(test_blocks):

    df_subm.loc[ix, 'open_channels'] = models[i].predict(test.signal.values[ix])

df_subm.to_csv("viterbi.csv", float_format='%.4f', index=False)
# Sanity check 

# https://www.kaggle.com/cdeotte/one-feature-model-0-930

plt.figure(figsize=(20,5))

res = 1000; let = ['A','B','C','D','E','F','G','H','I','J']

plt.plot(range(0,test.shape[0],res),df_subm.open_channels[0::res])

for i in range(5): plt.plot([i*500000,i*500000],[-5,12.5],'r')

for i in range(21): plt.plot([i*100000,i*100000],[-5,12.5],'r:')

for k in range(4): plt.text(k*500000+250000,10,str(k+1),size=20)

for k in range(10): plt.text(k*100000+40000,7.5,let[k],size=16)

plt.title('Test Data Predictions',size=16)

plt.show()