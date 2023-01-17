import os



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from tqdm.notebook import tqdm

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report



from scipy.stats import norm
train = pd.read_csv('../input/ghost-drift-and-outliers/train_clean.csv')

test  = pd.read_csv('../input/ghost-drift-and-outliers/test_clean.csv')

orig_train = pd.read_csv('../input/liverpool-ion-switching/train.csv')
class ViterbiClassifier:

    def __init__(self):

        self._p_trans = None

        self._p_signal = None

    

    def fit(self, x, y):

        self._states = np.unique(y)

        self._n_states = len(self._states)

        

        self._p_trans = self.markov_p_trans(y)

        

        self._dists = []

        for s in np.arange(y.min(), y.max() + 1):

            self._dists.append((np.mean(x[y == s]), np.std(x[y == s])))

        

        return self

        

    def predict(self, x, p_signal=None, proba=False):

        if p_signal is None:

            p_signal = self.markov_p_signal(x)



        preds, probs = self.viterbi(self._p_trans, p_signal[self._states], x)

        

        if proba:

            return probs

        else:

            return preds

    

    def markov_p_signal(self, signal):

        p_signal = np.zeros((self._n_states, len(signal)))

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

    

        return x, T1

    

class PosteriorDecoder:

    def __init__(self):

        self._p_trans = None

        self._p_signal = None

    

    def fit(self, x, y):

        self._states = np.unique(y)

        self._n_states = len(self._states)

        

        self._dists = []

        for s in np.arange(y.min(), y.max() + 1):

            self._dists.append((np.mean(x[y == s]), np.std(x[y == s])))

        

        self._p_trans = self.markov_p_trans(y)

        

        return self

        

    def predict(self, x, p_signal=None, proba=False):

        if p_signal is None:

            p_signal = self.markov_p_signal(x)

        preds = self.posterior_decoding(self._p_trans, p_signal[self._states])

        

        if proba:

            return probs

        else:

            return preds

    

    def markov_p_signal(self, signal):

        p_signal = np.zeros((self._n_states, len(signal)))

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

    

    def forward(self, p_trans, p_signal):

        """Calculate the probability of being in state `k` at time `t`, 

           given all previous observations `x_1 ... x_t`"""

        T1 = np.zeros(p_signal.shape)

        T1[:, 0] = p_signal[:, 0]

        T1[:, 0] /= np.sum(T1[:, 0])



        for j in range(1, p_signal.shape[1]):

            for i in range(len(p_trans)):

                T1[i, j] = p_signal[i, j] * np.sum(T1[:, j - 1] * p_trans[i, :])

            T1[:, j] /= np.sum(T1[:, j])



        return T1



    def backward(self, p_trans, p_signal):

        """Calculate the probability of observing `x_{t + 1} ... x_n` if we 

           start in state `k` at time `t`."""

        T1 = np.zeros(p_signal.shape)

        T1[:, -1] = p_signal[:, -1]

        T1[:, -1] /= np.sum(T1[:, -1])



        for j in range(p_signal.shape[1] - 2, -1, -1):

            for i in range(len(p_trans)):

                T1[i, j] = np.sum(T1[:, j + 1] * p_trans[:, i] * p_signal[:, j + 1])

            T1[:, j] /= np.sum(T1[:, j])



        return T1

    

    def posterior_decoding(self, p_trans, p_signal):

        fwd = self.forward(p_trans, p_signal)

        bwd = self.backward(p_trans, p_signal)



        x = np.empty(p_signal.shape[1], 'B')

        for i in range(p_signal.shape[1]):

            x[i] = np.argmax(fwd[:, i] * bwd[:, i])



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
# train['block'] = np.NaN

# for model, ix in enumerate(blocks):

#     train.loc[ix, 'block'] = model

# distributions = train.groupby(['block', 'open_channels'])['signal'].agg(['mean', 'std'])

# distributions
true_state = train.open_channels.values

signal = train.signal.values
models = []

viterbi_predictions = np.zeros(len(signal))

pos_dec_predictions = np.zeros(len(signal))

for i, ix in enumerate(blocks):

    sub_signal = signal[ix]

    

    viterbi = ViterbiClassifier().fit(sub_signal, true_state[ix])

    viterbi_predictions[ix] = viterbi.predict(sub_signal)

    

    decoder = PosteriorDecoder().fit(sub_signal, true_state[ix])

    pos_dec_predictions[ix] = decoder.predict(sub_signal)

    models.append(decoder)

    

    print('[Model #{}] || Pos. Dec.: F1 = {}'.format(

        i, 

        f1_score(y_pred=viterbi_predictions[ix], y_true=true_state[ix], average='macro'),

        f1_score(y_pred=pos_dec_predictions[ix], y_true=true_state[ix], average='macro')

    ))

    

    print(classification_report(y_pred=pos_dec_predictions[ix], y_true=true_state[ix]))
print("[Viterbi] Total Accuracy =", accuracy_score(y_pred=viterbi_predictions, y_true=true_state))

print("[Viterbi] Total F1 (macro) =", f1_score(y_pred=viterbi_predictions, y_true=true_state, average='macro'))



print("[Posterior Decoding] Total Accuracy =", accuracy_score(y_pred=pos_dec_predictions, y_true=true_state))

print("[Posterior Decoding] Total F1 (macro) =", f1_score(y_pred=pos_dec_predictions, y_true=true_state, average='macro'))



print(classification_report(y_pred=pos_dec_predictions, y_true=true_state))
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

df_subm.to_csv("forward_backward.csv", float_format='%.4f', index=False)
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