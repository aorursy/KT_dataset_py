from scipy.signal import butter, detrend, filtfilt

from lini_read_bcidat import lini_read_bcidat

import numpy as np



# default params

dp = {'car': 1, 'ica': 0, 'local_tend': 1, 'norm': 1, 'filtro': []}

dw = [-300, 800]

dch = np.arange(10)





def obten_p300_bci2000(f, channels=dch, window=dw, paco=dp, decimation=False, decimation_samples=4):



    data = lini_read_bcidat(f)

    raw_signal, states_matrix, labels, paco['fs'] = data



    states = {}



    if decimation:

        idx = np.arange(0, raw_signal.shape[0], decimation_samples)



        raw_signal = raw_signal[idx, :]

        states_matrix = states_matrix[idx, :]



    for i, label in enumerate(labels):

        states[label] = states_matrix[:, i]



    vi = int(np.round(np.abs(window[0]) * paco['fs'] / 1000))

    vf = int(np.round(window[1] * paco['fs'] / 1000))



    t = np.linspace(window[0], window[1], vf + vi)

    if not paco['filtro']:

        low = 2 / paco['fs']

        high = 24 / paco['fs']



        b, a = butter(4, [low, high], btype='bandpass')

        paco['filtro'].append({'b': b, 'a': a})



    signal = raw_signal[:, channels].copy()

    n_muestras, n_canales = signal.shape



    if paco['car'] == 1:

        signal = detrend(signal.T, type='constant', bp=0)

        signal = signal.T

        signal = detrend(signal, type='constant', bp=0)



    if len(paco['filtro']) != 0:

        for item in paco['filtro']:

            signal = filtfilt(item['b'], item['a'], signal.T).T



    ind = (np.array(np.where(np.diff(states['StimulusCode']) > 0)) + 1).reshape(-1)

    ind = ind[np.where(ind <= n_muestras - vf)]



    letras = (np.where(np.diff(states['PhaseInSequence']) > 0))[0]

    letras = np.concatenate([letras[0::2], [letras[-1]]])



    data = np.zeros([ind.shape[0], int(vf + vi), n_canales])

    etiqueta = -np.ones((ind.shape[0], 2))



    for i, index in enumerate(ind):

        rowcol = states['StimulusCode'][index]



        if index - vi < 0:

            raise ValueError('tiempo inicial muy largo')



        data[i, :, :] = signal[index - vi:index + vf, :]



        if states['StimulusType'][index] == 1:

            etiqueta[i, :] = [1, rowcol]

        else:

            etiqueta[i, :] = [-1, rowcol]



    return (raw_signal, signal, data, etiqueta, ind, states, t, letras)
processed = obten_p300_bci2000('../input/akimpech/P300db/ACS/ACS001/ACSS001R01.dat', decimation=True)

raw_signal, signal, data, etiqueta, ind, states, t, letras = processed
import matplotlib.pyplot as plt



def plot_channel(t, data, etiquetas, channel):

    plt.title('Channel {}'.format(channel + 1))

    plt.plot(t, data[etiqueta[:, 0] == 1, :, channel].mean(axis=0), label='ERP')

    plt.plot(t, data[etiqueta[:, 0] == -1, :, channel].mean(axis=0), label='Background')

    plt.xlabel('Time [ms]')

    plt.ylabel('a.u')

    plt.legend(loc='lower left')
plt.figure(figsize=(20, 10))



for i in range(2):

    for j in range(5):

        plt.subplot(2, 5, i * 5 + j + 1)

        plot_channel(t, data, etiqueta, i * 5 + j)



plt.tight_layout()

plt.show()
import sklearn

import os





def load_train(train_path, window=[-300, 800]):

    data = None

    

    for filename in os.listdir(train_path):

        full = os.path.join(train_path, filename)

        

        if '.dat' in full:

            print('[+] Loading {}'.format(full))

            

            rt, st, da, et, it, sta, tt, lt = obten_p300_bci2000(full, decimation=True)

            

            if data is None:

                data = da

                t = tt

                etiquetas = et

            else:

                data = np.vstack([data, da])

                etiquetas = np.vstack([etiquetas, et])

                

    return data, etiquetas, t





def plot_confusion_matrix(cm, ax=None, cbar=True):

    # Code inspired in https://github.com/scikit-learn/scikit-learn/blob/fd237278e/sklearn/metrics/_plot/confusion_matrix.py

    

    labels = ['Background', 'ERP']

    

    im = ax.imshow(cm)

    cmap_min, cmap_max = im.cmap(0), im.cmap(256)

    

    if cbar:

        ax.figure.colorbar(im)

    

    ax.set_xticks(np.arange(len(labels)))

    ax.set_yticks(np.arange(len(labels)))

    

    ax.set_ylim((len(labels) - 0.5, -0.5))

    

    ax.set_xticklabels(labels)

    ax.set_yticklabels(labels)

    

    ax.set_ylabel('True label')

    ax.set_xlabel('Predicted label')

    

    thresh = (cm.max() - cm.min()) / 2

    

    for i in range(len(labels)):

        for j in range(len(labels)):

            color = cmap_max if cm[i, j] < thresh else cmap_min

            text = ax.text(j, i, cm[i, j], ha='center', va='center', color=color)
base_path = '../input/akimpech/P300db/ACS/ACS001/'



signal, etiquetas, t = load_train(base_path)
signal.shape
print('# of signals of type background: {}'.format(np.where(etiquetas[:, 0] == -1)[0].shape[0]))

print('# of signals of type erp: {}'.format(np.where(etiquetas[:, 0] == 1)[0].shape[0]))
import sklearn

import sklearn.svm

import sklearn.ensemble

import sklearn.model_selection





sss = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)

classifier = sklearn.svm.SVC(kernel='rbf', gamma='scale', class_weight='balanced', random_state=0)
def clf_confusion_matrix(signal, labels, clf_class, channels, kwargs):

    # Channels is an array containing the channels to plot

    # Classifier is the class to use as the classifier

    # kwargs are the parameters for the classifier as dictionary

    

    classifiers = []

    y_preds = {}



    for train_index, test_index in sss.split(signal, labels):

        train_X = signal[train_index, :, :]

        train_y = labels[train_index]



        test_X = signal[test_index, :, :]

        test_y = labels[test_index]



        for channel in channels:

            print('[+] Training classifier for channel {}...'.format(channel + 1), end='')

            classifier = clf_class(**kwargs)

            classifier.fit(train_X[:, :, channel], train_y)

            print(' Done')



            classifiers.append(('Channel {} classifier'.format(channel + 1), channel, classifier))



        test_erp = None

        test_nerp = None



        for feature, label in zip(test_X, test_y):

            feature = feature[np.newaxis, :, :]



            if label == 0:

                if test_nerp is None:

                    test_nerp = feature

                else:

                    test_nerp = np.vstack([test_nerp, feature])



                latest = test_nerp



            elif label == 1:

                if test_erp is None:

                    test_erp = feature

                else:

                    test_erp = np.vstack([test_erp, feature])



                latest = test_erp



            for label, channel, classifier in classifiers:

                if channel not in y_preds:

                    y_preds[channel] = []



                pred = classifier.predict(latest[:, :, channel].mean(axis=0).reshape(1, -1))[0]

                y_preds[channel].append(pred)



        # Plotting section

        plt.figure(figsize=(8, 4 * len(channels)))

        for i, channel in enumerate(channels):

            print('Plotting channel', channel)

            

            plt.subplot(len(channels), 2, i * 2 + 1)



            plt.title('Classification With Averaging.\nChannel {}'.format(channel + 1))

            cm = sklearn.metrics.confusion_matrix(test_y, y_preds[channel])

            plot_confusion_matrix(cm, ax=plt.gca(), cbar=False)

            

            plt.subplot(len(channels), 2, i * 2 + 2)

            

            preds = classifier.predict(test_X[:, :, channel])

            cm_without = sklearn.metrics.confusion_matrix(test_y, preds)

            

            plt.title('Classification Without Averaging.\nChannel {}'.format(channel + 1))

            plot_confusion_matrix(cm_without, ax=plt.gca(), cbar=False)



            plt.tight_layout()

        

        plt.show()
labels = (etiquetas[:, 0] + 1) // 2



svm_params = {

    'kernel': 'rbf',

    'gamma': 'scale',

    'class_weight': 'balanced',

    'random_state': 0,

}



clf_confusion_matrix(signal, labels, sklearn.svm.SVC, [1,], svm_params)
clf_confusion_matrix(signal, labels, sklearn.svm.SVC, np.arange(signal.shape[2]), svm_params)
rnd_parameters = {

    'class_weight': 'balanced',

    'random_state': 0,

    'max_features': 'log2',

    'min_samples_split': 10,

    'max_depth': 50,

}



clf_confusion_matrix(signal, labels, sklearn.ensemble.RandomForestClassifier, [1,], rnd_parameters)
channel_selection = [1, 2, 3, 5, 7]

classifiers = []



for channel in channel_selection:

    print('[+] Training classifier for channel {}... '.format(channel + 1), end='')

    classifier = sklearn.svm.SVC(kernel='rbf', gamma='scale', class_weight='balanced')

    classifier.fit(signal[:, :, channel], labels)

    print('Done')

    

    classifiers.append(('Classifier channel {}'.format(channel + 1), channel, classifier))
test_path = '../input/akimpech/P300db/ACS/ACS002/'

signal_test, etiquetas_test, t = load_train(test_path)
from scipy.special import expit





def make_labels_sweep(input_labels):

    # Impute missing values with zeros

    reps = 12 - (input_labels.shape[0] % 12)

    missing = np.zeros((reps, 2))

    

    input_labels = np.vstack([input_labels, missing])

    

    splits = np.split(input_labels, int(input_labels.shape[0] / 12))

    labels = []

    

    for split in splits:

        labels.append(1 * (split[split[:, 1].argsort()][:, 0] == 1))

        

    return labels





def get_predictions(signal, clfs):

    predictions = []

    names = []

    

    for name, channel, clf in clfs:

        prediction = clf.decision_function(signal[:, :, channel])

        

        names.append(name)

        predictions.append(prediction)

        

    arr = np.array(predictions)

    scores = np.sum(expit(arr), axis=0)

    

    return {'scores': scores, 'names': names, 'predictions': predictions}





matrix = [

    ['a', 'b', 'c', 'd', 'e', 'f'],

    ['g', 'h', 'i', 'j', 'k', 'l'],

    ['m', 'n', 'o', 'p', 'q', 'r'],

    ['s', 't', 'u', 'v', 'w', 'x'],

    ['y', 'z', '_', '1', '2', '3'],

    ['4', '5', '6', '7', '8', '9'],

]
def predict(signal_test, etiquetas_test, clfs):

    # Impute missing values with zeros

    reps = 12 - (etiquetas_test.shape[0] % 12)

    missing = np.zeros((reps, 2))

    

    etiquetas_test = np.vstack([etiquetas_test, missing])

    

    sweeps = np.array_split(etiquetas_test[:, 1], int(etiquetas_test.shape[0] / 12))

    

    r_sum = np.zeros((12, signal_test.shape[1], signal_test.shape[2]))

    letters = []

    

    bn = 0

    i = 0

    for sweep in sweeps:

        sweep_data = np.zeros((12, signal_test.shape[1], signal_test.shape[2]))

        

        for row in sweep:

            if row > 0:

                row = int(row - 1)

                sweep_data[row, :, :] = signal_test[i, :, :]

                

                i += 1

        

        r_sum = r_sum + sweep_data

        

        if bn % 14 == 0 and bn > 0:

            channel_predictions = get_predictions(r_sum / 15, clfs)

            

            row_score, col_score = channel_predictions['scores'][:6], channel_predictions['scores'][6:]

            m1, m2 = max(row_score), max(col_score)

            p1, p2 = np.where(row_score == m1)[0][0], np.where(col_score == m2)[0][0]

            

            letters.append(matrix[p1][p2])

            

            bn = 0

            r_sum = np.zeros((12, signal_test.shape[1], signal_test.shape[2]))



        else:

            bn += 1

            

    return ''.join(letters)
predict(signal_test, etiquetas_test, classifiers)
import torch

import torch.nn as nn

import torch.nn.functional as F





class OCLNN(torch.nn.Module):

    def __init__(self, n_channels, n_samples, channel_order=None):

        super(OCLNN, self).__init__()

        

        if n_channels == 0:

            if channel_order is None:

                raise ValueError('if n_channels equals 0, channel_order must be specified.')

                

            n_channels = len(channel_order)

        

        div = n_samples // 15

        

        self.conv1 = nn.Conv1d(n_channels, 16, div, stride=div)

        self.dropout  = nn.Dropout(p=0.25)

        self.linear = nn.Linear(self.out_shape(self.conv1, n_samples), 2)

        

        if torch.cuda.is_available():

            device = torch.device('cuda')

        else:

            device = torch.device('cpu')

            

        print('[+] Using device', device)

        

        self.to(device)

        self.device = device

        self.channel_order = channel_order

        

    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = self.linear(x)

        

        return x

    

    def fit(self, signal, labels, epochs=1000):

        # Assuming the current signal shape is (batch, data, channels)

        # Assuming labels come directly from obten_p300...

        

        signal = np.swapaxes(signal, 1, 2)

        labels = (labels[:, 0] + 1) // 2

        

        if self.channel_order is not None:

            signal = signal[:, self.channel_order, :]

        

        criteron = torch.nn.CrossEntropyLoss().to(self.device)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)



        torch_signal = torch.from_numpy(signal).float().to(self.device)

        torch_labels = torch.from_numpy(labels).long().to(self.device)



        for t in range(epochs):

            y_pred = model(torch_signal).float()

            loss = criteron(y_pred.float(), torch_labels)



            if t % 100 == 99:

                print('[+] Loss at epoch {}: {:.6f}'.format(t, loss.item()))



            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            

    def predict(self, signal):

        # Assuming signal is in the form: (batch, samples, channels)

        

        signal = np.swapaxes(signal, 1, 2)

        

        if self.channel_order is not None:

            signal = signal[:, self.channel_order, :]

        

        signal = torch.from_numpy(signal).float().to(self.device)

        predictions = F.softmax(self(signal), dim=1)

        

        return predictions.detach().cpu().numpy()

    

    def speller_predictions(self, data):

        preds = self.predict(data)

        

        row_idx = np.argmax(preds[:6, 1])

        col_idx = np.argmax(preds[6:, 1])



        vector = np.zeros(preds.shape[0], dtype='uint8')



        vector[row_idx] = 1

        vector[col_idx + 6] = 1



        return row_idx, col_idx, vector

        

    def out_shape(self, conv, len_in):        

        num = len_in + 2 * conv.padding[0] - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1

        den = conv.stride[0]

        

        return ((np.floor(num / den) + 1) * conv.out_channels).astype('uint64')
signal, etiquetas, t = load_train(base_path)
n_channels = signal.shape[2]

n_samples = signal.shape[1]



model = OCLNN(n_channels, n_samples)
print(model)
model.fit(signal, etiquetas)
bincount = np.bincount(labels.astype('uint8'))



w0 = signal.shape[0] / (2 * bincount[0])

w1 = signal.shape[0] / (2 * bincount[1])
signal_test, etiquetas_test, t = load_train('../input/akimpech/P300db/ACS/ACS002/')
test_labels = (etiquetas_test[:, 0] + 1) // 2



if etiquetas_test.shape[0] % 12 > 0:

    reps = 12 - (etiquetas_test.shape[0] % 12)

    missing = np.zeros((reps, 2))

    

    etiquetas_test = np.vstack([etiquetas_test, missing])
preds = model.predict(signal_test)

preds = 1 * (preds[:, 0] < preds[:, 1])
test_nerp = None

test_erp = None



meaned_pred = []



for etiqueta, test_sig in zip(test_labels, signal_test):

    test_sig = test_sig[np.newaxis, :, :]

    

    if etiqueta == 0:

        if test_nerp is None:

            test_nerp = test_sig

        

        else:

            test_nerp = np.vstack([test_nerp, test_sig])

            

        last = test_nerp

            

    elif etiqueta == 1:

        if test_erp is None:

            test_erp = test_sig

        

        else:

            test_erp = np.vstack([test_erp, test_sig])

            

        last = test_erp



    pred = model.predict(last.mean(axis=0)[np.newaxis, :, :])[0]

    

    pred = 1 * (pred[0] < pred[1])

    meaned_pred.append(pred)
deep_cm_mean = sklearn.metrics.confusion_matrix(test_labels, meaned_pred)

deep_cm = sklearn.metrics.confusion_matrix(test_labels, preds)



plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)

plt.title('Confusion Matrix for OCLNN\nWithout Averaging')

plot_confusion_matrix(deep_cm, ax=plt.gca())



plt.subplot(1, 2, 2)

plt.title('Confusion Matrix for OCLNN\nWith Averaging')

plot_confusion_matrix(deep_cm_mean, ax=plt.gca())



plt.tight_layout()

plt.show()
sweeps = np.split(etiquetas_test[:, 1], int(etiquetas_test.shape[0] / 12))



r_sum = np.zeros((12, signal_test.shape[1], signal_test.shape[2]))

letters = []



bn = 0

i = 0



for sweep in sweeps:

    sweep_data = np.zeros((12, signal_test.shape[1], signal_test.shape[2]))

        

    for row in sweep:

        if row > 0:

            row = int(row - 1)

            sweep_data[row, :, :] = signal_test[i, :, :]

            

            i += 1

            

    r_sum = r_sum + sweep_data

    

    if bn % 14 == 0 and bn > 0:

        # Prediction per row here

        row_idx, col_idx, vec = model.speller_predictions(r_sum / 15)

        

        letters.append(matrix[row_idx][col_idx])

        

        bn = 0

        r_sum = np.zeros((12, signal_test.shape[1], signal_test.shape[2]))

        

    else:

        bn += 1

        

print(''.join(letters))