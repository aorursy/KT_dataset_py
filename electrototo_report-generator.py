from scipy.signal import butter, detrend, filtfilt

from lini_read_bcidat import lini_read_bcidat

import numpy as np



# default params

dp = {'car': 1, 'ica': 0, 'local_tend': 1, 'norm': 1, 'filtro': []}

dw = [-100., 800.]

dch = np.arange(10)





def obten_p300_bci2000(f, channels=dch, window=dw, paco=dp, decimation=False, decimation_samples=4):



    data = lini_read_bcidat(f)

    raw_signal, states_matrix, labels, paco['fs'] = data



    states = {}



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



    letras = (np.where(np.diff(states['PhaseInSequence']) > 0))[0]

    letras = np.concatenate([letras[0::2], [letras[-1]]])

    

    data = np.zeros([ind.shape[0], int(vf + vi), n_canales])

    etiqueta = -np.ones((ind.shape[0], 2))

    

    for i, index in enumerate(ind):

        rowcol = states['StimulusCode'][index]



        if index - vi < 0:

            raise ValueError('tiempo inicial muy largo')

            

        data[i, :, :] = signal[(index - vi):(index + vf), :]



        if states['StimulusType'][index] == 1:

            etiqueta[i, :] = [1, rowcol]

        else:

            etiqueta[i, :] = [-1, rowcol]

            

    if decimation:

        idx = np.arange(0, data.shape[1], decimation_samples)

        data = data[:, idx, :]

        t = t[idx]



    return (raw_signal, signal, data, etiqueta, ind, states, t, letras)





channel_names = [

    'Fz', 'C3', 'Cz', 'C4',

    'P3', 'Pz', 'P4', 'PO7',

    'PO8', 'Oz'

]
import os



def load_train(train_path):

    data = None

    

    for filename in os.listdir(train_path):

        full = os.path.join(train_path, filename)

        

        if '.dat' in full:            

            rt, st, da, et, it, sta, tt, lt = obten_p300_bci2000(full, decimation=True)

            

            if data is None:

                data = da

                t = tt

                etiquetas = et

            else:

                data = np.vstack([data, da])

                etiquetas = np.vstack([etiquetas, et])

                

    return data, etiquetas, t





def load_test(test_path):

    

    files = []

    for filename in os.listdir(test_path):

        full = os.path.join(test_path, filename)

        

        if '.dat' in filename:

            files.append(full)

    

    return files
def avg_train_idx(signal, split_samples=5, shuffle=True):

    # Returns the indices of the signal split into subarrays

    

    samples = signal.shape[0]

    

    if samples % split_samples > 0:

        samples = samples - (samples % split_samples)

    

    idx = np.arange(samples)

    

    if shuffle:

        np.random.shuffle(idx)



    idx = np.split(idx, samples / split_samples)

    

    return idx





def avg_split_signal(signal, labels, split_samples=5, shuffle=True):

    erp_idx = np.where(labels == 1)[0]

    nerp_idx = np.where(labels == 0)[0]



    erp = signal[erp_idx, :, :]

    nerp = signal[nerp_idx, :, :]



    erp_shuffle_idx = avg_train_idx(erp, split_samples=split_samples, shuffle=shuffle)

    nerp_shuffle_idx = avg_train_idx(nerp, split_samples=split_samples, shuffle=shuffle)



    erp = erp[erp_shuffle_idx].mean(axis=1)

    nerp = nerp[nerp_shuffle_idx].mean(axis=1)



    new_labels = np.hstack([np.ones(erp.shape[0]), np.zeros(nerp.shape[0])])

    new_signal = np.vstack([erp, nerp])



    idx = np.arange(new_signal.shape[0])

    

    if shuffle:

        np.random.shuffle(idx)



    new_signal = new_signal[idx]

    new_labels = new_labels[idx]

    

    return new_signal, new_labels





def make_labels_sweep(input_labels):

    splits = np.split(input_labels, int(input_labels.shape[0] / 12))

    labels = []

    

    for split in splits:

        labels.append(1 * (split[split[:, 1].argsort()][:, 0] == 1))

        

    return labels
class BaseClassifier:

    def speller_predictions(self, data):

        preds = self.predict(data)

        

        row_idx = np.argmax(preds[:6, 1])

        col_idx = np.argmax(preds[6:, 1])



        vector = np.zeros(preds.shape[0], dtype='uint8')



        vector[row_idx] = 1

        vector[col_idx + 6] = 1



        return row_idx, col_idx, vector
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
import sklearn.svm

import scipy





class SVMBaseline(BaseClassifier):

    def __init__(self, channels):

        # channels contains the channels that will be used to train the classifier

        # note: this channels are zero-indexed.

        

        clfs = []

        

        for channel in channels:

            clf = sklearn.svm.SVC(kernel='rbf', gamma='scale', class_weight='balanced')

            clfs.append(('Classifier for channel {}'.format(channel + 1), channel, clf))

            

        self.clfs = clfs



    def fit(self, signal, labels, avg=True, verbose=False):

        # avg is for averaging the signal before training

        

        labels = (labels[:, 0] + 1) // 2

        

        if avg:

            signal, labels = avg_split_signal(signal, labels)

        

        for label, channel, clf in self.clfs:   

            if verbose:

                print('[+] Training classifier for channel {}'.format(channel + 1))



            clf.fit(signal[:, :, channel], labels)

            

    def predict(self, signal, hard=False):

        predictions = np.zeros((signal.shape[0], 2))

        

        for label, channel, clf in self.clfs:

            prediction = clf.decision_function(signal[:, :, channel])

            

            idx = np.where(prediction < clf.intercept_)

            predictions[idx, 0] = predictions[idx, 0] + np.abs(prediction[idx])

            

            idx = np.where(prediction >= clf.intercept_)

            predictions[idx, 1] = predictions[idx, 1] + np.abs(prediction[idx])

        

        return scipy.special.softmax(predictions, axis=1)
import torch

import torch.nn as nn

import torch.nn.functional as F





class OCLNN(BaseClassifier, torch.nn.Module):

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

        

        self.to(device)

        self.device = device

        self.channel_order = channel_order

        

    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = self.linear(x)

        

        return x

    

    def fit(self, signal, labels, epochs=1000, avg=True, verbose=True):

        # Assuming the current signal shape is (batch, data, channels)

        # Assuming labels come directly from obten_p300...

        self.train()

        

        signal = np.swapaxes(signal, 1, 2)

        labels = (labels[:, 0] + 1) // 2

        

        if self.channel_order is not None:

            signal = signal[:, self.channel_order, :]

        

        criteron = torch.nn.CrossEntropyLoss().to(self.device)

        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

        

        if avg:

            signal, labels = avg_split_signal(signal, labels, split_samples=5, shuffle=True)



        torch_signal = torch.from_numpy(signal).float().to(self.device)

        torch_labels = torch.from_numpy(labels).long().to(self.device)



        for t in range(epochs):

            y_pred = self(torch_signal).float()

            loss = criteron(y_pred.float(), torch_labels)



            if t % 100 == 99 and verbose:

                print('[+] Loss at epoch {}: {:.6f}'.format(t, loss.item()))



            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            

    def predict(self, signal):

        # Assuming signal is in the form: (batch, samples, channels)

        self.eval()

        

        signal = np.swapaxes(signal, 1, 2)

        

        if self.channel_order is not None:

            signal = signal[:, self.channel_order, :]

        

        signal = torch.from_numpy(signal).float().to(self.device)

        predictions = F.softmax(self(signal), dim=1)

        

        return predictions.detach().cpu().numpy()

        

    def out_shape(self, conv, len_in):        

        num = len_in + 2 * conv.padding[0] - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1

        den = conv.stride[0]

        

        return ((np.floor(num / den) + 1) * conv.out_channels).astype('uint64')
class Benchmark:

    def __init__(self, test_path):

        _, _, signal, labels, _, _, _, _ = obten_p300_bci2000(test_path, decimation=True)

            

        type_labels = (labels[:, 0] + 1) // 2

        

        self.type_labels = type_labels

        self.test_path = test_path

        self.labels = labels

        self.signal = signal

        

        matrix = [

            ['a', 'b', 'c', 'd', 'e', 'f'],

            ['g', 'h', 'i', 'j', 'k', 'l'],

            ['m', 'n', 'o', 'p', 'q', 'r'],

            ['s', 't', 'u', 'v', 'w', 'x'],

            ['y', 'z', '_', '1', '2', '3'],

            ['4', '5', '6', '7', '8', '9'],

        ]

        

        self.matrix = matrix

        

    def confusion_matrix(self, classifier):

        # Returns the confusion matrices:

        #     1. Individual prediction

        #     2. Averaged prediction

        

        test_signal = self.signal

        test_labels = self.type_labels

        

        preds = []

        

        # Signal averaging

        test_erp = None

        test_nerp = None

        

        for feature, label in zip(test_signal, test_labels):

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

                

            pred = classifier.predict(latest.mean(axis=0)[np.newaxis])[0]

            pred = 1 * (pred[0] < pred[1])

            

            preds.append(pred)

            

        ind_pred = classifier.predict(test_signal)

        ind_pred = 1 * (ind_pred[:, 0] < ind_pred[:, 1])

        

        cm1 = sklearn.metrics.confusion_matrix(test_labels, ind_pred)

        cm2 = sklearn.metrics.confusion_matrix(test_labels, preds)

        

        return cm1, cm2

    

    def speller_accuracy(self, classifier):

        signal_test = self.signal

        etiquetas_test = self.labels

        matrix = self.matrix

        

        sweeps = np.array_split(etiquetas_test[:, 1], int(etiquetas_test.shape[0] / 12))



        r_sum = np.zeros((12, signal_test.shape[1], signal_test.shape[2]))

        letters = []



        targets = make_labels_sweep(etiquetas_test)

        epoch_preds = np.zeros(15)

        epoch_matrix = []

        row_matrix = []



        bn = 0

        i = 0

        for target, sweep in zip(targets, sweeps):

            sweep_data = np.zeros((12, signal_test.shape[1], signal_test.shape[2]))



            for row in sweep:

                if row > 0:

                    row = int(row - 1)

                    sweep_data[row, :, :] = signal_test[i, :, :]



                    i += 1



            r_sum = r_sum + sweep_data



            if bn % 14 == 0 and bn > 0:

                row_idx, col_idx, vec = classifier.speller_predictions(r_sum / 15)



                if np.array_equal(vec, target):

                    epoch_preds[bn] += 1



                l = matrix[row_idx][col_idx]



                row_matrix.append(l)

                epoch_matrix.append(row_matrix)

                letters.append(l)



                bn = 0

                row_matrix = []

                r_sum = np.zeros((12, signal_test.shape[1], signal_test.shape[2]))



            else:

                row_idx, col_idx, vec = classifier.speller_predictions(r_sum / (bn + 1))



                if np.array_equal(vec, target):

                    epoch_preds[bn] += 1



                l = matrix[row_idx][col_idx]

                row_matrix.append(l)



                bn += 1



        return epoch_preds, epoch_matrix, ''.join(letters)
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from matplotlib.backends.backend_pdf import PdfPages
subject = 'LAG'



base_path = '../input/akimpech/P300db/'

kaggle_output = f'/kaggle/working/{subject}'



train_path = f'{base_path}{subject}/{subject}001/'



signal, etiquetas, t = load_train(train_path)

        

if not os.path.exists(kaggle_output):

    os.mkdir(kaggle_output)
fig = plt.figure(figsize=(15, 25))

outer_grid = fig.add_gridspec(5, 1, hspace=0.5)



plot_offset = 0.032



erp = signal[np.where(etiquetas[:, 0] == 1)[0], :, :]

nerp = signal[np.where(etiquetas[:, 0] == -1)[0], :, :]



inner_grid = outer_grid[0:2].subgridspec(2, 5, wspace=0.5, hspace=0.5)



plt.figtext(0.5, 0.92, 'Report for Subject {}'.format(subject), ha='center', fontsize='xx-large', weight='bold')

plt.figtext(0.5, 0.9, 'Channel Averages', ha='center', fontsize='large', weight='bold')



for channel in range(10):

    ax = fig.add_subplot(inner_grid[channel])



    ax.plot(t, erp[:, :, channel].mean(axis=0), label='ERP')

    ax.plot(t, nerp[:, :, channel].mean(axis=0), label='Background')

    

    ax.set(

        title=channel_names[channel],

        ylabel='A.U.',

        xlabel='Time [ms]',

    )

    

    ax.legend(loc='lower left')

    ax.grid()

    fig.add_subplot(ax)



    

ocl_avg = OCLNN(signal.shape[2], signal.shape[1])

ocl_nrm = OCLNN(signal.shape[2], signal.shape[1])



ocl_avg.fit(signal, etiquetas, verbose=False)

ocl_nrm.fit(signal, etiquetas, avg=False, verbose=False)



svm_avg = SVMBaseline([1, 2, 3, 5, 7])

svm_nrm = SVMBaseline([1, 2, 3, 5, 7])



svm_avg.fit(signal, etiquetas)

svm_nrm.fit(signal, etiquetas, avg=False)



bench = Benchmark(f'{base_path}{subject}/{subject}002/{subject}S002R01.dat')



ocl_cm = bench.confusion_matrix(ocl_nrm)

svm_cm = bench.confusion_matrix(svm_nrm)



inner_grid = outer_grid[2].subgridspec(1, 4, wspace=0.7)



ax1 = fig.add_subplot(inner_grid[0])

ax1.set_title('CNN\nIndividual')

plot_confusion_matrix(ocl_cm[0], ax=ax1, cbar=False)



ax2 = fig.add_subplot(inner_grid[1])

ax2.set_title('CNN\nAveraging')

plot_confusion_matrix(ocl_cm[1], ax=ax2, cbar=False)



ax3 = fig.add_subplot(inner_grid[2])

ax3.set_title('SVM\nIndividual')

plot_confusion_matrix(svm_cm[0], ax=ax3, cbar=False)



ax4 = fig.add_subplot(inner_grid[3])

ax4.set_title('SVM\nAveraging')

plot_confusion_matrix(svm_cm[1], ax=ax4, cbar=False)



bbox = ax4.get_position().bounds

plt.figtext(0.5, bbox[1] + bbox[3] + plot_offset, 'Classifiers Trained Without Averaging Aggregated Data', ha='center', fontsize='large', weight='bold')



fig.add_subplot(ax1)

fig.add_subplot(ax2)

fig.add_subplot(ax3)

fig.add_subplot(ax4)



ocl_cm = bench.confusion_matrix(ocl_avg)

svm_cm = bench.confusion_matrix(svm_avg)



inner_grid = outer_grid[3].subgridspec(1, 4, wspace=0.7)



ax1 = fig.add_subplot(inner_grid[0])

ax1.set_title('CNN\nIndividual')

plot_confusion_matrix(ocl_cm[0], ax=ax1, cbar=False)



ax2 = fig.add_subplot(inner_grid[1])

ax2.set_title('CNN\nAveraging')

plot_confusion_matrix(ocl_cm[1], ax=ax2, cbar=False)



ax3 = fig.add_subplot(inner_grid[2])

ax3.set_title('SVM\nIndividual')

plot_confusion_matrix(svm_cm[0], ax=ax3, cbar=False)



ax4 = fig.add_subplot(inner_grid[3])

ax4.set_title('SVM\nAveraging')

plot_confusion_matrix(svm_cm[1], ax=ax4, cbar=False)



bbox = ax4.get_position().bounds

plt.figtext(0.5, bbox[1] + bbox[3] + plot_offset, 'Classifiers Trained Averaging Aggregated Data', ha='center', fontsize='large', weight='bold')



fig.add_subplot(ax1)

fig.add_subplot(ax2)

fig.add_subplot(ax3)

fig.add_subplot(ax4)



# Speller



ocl_nrm_speller = bench.speller_accuracy(ocl_nrm)

svm_nrm_speller = bench.speller_accuracy(svm_nrm)



ocl_avg_speller = bench.speller_accuracy(ocl_avg)

svm_avg_speller = bench.speller_accuracy(svm_avg)



epochs = np.arange(ocl_nrm_speller[0].shape[0]) + 1

word_size = len(ocl_nrm_speller[2])



inner_grid = outer_grid[4].subgridspec(1, 2)



ax = fig.add_subplot(inner_grid[0])



ax.plot(epochs, ocl_nrm_speller[0] / word_size, label='OCLNN')

ax.plot(epochs, svm_nrm_speller[0] / word_size, label='SVM')



ax.set(

    title='Trained Without Averaging Data',

    xlabel='# Epoch',

    ylabel='Accuracy',

    ylim=[0, 1.1],

    xlim=[1, 15],

)



ax.grid()

ax.legend(loc='lower right')



fig.add_subplot(ax)



ax = fig.add_subplot(inner_grid[1])



ax.plot(epochs, ocl_avg_speller[0] / word_size, label='OCLNN')

ax.plot(epochs, svm_avg_speller[0] / word_size, label='SVM')



ax.set(

    title='Trained Averaging Data',

    xlabel='# Epoch',

    ylabel='Accuracy',

    ylim=[0, 1.1],

    xlim=[1, 15],

)



ax.grid()

ax.legend(loc='lower right')



bbox = ax.get_position().bounds

plt.figtext(0.5, bbox[1] + bbox[3] + plot_offset, 'Speller Accuracy for Second Session', ha='center', fontsize='large', weight='bold')



fig.add_subplot(ax)



plt.show()
for subject in os.listdir(base_path):

    if os.path.isdir(os.path.join(base_path, subject)):

        print('[+] Creating results for subject {}'.format(subject))

        

        kaggle_output = f'/kaggle/working/{subject}'

        pdf_output = f'/kaggle/working/{subject}/report.pdf'

        

        train_path = f'{base_path}{subject}/{subject}001/'



        signal, etiquetas, t = load_train(train_path)

        

        if not os.path.exists(kaggle_output):

            os.mkdir(kaggle_output)

        

        fig = plt.figure(figsize=(15, 25))

        outer_grid = fig.add_gridspec(5, 1, hspace=0.5)



        plot_offset = 0.032



        erp = signal[np.where(etiquetas[:, 0] == 1)[0], :, :]

        nerp = signal[np.where(etiquetas[:, 0] == -1)[0], :, :]



        inner_grid = outer_grid[0:2].subgridspec(2, 5, wspace=0.5, hspace=0.5)



        plt.figtext(0.5, 0.92, 'Report for subject {}'.format(subject), ha='center', fontsize='xx-large', weight='bold')

        plt.figtext(0.5, 0.9, 'Channel Averages', ha='center', fontsize='large', weight='bold')



        for channel in range(10):

            ax = fig.add_subplot(inner_grid[channel])



            ax.plot(t, erp[:, :, channel].mean(axis=0), label='ERP')

            ax.plot(t, nerp[:, :, channel].mean(axis=0), label='Background')



            ax.set(

                title=channel_names[channel],

                ylabel='A.U.',

                xlabel='Time [ms]',

            )



            ax.legend(loc='lower left')

            ax.grid()

            fig.add_subplot(ax)





        ocl_avg = OCLNN(signal.shape[2], signal.shape[1])

        ocl_nrm = OCLNN(signal.shape[2], signal.shape[1])



        ocl_avg.fit(signal, etiquetas, verbose=False)

        ocl_nrm.fit(signal, etiquetas, avg=False, verbose=False)



        svm_avg = SVMBaseline([1, 2, 3, 5, 7])

        svm_nrm = SVMBaseline([1, 2, 3, 5, 7])



        svm_avg.fit(signal, etiquetas)

        svm_nrm.fit(signal, etiquetas, avg=False)

        

        # Try to find the file associated with the second session.

        base_subject = f'{base_path}{subject}/{subject}002/'

        

        fname = list(filter(lambda x: '.dat' in x, os.listdir(base_subject)))[0]

        fname = os.path.join(base_subject, fname)

        

        print(fname)

        bench = Benchmark(fname)



        ocl_cm = bench.confusion_matrix(ocl_nrm)

        svm_cm = bench.confusion_matrix(svm_nrm)



        inner_grid = outer_grid[2].subgridspec(1, 4, wspace=0.7)



        ax1 = fig.add_subplot(inner_grid[0])

        ax1.set_title('CNN\nIndividual')

        plot_confusion_matrix(ocl_cm[0], ax=ax1, cbar=False)



        ax2 = fig.add_subplot(inner_grid[1])

        ax2.set_title('CNN\nAveraging')

        plot_confusion_matrix(ocl_cm[1], ax=ax2, cbar=False)



        ax3 = fig.add_subplot(inner_grid[2])

        ax3.set_title('SVM\nIndividual')

        plot_confusion_matrix(svm_cm[0], ax=ax3, cbar=False)



        ax4 = fig.add_subplot(inner_grid[3])

        ax4.set_title('SVM\nAveraging')

        plot_confusion_matrix(svm_cm[1], ax=ax4, cbar=False)



        bbox = ax4.get_position().bounds

        plt.figtext(0.5, bbox[1] + bbox[3] + plot_offset, 'Classifiers Trained Without Averaging Aggregated Data', ha='center', fontsize='large', weight='bold')



        fig.add_subplot(ax1)

        fig.add_subplot(ax2)

        fig.add_subplot(ax3)

        fig.add_subplot(ax4)



        ocl_cm = bench.confusion_matrix(ocl_avg)

        svm_cm = bench.confusion_matrix(svm_avg)



        inner_grid = outer_grid[3].subgridspec(1, 4, wspace=0.7)



        ax1 = fig.add_subplot(inner_grid[0])

        ax1.set_title('CNN\nIndividual')

        plot_confusion_matrix(ocl_cm[0], ax=ax1, cbar=False)



        ax2 = fig.add_subplot(inner_grid[1])

        ax2.set_title('CNN\nAveraging')

        plot_confusion_matrix(ocl_cm[1], ax=ax2, cbar=False)



        ax3 = fig.add_subplot(inner_grid[2])

        ax3.set_title('SVM\nIndividual')

        plot_confusion_matrix(svm_cm[0], ax=ax3, cbar=False)



        ax4 = fig.add_subplot(inner_grid[3])

        ax4.set_title('SVM\nAveraging')

        plot_confusion_matrix(svm_cm[1], ax=ax4, cbar=False)



        bbox = ax4.get_position().bounds

        plt.figtext(0.5, bbox[1] + bbox[3] + plot_offset, 'Classifiers Trained Averaging Aggregated Data', ha='center', fontsize='large', weight='bold')



        fig.add_subplot(ax1)

        fig.add_subplot(ax2)

        fig.add_subplot(ax3)

        fig.add_subplot(ax4)



        # Speller



        ocl_nrm_speller = bench.speller_accuracy(ocl_nrm)

        svm_nrm_speller = bench.speller_accuracy(svm_nrm)



        ocl_avg_speller = bench.speller_accuracy(ocl_avg)

        svm_avg_speller = bench.speller_accuracy(svm_avg)



        epochs = np.arange(ocl_nrm_speller[0].shape[0]) + 1

        word_size = len(ocl_nrm_speller[2])



        inner_grid = outer_grid[4].subgridspec(1, 2)



        ax = fig.add_subplot(inner_grid[0])



        ax.plot(epochs, ocl_nrm_speller[0] / word_size, label='OCLNN')

        ax.plot(epochs, svm_nrm_speller[0] / word_size, label='SVM')



        ax.set(

            title='Trained Without Averaging Data',

            xlabel='# Epoch',

            ylabel='Accuracy',

            ylim=[0, 1.1],

            xlim=[1, 15],

        )



        ax.grid()

        ax.legend(loc='lower right')



        fig.add_subplot(ax)



        ax = fig.add_subplot(inner_grid[1])



        ax.plot(epochs, ocl_avg_speller[0] / word_size, label='OCLNN')

        ax.plot(epochs, svm_avg_speller[0] / word_size, label='SVM')



        ax.set(

            title='Trained Averaging Data',

            xlabel='# Epoch',

            ylabel='Accuracy',

            ylim=[0, 1.1],

            xlim=[1, 15],

        )



        ax.grid()

        ax.legend(loc='lower right')



        bbox = ax.get_position().bounds

        plt.figtext(0.5, bbox[1] + bbox[3] + plot_offset, 'Speller Accuracy for Second Session', ha='center', fontsize='large', weight='bold')



        fig.add_subplot(ax)

        

        with PdfPages(pdf_output) as pdf:

            pdf.savefig(fig)

            

        plt.close()
!zip -r output.zip /kaggle/working/