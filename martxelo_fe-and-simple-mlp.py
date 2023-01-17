# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# imports

%matplotlib inline



import matplotlib.pyplot as plt

import seaborn as sns

from scipy import signal

from tqdm import tqdm

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, f1_score, plot_confusion_matrix

from keras.models import Model

import keras.layers as L

from keras.utils import to_categorical, plot_model
# read data

data = pd.read_csv('../input/liverpool-ion-switching/train.csv')

data.head()
def calc_gradients(s, n_grads=4):

    '''

    Calculate gradients for a pandas series. Returns the same number of samples

    '''

    grads = pd.DataFrame()

    

    g = s.values

    for i in range(n_grads):

        g = np.gradient(g)

        grads['grad_' + str(i+1)] = g

        

    return grads
def calc_low_pass(s, n_filts=10):

    '''

    Applies low pass filters to the signal. Left delayed and no delayed

    '''

    wns = np.logspace(-2, -0.3, n_filts)

    

    low_pass = pd.DataFrame()

    for wn in wns:

        b, a = signal.butter(1, Wn=wn, btype='low')

        low_pass['lowpass_lf_' + str('%.4f' %wn)] = signal.lfilter(b, a, s.values)

        low_pass['lowpass_ff_' + str('%.4f' %wn)] = signal.filtfilt(b, a, s.values)

        

    return low_pass
def calc_high_pass(s, n_filts=10):

    '''

    Applies high pass filters to the signal. Left delayed and no delayed

    '''

    wns = np.logspace(-2, -0.1, n_filts)

    

    high_pass = pd.DataFrame()

    for wn in wns:

        b, a = signal.butter(1, Wn=wn, btype='high')

        high_pass['hihgpass_lf_' + str('%.4f' %wn)] = signal.lfilter(b, a, s.values)

        high_pass['hihgpass_ff_' + str('%.4f' %wn)] = signal.filtfilt(b, a, s.values)

        

    return high_pass
def calc_roll_stats(s, windows=[10, 50, 100, 500, 1000]):

    '''

    Calculates rolling stats like mean, std, min, max...

    '''

    roll_stats = pd.DataFrame()

    for window in windows:

        roll_stats['roll_mean_' + str(window)] = s.rolling(window=window, min_periods=1).mean()

        roll_stats['roll_std_' + str(window)] = s.rolling(window=window, min_periods=1).std()

        roll_stats['roll_min_' + str(window)] = s.rolling(window=window, min_periods=1).min()

        roll_stats['roll_max_' + str(window)] = s.rolling(window=window, min_periods=1).max()

        roll_stats['roll_range_' + str(window)] = roll_stats['roll_max_' + str(window)] - roll_stats['roll_min_' + str(window)]

        roll_stats['roll_q10_' + str(window)] = s.rolling(window=window, min_periods=1).quantile(0.10)

        roll_stats['roll_q25_' + str(window)] = s.rolling(window=window, min_periods=1).quantile(0.25)

        roll_stats['roll_q50_' + str(window)] = s.rolling(window=window, min_periods=1).quantile(0.50)

        roll_stats['roll_q75_' + str(window)] = s.rolling(window=window, min_periods=1).quantile(0.75)

        roll_stats['roll_q90_' + str(window)] = s.rolling(window=window, min_periods=1).quantile(0.90)

    

    # add zeros when na values (std)

    roll_stats = roll_stats.fillna(value=0)

             

    return roll_stats
def calc_ewm(s, windows=[10, 50, 100, 500, 1000]):

    '''

    Calculates exponential weighted functions

    '''

    ewm = pd.DataFrame()

    for window in windows:

        ewm['ewm_mean_' + str(window)] = s.ewm(span=window, min_periods=1).mean()

        ewm['ewm_std_' + str(window)] = s.ewm(span=window, min_periods=1).std()

        

    # add zeros when na values (std)

    ewm = ewm.fillna(value=0)

        

    return ewm
def add_features(s):

    '''

    All calculations together

    '''

    

    gradients = calc_gradients(s)

    low_pass = calc_low_pass(s)

    high_pass = calc_high_pass(s)

    roll_stats = calc_roll_stats(s)

    ewm = calc_ewm(s)

    

    return pd.concat([s, gradients, low_pass, high_pass, roll_stats, ewm], axis=1)





def divide_and_add_features(s, signal_size=500000):

    '''

    Divide the signal in bags of "signal_size".

    Normalize the data dividing it by 15.0

    '''

    # normalize

    s = s/15.0

    

    ls = []

    for i in tqdm(range(int(s.shape[0]/signal_size))):

        sig = s[i*signal_size:(i+1)*signal_size].copy().reset_index(drop=True)

        sig_featured = add_features(sig)

        ls.append(sig_featured)

    

    return pd.concat(ls, axis=0)
# apply every feature to data

df = divide_and_add_features(data['signal'])

df.head()
print('df.shape=', df.shape)
# Get train and test data

x_train, x_test, y_train, y_test = train_test_split(df.values, data['open_channels'].values, test_size=0.2)



print('x_train.shape=', x_train.shape)

print('x_test.shape=', x_test.shape)

print('y_train.shape=', y_train.shape)

print('y_test.shape=', y_test.shape)
def create_mpl(shape):

    '''

    Returns a keras model

    '''

    

    X_input = L.Input(shape)

    

    X = L.Dense(150, activation='relu')(X_input)

    X = L.Dense(125, activation='relu')(X)

    X = L.Dense(75, activation='relu')(X)

    X = L.Dense(50, activation='relu')(X)

    X = L.Dense(25, activation='relu')(X)

    X = L.Dense(11, activation='softmax')(X)

    

    model = Model(inputs=X_input, outputs=X)

    

    return model





mlp = create_mpl(x_train[0].shape)

mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

print(mlp.summary())
def get_class_weight(classes):

    '''

    Weight of the class is inversely proportional to the population of the class

    '''

    hist, _ = np.histogram(classes, bins=np.arange(12)-0.5)

    class_weight = hist.sum()/hist

    

    return class_weight



class_weight = get_class_weight(y_train)
# fit the model

mlp.fit(x=x_train, y=y_train, epochs=30, batch_size=1024, class_weight=class_weight)
# plot history

plt.figure(1)

plt.plot(mlp.history.history['loss'], 'b', label='loss')

plt.xlabel('epochs')

plt.legend()

plt.figure(2)

plt.plot(mlp.history.history['sparse_categorical_accuracy'], 'g', label='sparse_categorical_accuracy')

plt.xlabel('epochs')

plt.legend()
# predict on test

y_pred = mlp.predict(x_test)

y_pred = np.argmax(y_pred, axis=-1)
# Thanks to https://www.kaggle.com/marcovasquez/basic-nlp-with-tensorflow-and-wordcloud

def plot_cm(y_true, y_pred, title):

    figsize=(16,16)

    y_pred = y_pred.astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))

    cm_sum = np.sum(cm, axis=1, keepdims=True)

    cm_perc = cm / cm_sum.astype(float) * 100

    annot = np.empty_like(cm).astype(str)

    nrows, ncols = cm.shape

    for i in range(nrows):

        for j in range(ncols):

            c = cm[i, j]

            p = cm_perc[i, j]

            if i == j:

                s = cm_sum[i]

                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

            elif c == 0:

                annot[i, j] = ''

            else:

                annot[i, j] = '%.1f%%\n%d' % (p, c)

    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))

    cm.index.name = 'Actual'

    cm.columns.name = 'Predicted'

    fig, ax = plt.subplots(figsize=figsize)

    plt.title(title)

    sns.heatmap(cm, cmap='viridis', annot=annot, fmt='', ax=ax)



# f1 score

f1 = f1_score(y_test, y_pred, average='macro')



# plot confusion matrix

plot_cm(y_test, y_pred, 'MLP f1_score=' + str('%.4f' %f1))
# read test

test = pd.read_csv('../input/liverpool-ion-switching/test.csv')

test.head()
# create df_submit

df_submit = divide_and_add_features(test['signal'])

print('df_submit.shape=', df_submit.shape)

df_submit.head()
# predict open channels

y_submit = mlp.predict(df_submit.values)

y_submit = np.argmax(y_submit, axis=-1)
# create submission

submission = pd.DataFrame()

submission['time'] = test['time']

submission['open_channels'] = y_submit



# write file

submission.to_csv('submission.csv', index=False, float_format='%.4f')