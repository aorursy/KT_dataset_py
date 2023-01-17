import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from keras import Sequential

from keras.layers import Dense, LeakyReLU, Dropout

from keras.optimizers import adam

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import MinMaxScaler
# Set Parameters

pd.options.display.max_rows = 30

pd.options.display.max_columns = 20

pd.options.display.width=500

np.set_printoptions(formatter={'float_kind': '{:.4f}'.format}, linewidth=200, edgeitems=5)  # Limit float display to 4 decimals

train_pct = 0.8

epochs = 10

batch_size = 128

verbose = 2

sns.set()

states = {'A': 'Baseline', 'B': 'SS', 'C': 'CA', 'D': 'DA'}

x_cont = ['eeg_fp1', 'eeg_f7', 'eeg_f8', 'eeg_t4', 'eeg_t6', 'eeg_t5', 'eeg_t3', 'eeg_fp2', 'eeg_o1', 'eeg_p3', 'eeg_pz', 'eeg_f3', 'eeg_fz', 'eeg_f4',

          'eeg_c4', 'eeg_p4', 'eeg_poz', 'eeg_c3', 'eeg_cz', 'eeg_o2', 'ecg', 'r', 'gsr']

# Random Forest Model

def random_forest(trainX, trainY, validX, validY):

    forest = ExtraTreesClassifier(n_estimators=10, n_jobs=-1, class_weight='balanced')



    forest.fit(trainX, trainY)

    forest.score(validX,validY)

    predY = forest.predict(validX)

    conf_mat = confusion_matrix(validY, predY)



    # Normalize values

    row_sums = conf_mat.sum(axis=1, keepdims=True)

    norm_conf_mat = conf_mat / row_sums



    conf_mat = norm_conf_mat

    return conf_mat

# Read Data

train = pd.read_csv('../input/train.csv')

train['event'] = train['event'].astype('category')

train[x_cont] = train[x_cont].astype('float64')



# Prepare Data

valid = train[train['crew'] == 13]

train = train[train['crew'] != 13]

trainX = train[x_cont]

trainY = train['event']

validX = valid[x_cont]

validY = valid['event']

train = pd.concat([trainX,trainY], axis='columns')
from matplotlib import rcParams

rcParams['figure.max_open_warning']=False

features = train.columns

fig, axes = plt.subplots(5,5, figsize=(20,20))

for r in range(0,5):

    for c in range(0,5):

        i= r * 5 + c

        if i >= len(features)-1: break

        feature = train.columns[i]

        sns.catplot(x='event', y=feature, data=train,ax=axes[r][c])
# Random Forest

conf_mat = random_forest(trainX, trainY, validX, validY)

print(conf_mat)
# Plot Confusion Matrix

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)

fig.colorbar(cax)

ax.set_xticklabels([''] + list(states.keys()))

ax.set_yticklabels([''] + list(states.keys()))

plt.title('Confusion Matrix')

plt.xlabel('Predicted')

plt.ylabel('Expected')

plt.show()