import numpy as np

def mixup(data, one_hot_labels, alpha=1, debug=False):
    np.random.seed(42)

    batch_size = len(data)
    weights = np.random.beta(alpha, alpha, batch_size)
    index = np.random.permutation(batch_size)
    x1, x2 = data, data[index]
    x = np.array([x1[i] * weights [i] + x2[i] * (1 - weights[i]) for i in range(len(weights))])
    y1 = np.array(one_hot_labels).astype(np.float)
    y2 = np.array(np.array(one_hot_labels)[index]).astype(np.float)
    y = np.array([y1[i] * weights[i] + y2[i] * (1 - weights[i]) for i in range(len(weights))])
    if debug:
        print('Mixup weights', weights)
    return x, y
%matplotlib inline
from matplotlib import pyplot as plt
import os
import numpy as np
import pandas as pd
import keras

datadir = "../input/"

# Load data as is
X_train_org = np.load(datadir+'X_train.npy')
X_test = np.load(datadir+'X_test.npy')
y_labels_train = pd.read_csv(datadir+'y_train.csv', sep=',')['scene_label'].tolist()

# Make label list and converters
labels = sorted(list(set(y_labels_train)))
label2int = {l:i for i, l in enumerate(labels)}
int2label = {i:l for i, l in enumerate(labels)}

# Map y_train to int labels
y_train_org = keras.utils.to_categorical([label2int[l] for l in y_labels_train])

# Train/Validation split --> X_train/y_train, X_valid/y_valid
splitlist = pd.read_csv(datadir+'crossvalidation_train.csv', sep=',')['set'].tolist()
X_train = np.array([x for i, x in enumerate(X_train_org) if splitlist[i] == 'train'])
X_valid = np.array([x for i, x in enumerate(X_train_org) if splitlist[i] == 'test'])
y_train = np.array([y for i, y in enumerate(y_train_org) if splitlist[i] == 'train'])
y_valid = np.array([y for i, y in enumerate(y_train_org) if splitlist[i] == 'test'])
# Now augment by mixup --> X_train + mixup'ed X_train, y_train + mixup'ed y_train
# ** This is slight deviation from original idea in paper,
#    using mixup as preprocessing. **
tmp_X, tmp_y = mixup(X_train, y_train, alpha=1)
X_train, y_train = np.r_[X_train, tmp_X], np.r_[y_train, tmp_y]
# Here we pick first five samples, and mix-up.
five_X, five_y = X_train[:5], y_train[:5]
mixup_X, mixup_y = mixup(five_X, five_y, alpha=3, debug=True)
# Visualize them.
def plot_dataset(XYs, titles):
    for i, (x, y) in enumerate(XYs):
        fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))
        axL.pcolormesh(x)
        axL.set_title('%s [%d]' % (titles[0], i))
        axL.grid(True)
        axR.pcolormesh(y)
        axR.set_title('%s [%d]' % (titles[1], i))
        axR.grid(True)
        plt.show()

plot_dataset(zip(five_X, mixup_X), ('Original data', 'Mixup result'))# Here we pick first two samples, and mix-up.
print(mixup_y)
