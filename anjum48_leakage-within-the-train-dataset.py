# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from tqdm import tqdm_notebook

import itertools

from sklearn.metrics import accuracy_score, confusion_matrix, pairwise_distances, pairwise_distances_argmin

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, train_test_split, GroupShuffleSplit, StratifiedShuffleSplit, GroupKFold

from sklearn.neighbors import NearestNeighbors



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/X_train.csv')

y = pd.read_csv('../input/y_train.csv')

test = pd.read_csv('../input/X_test.csv')

sub = pd.read_csv('../input/sample_submission.csv')
def get_start_end_points(data):

    start = data.query("measurement_number==0").reset_index()

    end = data.query("measurement_number==127").reset_index()



    columns = ["orientation_X", "orientation_Y", "orientation_Z", "orientation_W"]

    

    start, end = start[columns], end[columns]



    points = start.join(end, lsuffix="_start", rsuffix="_end").join(y)

    return points
train_points = get_start_end_points(train)

test_points = get_start_end_points(test)



train_points.head()
# https://www.kaggle.com/artgor/where-do-the-robots-drive



def plot_confusion_matrix(truth, pred, classes, normalize=False, title=''):

    cm = confusion_matrix(truth, pred)

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    

    plt.figure(figsize=(10, 10))

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    plt.title('Confusion matrix', size=15)

    plt.colorbar(fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.grid(False)

    plt.tight_layout()
le = LabelEncoder()

y['surface'] = le.fit_transform(y['surface'])

scores = []



folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)



for fold, (trn_idx, val_idx) in enumerate(folds.split(train_points.values, y['surface'].values)):

    # Compare start to end, and end to start

    x_train_se, x_valid_se = train_points.iloc[trn_idx].filter(regex='start'), train_points.iloc[val_idx].filter(regex='end')

    x_train_es, x_valid_es = train_points.iloc[trn_idx].filter(regex='end'), train_points.iloc[val_idx].filter(regex='start')



    y_train, y_valid = y["surface"][trn_idx], y["surface"][val_idx],

        

    neigh = NearestNeighbors(1)

    

    neigh.fit(x_train_se)

    distances_se, indices_se = neigh.kneighbors(x_valid_se)

    

    neigh.fit(x_train_es)

    distances_es, indices_es = neigh.kneighbors(x_valid_es)

    

    # Find the minimum distance to select the nearest match

    distances = np.concatenate([distances_se, distances_es], -1)

    indices = np.concatenate([indices_se, indices_es], -1)    

    indices_best = np.array([indices[i, x] for i, x in enumerate(np.argmin(distances, axis=1))])

    

    indices = indices_best.flatten()

    accuracy = accuracy_score(y_valid, y_train.iloc[indices])

    scores.append(accuracy)

    print("Fold %i, score: %0.5f, mean distance %0.5f" % (fold, accuracy, np.mean(distances)))

    

    plot_confusion_matrix(y_valid, y_train.iloc[indices], le.classes_, normalize=True)

    

print("Average accuracy %0.5f" % np.mean(scores))