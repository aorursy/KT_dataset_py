import pandas as pd

from sklearn import metrics

import numpy as np
train = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

target_columns = ['grapheme_root', 'consonant_diacritic', 'vowel_diacritic']

y_train = train[target_columns].values
def metric(y, p):

    scores = []

    for i in range(3):

        y_true_subset = y[:,i]

        y_pred_subset = p[:,i]

        recalls = []

        for c in set(y_true_subset):

            idx = np.where(y_true_subset==c)

            s = (y_true_subset[idx] == y_pred_subset[idx]).mean()

            recalls.append(s)

        s = np.mean(recalls)

        scores.append(s)

    final_score = np.average(scores, weights=[2,1,1])

    return final_score, scores
r = np.zeros(len(train))

c = np.zeros(len(train))

v = np.zeros(len(train))

x = np.vstack([r,c,v]).T
metric(y_train, x)
_, scores = metric(y_train, x)
# calculate R score

r_lb = 0.5500

(0.25*scores[1] + 0.25*scores[2]) / (-0.5) + (r_lb / 0.5)
# calculate C score

c_lb = 0.2720

(0.5*scores[0] + 0.25*scores[2]) / (-0.25) + (c_lb / 0.25)
# calculate V score

v_lb = 0.2860

(0.25*scores[0] + 0.25*scores[1]) / (-0.25) + (v_lb / 0.25)