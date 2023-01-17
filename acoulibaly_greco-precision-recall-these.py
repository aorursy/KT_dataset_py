# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

"""

This module illustrates how to compute Precision at k and Recall at k metrics.

"""



from __future__ import (absolute_import, division, print_function,

                        unicode_literals)

from collections import defaultdict



from surprise import Dataset

from surprise import SVD

from surprise import Reader

from surprise.model_selection import KFold





def precision_recall_at_k(predictions, k=10, threshold=3.5):

    '''Return precision and recall at k metrics for each user.'''



    # First map the predictions to each user.

    user_est_true = defaultdict(list)

    for uid, _, true_r, est, _ in predictions:

        user_est_true[uid].append((est, true_r))



    precisions = dict()

    recalls = dict()

    for uid, user_ratings in user_est_true.items():



        # Sort user ratings by estimated value

        user_ratings.sort(key=lambda x: x[0], reverse=True)



        # Number of relevant items

        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)



        # Number of recommended items in top k

        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])



        # Number of relevant and recommended items in top k

        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))

                              for (est, true_r) in user_ratings[:k])



        # Precision@K: Proportion of recommended items that are relevant

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1



        # Recall@K: Proportion of relevant items that are recommended

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1



    return precisions, recalls





#data = Dataset.load_builtin('ml-100k')

# added by amc 

# Creation of the dataframe. Column names are irrelevant.

# voting procedure rating  where generate randomly  

#itemID = procedure_id (Borda, condorcet, pluralite, black et copeland)

# userID = facilitateur id

# rating = facilitateur rating score

ratings_dict = {'itemID': [2, 5, 1, 3, 1, 1, 1, 4, 4, 2, 1, 5, 5, 4, 4, 5, 5, 1, 4, 1, 5, 1, 2, 5, 5, 4, 4, 4, 3, 3, 5, 4, 5, 5, 5, 3, 4, 2, 5, 5, 5, 5, 1, 5, 4, 3, 2, 2, 2, 5, 4, 1, 4, 4, 1, 2, 1, 2, 1, 2, 2, 4, 5, 5, 4, 2, 1, 3, 1, 4, 4, 1, 1, 4, 1, 2, 4, 1, 2, 2, 4, 3, 4, 4, 5, 2, 3, 3, 2, 3, 3, 3, 4, 4, 5, 2, 4, 4, 5, 2],

                'userID': [45, 32, 9, 45, 23, 9, 23, 23, 9, 2, 23, 32, 9, 23, 32, 45, 32, 32, 32, 32, 45, 23, 9, 9, 45, 23, 9, 2, 2, 23, 2, 2, 45, 9, 45, 32, 23, 2, 45, 32, 9, 32, 23, 23, 45, 32, 2, 9, 9, 23, 45, 45, 23, 45, 32, 23, 2, 9, 45, 32, 45, 23, 23, 45, 23, 23, 9, 32, 9, 23, 32, 2, 2, 32, 23, 45, 23, 9, 9, 32, 45, 9, 23, 45, 32, 32, 9, 23, 9, 45, 23, 32, 45, 2, 32, 2, 2, 2, 23, 23],

                'rating': [3, 4, 3, 3, 5, 4, 3, 5, 5, 3, 4, 4, 5, 5, 4, 3, 3, 5, 5, 4, 4, 3, 5, 4, 3, 3, 4, 4, 5, 3, 5, 3, 4, 5, 4, 3, 3, 4, 5, 3, 3, 4, 4, 5, 4, 5, 5, 3, 4, 3, 5, 3, 5, 4, 4, 3, 3, 5, 5, 5, 5, 4, 3, 5, 5, 5, 4, 3, 4, 4, 4, 3, 3, 4, 3, 4, 4, 4, 3, 3, 3, 5, 5, 4, 5, 3, 5, 3, 5, 4, 5, 4, 3, 4, 4, 3, 5, 5, 5, 5]}

df = pd.DataFrame(ratings_dict)

reader = Reader(rating_scale=(1, 5))

# The columns must correspond to user id=facilitator id, item id=voting procedure id and ratings (in that order).

data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)





kf = KFold(n_splits=5)

algo = SVD()



# Plotting data for greco

import matplotlib.pyplot as plt

Greco_presion = []

Greco_recall =  []

Greco_F_mesure = []

for trainset, testset in kf.split(data):

    algo.fit(trainset)

    predictions = algo.test(testset)

    precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=4)



    # Precision and recall can then be averaged over all users

    print('Greco Présicion =')

    print(sum(prec for prec in precisions.values()) / len(precisions))

    greco_p = sum(prec for prec in precisions.values()) / len(precisions)

    Greco_presion.append(greco_p)

    print('Greco Recall =')

    print(sum(rec for rec in recalls.values()) / len(recalls),)

    greco_re = sum(rec for rec in recalls.values()) / len(recalls)

    Greco_recall.append(greco_re)

    f_mesure = 2*((greco_p*greco_re)/(greco_p+greco_re))

    print('Greco F_mesure =',f_mesure)

    Greco_F_mesure.append(f_mesure)



plt.subplot(211)

plt.plot(Greco_presion, label="Précision")

plt.plot(Greco_recall, label=" Rappel")

plt.plot(Greco_F_mesure, label="F_mesure")

plt.legend(loc=3,

           ncol=3, mode="expand", borderaxespad=0.)

plt.ylim(0.0, 1.0)

plt.title('GRECO: Mesure de prévision d\'utilisation')

plt.show()
