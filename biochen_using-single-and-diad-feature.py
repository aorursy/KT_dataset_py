# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from itertools import islice

import numpy as np

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
label_dict = {"binding site": 1, "non-binding site": 0}

single_dict = {"A":1, "T": 2, "C": 3, "G": 4}

diad_dict = {"AA": 1, "AT": 2, "AC": 3, "AG": 4, 

             "TA": 5, "TT": 6, "TC": 7, "TG": 8,

             "CA": 9, "CT": 10, "CC": 11, "CG": 12,

             "GA": 13, "GT": 14, "GC": 15, "GG": 16,}
feature_array = []

label_array = []

with open("../input/Raw_data.csv", "r") as Raw_data:

    for line in islice(Raw_data, 1, None):

        line = line.rstrip()

        sequence, label = line.split(',')

        label_value = label_dict[label]

        label_array.append(label_value)

        feature_list = []

        for i in range(len(sequence)):

            single = sequence[i]

            feature_value = single_dict[single]

            feature_list.append(feature_value)

        for i in range(len(sequence) - 1):

            diad = sequence[i: i + 2]

            feature_value = diad_dict[diad] 

            feature_list.append(feature_value)

        feature_array.append(feature_list) 



X = np.array(feature_array)

y = np.array(label_array)
knc = KNeighborsClassifier(n_neighbors=20)

abc = AdaBoostClassifier(n_estimators= 101, learning_rate=1.0, random_state=0)

gbc = GradientBoostingClassifier(n_estimators= 103,learning_rate=1.0, max_depth=1, random_state=0)

vc = VotingClassifier(estimators=[('knc', knc), ('abc', abc), ('gbc', gbc)], voting='hard')

scores = cross_val_score(vc, X, y, cv=20)
print(scores.max())

print(scores.mean())

print(scores.min())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)



Classifier_list = [[knc, "darkorange", "KNeighbors"], [abc, "blue", "AdaBoost"], 

                   [gbc, "indigo", "GradientBoosting"]]



for Classifier in Classifier_list:

    probas_ = Classifier[0].fit(X_train, y_train).predict_proba(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])

    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=2, color=Classifier[1], 

             label= "{0} (AUC = {1:0.4f})".format(Classifier[2],roc_auc))



plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', )

plt.xlim([-0.05, 1.05])

plt.ylim([-0.05, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()