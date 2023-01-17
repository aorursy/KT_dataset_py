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
data = pd.read_csv("../input/creditcard.csv")



X = data.drop(["Time", "Class"], axis=1)

y = data.Class.values
from sklearn.ensemble import IsolationForest



isof = IsolationForest(n_estimators=30, random_state=1)

isof.fit(X)



y_score = - isof.decision_function(X)
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score

print("ROC AUC: %0.1f%%" % (roc_auc_score(y, y_score) * 100.))

print("All negative accuracy: %0.2f%%" % (accuracy_score(y, np.zeros_like(y)) * 100.))
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve



fp, tp, thres = roc_curve(y, y_score)

plt.plot(fp, tp, label="Isolation Forest")



fp, tp, thres = roc_curve(y, np.random.rand(len(y)))

plt.plot(fp, tp, label="Random")



plt.xlabel("false positive rate")

plt.ylabel("true positive rate (recall)")

plt.legend()