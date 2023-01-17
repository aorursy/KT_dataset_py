import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, precision_recall_curve, plot_precision_recall_curve

sns.set()

try:
    os.environ['KAGGLE_DATA_PROXY_TOKEN']
except KeyError:
    path = "creditcard.csv"
else:
    path = "/kaggle/input/creditcardfraud/creditcard.csv"
data = pd.read_csv(path)
data.shape
data
data[data.Class == 1].shape
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_ix, test_ix in split.split(data, data["Class"]):
    data_train = data.iloc[train_ix]
    data_test = data.iloc[test_ix]

test_labels = data_test["Class"]
data_test = data_test.loc[:,:"Class"]

# split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
# for train_ix, devtest_ix in split.split(data_train, data_train["Class"]):
#     data_train = data_train.iloc[train_ix]
#     data_devtest = data.iloc[devtest_ix]

# devtest_labels = data_test["Class"]
# data_devtest = data_test.loc[:,:"Class"]
    
train_labels = data_train["Class"]
data_train = data_train.loc[:,:"Class"]
knb_clf = KNeighborsClassifier()
knb_clf.fit(data_train, train_labels)
cross_val_score(knb_clf, data_train, train_labels, cv=10, scoring="recall")
probas = knb_clf.predict_proba(data_train)
precisions, recalls, thresholds = precision_recall_curve(train_labels, probas[:,1])
plot_precision_recall_curve(knb_clf, data_train, train_labels)
plt.plot(thresholds, precisions[:-1], c="r")
plt.plot(thresholds, recalls[:-1], c="b")
args = np.argmax(recalls > 0.9)
mod_thresh = thresholds[args]
pred_highrecall = (probas[:,1] >= mod_thresh)
recall_score(train_labels, pred_highrecall)
precision_score(train_labels, pred_highrecall)
accuracy_score(train_labels, pred_highrecall)
confusion_matrix(train_labels, pred_highrecall)
probas_test = knb_clf.predict_proba(data_test)
pred_highrecall_test = (probas_test[:,1] >= mod_thresh)
confusion_matrix(test_labels, pred_highrecall_test)