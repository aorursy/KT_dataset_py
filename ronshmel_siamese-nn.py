import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install ivis



!git clone https://github.com/beringresearch/ivis-explain

!pip install --editable  ./ivis-explain
import pandas as pd

import matplotlib.pyplot as plt

import sys

import os



sys.path.append(os.path.abspath('./ivis-explain'))

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, average_precision_score, roc_auc_score, classification_report, roc_curve, precision_recall_curve

from sklearn.linear_model import LogisticRegression



from ivis import Ivis

from ivis_explanations import LinearExplainer



data = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

Y = data['Class']

X = data.drop(['Class','Time'], axis=1)
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, stratify=Y, test_size=0.8, random_state=1234)
# standard_scaler = StandardScaler().fit(train_X[['Time', 'Amount']])

# train_X.loc[:, ['Time', 'Amount']] = standard_scaler.transform(train_X[['Time', 'Amount']])

# test_X.loc[:, ['Time', 'Amount']] = standard_scaler.transform(test_X[['Time', 'Amount']])



standard_scaler = StandardScaler().fit(train_X[['Amount']])

train_X.loc[:, ['Amount']] = standard_scaler.transform(train_X[['Amount']])

test_X.loc[:, ['Amount']] = standard_scaler.transform(test_X[['Amount']])

minmax_scaler = MinMaxScaler().fit(train_X)

train_X = minmax_scaler.transform(train_X)

test_X = minmax_scaler.transform(test_X)
ivis = Ivis(embedding_dims=2, model='maaten',

            k=15, n_epochs_without_progress=5,

            supervision_weight=0.95,

            verbose=0)

ivis.fit(train_X, train_Y.values)
ivis.save_model('ivis-supervised-fraud', overwrite=True)
train_embeddings = ivis.transform(train_X)

test_embeddings = ivis.transform(test_X)
fig, ax = plt.subplots(1, 2, figsize=(17, 7), dpi=200)

ax[0].scatter(x=train_embeddings[:, 0], y=train_embeddings[:, 1], c=train_Y, s=3, cmap='RdYlBu_r')

ax[0].set_xlabel('ivis 1')

ax[0].set_ylabel('ivis 2')

ax[0].set_title('Training Set')



ax[1].scatter(x=test_embeddings[:, 0], y=test_embeddings[:, 1], c=test_Y, s=3, cmap='RdYlBu_r')

ax[1].set_xlabel('ivis 1')

ax[1].set_ylabel('ivis 2')

ax[1].set_title('Testing Set')
clf = LogisticRegression(solver="lbfgs").fit(train_embeddings, train_Y)
labels = clf.predict(test_embeddings)

proba = clf.predict_proba(test_embeddings)
print(classification_report(test_Y, labels))



print('Confusion Matrix')

print(confusion_matrix(test_Y, labels))

print('Average Precision: '+str(average_precision_score(test_Y, proba[:, 1])))

print('ROC AUC: '+str(roc_auc_score(test_Y, labels)))
# retrieve just the probabilities for the positive class

pos_probs = proba[:, 1]



# calculate roc curve for model

fpr, tpr, thresholds = roc_curve(test_Y, pos_probs)



# plot no skill roc curve

plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')

# plot model roc curve

plt.plot(fpr, tpr, marker='.', label='Logistic')

# axis labels

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

# show the legend

plt.legend()

# show the plot

plt.show()
# calculate the no skill line as the proportion of the positive class

no_skill = len(Y[Y==1]) / len(Y)

# plot the no skill precision-recall curve

plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

# calculate model precision-recall curve

precision, recall, _ = precision_recall_curve(test_Y, pos_probs)

# plot the model precision-recall curve

plt.plot(recall, precision, marker='.', label='Logistic')

# axis labels

plt.xlabel('Recall')

plt.ylabel('Precision')

# show the legend

plt.legend()

# show the plot

plt.show()
# create a histogram of the predicted probabilities

plt.hist(pos_probs, bins=100)

plt.show()
explainer = LinearExplainer(ivis)

explainer.feature_importances_(train_X)