# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import matplotlib.pylab as plt # Plotting

import sklearn # Machine learning models.

from sklearn.naive_bayes import GaussianNB # Gaussian Naive Bayes.

import sklearn.metrics # Area Under the ROC calculations.



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



data = np.loadtxt('/kaggle/input/higgs/train.csv', skiprows=1, delimiter=',')



# Any results you write to the current directory are saved as output.
# Split off validation set for testing.

Xtrain = data[:40000, 1:]

Ytrain = data[:40000, 0:1]

Xvalid = data[40000:, 1:]

Yvalid = data[40000:, 0:1]
# Fit model to train.

model = GaussianNB()

model.fit(Xtrain, Ytrain)



# Make hard predictions.

hard_predictions = model.predict(Xvalid)



# Make probabilistic predictions.

predictions = model.predict_proba(Xvalid)



# Compute AUROC.

val = sklearn.metrics.roc_auc_score(Yvalid, predictions[:,1])

print(f'Validation AUROC: {val}' )



# Plot ROC curve.

fpr, tpr, thresholds = sklearn.metrics.roc_curve(Yvalid, predictions[:,1])

plt.plot(fpr, tpr)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')
# Make probabilistic predictions.

Xtest1 = np.loadtxt('/kaggle/input/higgs/test.csv', skiprows=1, delimiter=',')

predictions = model.predict_proba(Xtest1)

predictions = predictions[:,1:2] # Probability that label=1

N = predictions.shape[0]

assert N == 50000, "Predictions should have lenght 50000."

submission = np.hstack((np.arange(N).reshape(-1,1), predictions)) # Add Id column.

np.savetxt(fname='submission.csv', X=submission, header='Id,Predicted', delimiter=',', comments='')



# Submission can be downloaded from this Kaggle Notebook under Sessions->Data->output->/kaggle/working.