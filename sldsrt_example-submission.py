# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

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
import numpy as np # linear algebra
import matplotlib.pylab as plt # Plotting
import sklearn # Machine learning models.
from sklearn.naive_bayes import GaussianNB # Gaussian Naive Bayes.
import sklearn.metrics # Area Under the ROC calculations.

# Load data. This is the filename if running on kaggle.com. Otherwise change this.
filename = '/kaggle/input/higgs-boson-detection/train.csv'
data = np.loadtxt(filename, skiprows=1, delimiter=',')

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
filename = '/kaggle/input/higgs-boson-detection/test.csv' # This is the path if running on kaggle.com. Otherwise change this.
Xtest1 = np.loadtxt(filename, skiprows=1, delimiter=',', usecols=range(1,29))
predictions = model.predict_proba(Xtest1)
predictions = predictions[:,1:2] # Predictions has two columns. Get probability that label=1.
N = predictions.shape[0]
assert N == 50000, "Predictions should have length 50000."
submission = np.hstack((np.arange(N).reshape(-1,1), predictions)) # Add Id column.
np.savetxt(fname='submission.csv', X=submission, header='Id,Predicted', delimiter=',', comments='')

# If running on Kaggle.com, submission.csv can be downloaded from this Kaggle Notebook under Sessions->Data->output->/kaggle/working.