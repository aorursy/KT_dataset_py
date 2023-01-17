import numpy as np, pandas as pd, matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

from sklearn.model_selection import cross_val_score, cross_val_predict

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

test = pd.read_csv('../input/nlp-getting-started/test.csv')
train = pd.read_csv('../input/nlp-getting-started/train.csv')

print('\nTrain shape: ', train.shape)
print('Test shape: ', test.shape)
train[10:200]
# exploring volcano category

temp_df = test[test['keyword'].isin(['volcano'])]
temp_df
from sklearn.base import BaseEstimator

class FindTrueDisasters(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict (self, X):
        return np.zeros((len(X), 1), dtype=bool)
X_train = train[:5614].drop('target', axis=1) # .. added one more value because there is one more in my testing set somehow
X_test = train[5614:].drop('target', axis=1)
y_train = train['target'].loc[:5613]
y_test = train['target'].loc[5613:]


y_train_ftd = (y_train == True)
y_test_ftd = (y_test == True)
y_train_ftd
ftd_clf = FindTrueDisasters()

print(cross_val_score(ftd_clf, X_train, y_train_ftd, scoring="accuracy"))

# accuracy of 60%, analyze scatter plots ? any possible outliers ? looking for further clues
# RESOURCES // 

# https://www.kaggle.com/db102291/disaster-tweet-prediction-exploratory-analysis
# NOTES

# explore: detect level of shock upon tweeting .. compare that to other parameters
# explore: detect location based on tweet and language
# explore: classify by type of disaster: heat wave, 