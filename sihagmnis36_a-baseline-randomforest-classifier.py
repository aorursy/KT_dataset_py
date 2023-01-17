import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')
targets = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
sub = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')
train.shape, test.shape, targets.shape, sub.shape
encode_values = {"cp_type": {"trt_cp": 0, "ctl_vehicle": 1},
                 "cp_time": {24: 0, 48: 1, 72: 2},
                 "cp_dose": {"D1": 0, "D2": 1}}

train.replace(encode_values, inplace=True)
test.replace(encode_values, inplace=True)
X_train = train.iloc[:,1:].to_numpy()
X_test = test.iloc[:,1:].to_numpy()
y_train = targets.iloc[:,1:].to_numpy()
from skmultilearn.adapt import MLkNN

classifier = MLkNN(k=50)

# train
classifier.fit(X_train, y_train)
# predict
train_pred_probs = classifier.predict_proba(X_train)
test_pred_probs = classifier.predict_proba(X_test)
from sklearn.metrics import log_loss
test_pred_probs.toarray()
log_loss(np.ravel(y_train), np.ravel(train_pred_probs.toarray()))
sub.head(3)
results = pd.DataFrame(columns=sub.columns)
results.sig_id = sub.sig_id
results.head()
results.iloc[:,1:] = test_pred_probs.toarray()
results.to_csv('submission.csv', index=False)
results.head(5)
