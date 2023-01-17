from sklearn.metrics import log_loss

import numpy as np



y_pred = np.ones(4000)*0.8

y_true = np.ones(4000)

n = 2000

for i in range(n-1, n):

    y_true[-(i+1):] = 0

    loss = log_loss(y_true, y_pred)

    print('{} zeros in 4000 data, the loss is {}'.format(i+1, loss))
import pandas as pd

sample_submission = pd.read_csv("../input/deepfake-detection-challenge/sample_submission.csv")

sample_submission['label'] = 0.8

sample_submission.to_csv('submission.csv', index=False)

!ls