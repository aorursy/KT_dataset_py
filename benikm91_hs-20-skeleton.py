import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
train_data = pd.read_csv("/kaggle/input/machine-learning-lab-cas-data-science-hs-20/train-data.csv", index_col=0)
train_data.shape
train_data.describe()
X_train = train_data.drop(columns='G3')
y_train = train_data['G3']

X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.1)
# TODO 
# TODO
y_dev_pred = np.full(X_dev.shape[0], y_train.mean())
mean_absolute_error(y_dev, y_dev_pred)
X_test = pd.read_csv("/kaggle/input/machine-learning-lab-cas-data-science-hs-20/test-data.csv", index_col=0)
X_test.describe()
y_test_pred = np.full(X_test.shape[0], y_train.mean())
X_test_submission = pd.DataFrame(index=X_test.index)
X_test_submission['G3'] = y_test_pred
X_test_submission.to_csv('baseline_submission.csv', header=True, index_label='id')