import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split
train_data = pd.read_csv('/kaggle/input/cas-data-science-hs-19/houses_train.csv', index_col=0)
X_train = train_data.drop(columns='price')

y_train = train_data['price']
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, stratify=X_train['object_type_name'], test_size=0.1)
y_dev_pred = np.random.randint(10000, 2000000, X_dev.shape[0])
def mean_absolute_percentage_error(y_true, y_pred): 

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
mean_absolute_percentage_error(y_dev, y_dev_pred)
X_test = pd.read_csv('/kaggle/input/cas-data-science-hs-19/houses_test.csv', index_col=0)
y_test_pred = np.random.randint(10000, 2000000, X_test.shape[0])
X_test_submission = pd.DataFrame(index=X_test.index)
X_test_submission['price'] = y_test_pred
X_test_submission.to_csv('random_submission.csv', header=True, index_label='id')