# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from math import log

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
df_test = pd.read_csv('../input/test.csv',
                     usecols=['test_id'])
df_test['is_duplicate'] = 0.5
df_test.to_csv('all-half.csv', index=False)
df_test.head()
log(2)
eps = 1E-6
xs = np.linspace(0 + eps, 1 - eps)
y1s = -np.log(xs)
y0s = -np.log(1 - xs)


plt.figure(figsize=(12, 5))
plt.plot(xs, y1s, label="is duplicate $-\log(p)$")
plt.plot(xs, y0s, label="isn't duplicate: $-\log(1-p)$")
plt.legend(loc='upper center')
axes = plt.gca()
axes.set_ylim([0,2])
plt.title('log loss in cas of a duplicate or non duplicate question')
plt.xlabel('Predicted probability')
plt.ylabel('log loss')
df_train = pd.read_csv('../input/train.csv',
                       usecols=['is_duplicate'])
df_train['is_duplicate'].sum()/ df_train.shape[0]
def create_array(val, df=df_train):
    """Return a constant array with value val of same length as df"""
    return val * np.ones_like(df_train.index)
    
ll = log_loss(df_train["is_duplicate"], create_array(0.2))
ll
ratio = (ll + log(0.8)) / (log(0.8) -log(0.2))
ratio
xs = np.linspace(0, 1, 100)
lls = [log_loss(df_train["is_duplicate"], create_array(x)) for x in xs]
# find minimum
min_index = np.where(lls == np.min(lls))[0][0]
x_min = xs[min_index]
y_min = lls[min_index]

plt.figure(figsize=(12, 5))
plt.plot(xs, lls, '-gD', markevery=[min_index])

plt.annotate('minimum ({:.3f}, {:.3f})'.format(x_min, y_min), xy=(x_min, y_min), xytext=(x_min, y_min - 0.1))
plt.title('log loss vs predicted constant probability')
plt.xlabel('Predicted probability')
plt.ylabel('log loss')
axes = plt.gca()
axes.set_ylim([0,2])
print()
log_loss(df_train["is_duplicate"], create_array(ratio))
df_test = pd.read_csv('../input/test.csv',
                      usecols=['test_id']
                      )
df_test['is_duplicate'] = 0.2
df_test.to_csv('submission-0.2.csv', index=False)
ll = 0.46473
ratio = (ll + log(0.8)) / (log(0.8) -log(0.2))
ratio
df_test['is_duplicate'] = ratio
df_test.to_csv('submission-ratio.csv', index=False)
