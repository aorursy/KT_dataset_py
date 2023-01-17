# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")
train.head()
x_test = test.values

x_train, x_val, y_train, y_val = train_test_split(

    train.values[:,1:], train.values[:,0], test_size=0.2)
fig, ax = plt.subplots(4, 4, figsize=(8,8))

for i in range(4):

    for j in range(4):

        ax[i, j].imshow(x_train[i*4+j*4].reshape(28,28), cmap='gray')

        ax[i, j].set_title('label = %s' % (y_train[i*4 + j*4]))

        ax[i, j].set_xticks([])

        ax[i, j].set_yticks([])
lr = LogisticRegression()
lr.fit(x_train, y_train)
from sklearn.metrics import accuracy_score

y_val_pred = lr.predict(x_val)
print("Model accuracy is %0.3f" % (accuracy_score(y_val, y_val_pred)))

print("Confusion Matrix:")

print(confusion_matrix(y_val, y_val_pred))
print("Correctly predicted images:")

x_val_correct = x_val[y_val==y_val_pred,:]

y_val_correct = y_val[y_val==y_val_pred]

y_val_pred_correct = y_val_pred[y_val==y_val_pred]



fig, ax = plt.subplots(4, 4, figsize=(10,10))

for i in range(4):

    for j in range(4):

        ax[i, j].imshow(x_val_correct[i*4+j*4].reshape(28,28), cmap='gray')

        ax[i, j].set_title('Label:%s, pred:%s' % (y_val_correct[i*4+j*4], y_val_pred_correct[i*4+j*4]))

        ax[i, j].set_xticks([])

        ax[i, j].set_yticks([])

print("Incorrectly predicted images:")

x_val_incorrect = x_val[y_val!=y_val_pred,:]

y_val_incorrect = y_val[y_val!=y_val_pred]

y_val_pred_incorrect = y_val_pred[y_val!=y_val_pred]



fig, ax = plt.subplots(4, 4, figsize=(10,10))

for i in range(4):

    for j in range(4):

        ax[i, j].imshow(x_val_incorrect[i*4+j*4].reshape(28,28), cmap='gray')

        ax[i, j].set_title('Label:%s, pred:%s' % (y_val_incorrect[i*4+j*4], y_val_pred_incorrect[i*4+j*4]))

        ax[i, j].set_xticks([])

        ax[i, j].set_yticks([])

preds = lr.predict(x_test)
sample_submission['Label'] = preds

sample_submission.to_csv('submission.csv', index=False)