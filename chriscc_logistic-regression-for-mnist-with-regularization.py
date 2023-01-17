# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import Normalizer







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import warnings

warnings.filterwarnings('ignore')



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
train.shape
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
%%time

lr = LogisticRegression(solver='lbfgs')

lr.fit(x_train, y_train)



y_val_pred = lr.predict(x_val)

print("Model accuracy is %0.3f" % (accuracy_score(y_val, y_val_pred)))

print("Confusion Matrix:")

print(confusion_matrix(y_val, y_val_pred))



preds = lr.predict(x_test)

sample_submission['Label'] = preds

sample_submission.to_csv('submission_naive.csv', index=False)
%%time

lr_norm = LogisticRegression(solver='lbfgs')



# create a normalizer

scaler = Normalizer()



# train the model with normalized data

# we can use Normalizer().fit_transform() function to normalize the data for training purpose

lr_norm = LogisticRegression(solver='lbfgs')



lr_norm.fit(scaler.fit_transform(x_train), y_train)



# We have to make sure the same transformation applied on training data is applied on validation and test data as well

# For validation and test we can use Normalizer.transform()



y_val_pred = lr_norm.predict(scaler.transform(x_val))

print("Model accuracy with normalization is %0.3f" % (accuracy_score(y_val, y_val_pred)))

print("Confusion Matrix:")

print(confusion_matrix(y_val, y_val_pred))



preds = lr_norm.predict(scaler.transform(x_test))

sample_submission['Label'] = preds

sample_submission.to_csv('submission_norm.csv', index=False)
%%time

# create a normalizer

scaler = Normalizer()



# train the model with normalized data

# we can use Normalizer().fit_transform() function to normalize the data for training purpose

lr_norm = LogisticRegression(solver='lbfgs', C=20)



lr_norm.fit(scaler.fit_transform(x_train), y_train)



# We have to make sure the same transformation applied on training data is applied on validation and test data as well

# For validation and test we can use Normalizer.transform()



y_val_pred = lr_norm.predict(scaler.transform(x_val))

print("Model accuracy with normalization is %0.3f" % (accuracy_score(y_val, y_val_pred)))

print("Confusion Matrix:")

print(confusion_matrix(y_val, y_val_pred))



preds = lr_norm.predict(scaler.transform(x_test))

sample_submission['Label'] = preds

sample_submission.to_csv('submission_norm_l2_C20.csv', index=False)