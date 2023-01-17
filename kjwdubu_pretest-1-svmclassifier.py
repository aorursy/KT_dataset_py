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
test_csv_path = "/kaggle/input/sejongai-challenge-pretest-1/test_data.csv"

train_csv_path = "/kaggle/input/sejongai-challenge-pretest-1/train.csv"

submit_csv_path = "/kaggle/input/sejongai-challenge-pretest-1/submit_sample.csv"



test_csv = pd.read_csv(test_csv_path)

train_csv = pd.read_csv(train_csv_path)

submit_csv = pd.read_csv(submit_csv_path)
# train & test



train_all_x = np.asarray(train_csv.iloc[:, 1:-1])

train_all_y = np.asarray(train_csv.iloc[:, -1:]).squeeze(1)



test_x = np.asarray(test_csv.iloc[:,1:])



# valid



val_test_mask = np.arange(0, train_all_x.shape[0], 10)

val_train_mask = np.setdiff1d(np.arange(0, train_all_x.shape[0], 1), val_test_mask)



train_val_x = train_all_x[val_train_mask]

train_val_y = train_all_y[val_train_mask]



test_val_x = train_all_x[val_test_mask]

test_val_y = train_all_y[val_test_mask]





print("#----------info----------#")

print("train set : {}".format(train_all_x.shape[0]))

print("test set : {}".format(test_x.shape[0]))

print("train set for validation : {}".format(train_val_x.shape[0]))

print("test set for validation : {}".format(test_val_x.shape[0]))
from sklearn import svm, metrics



# Create a classifier: a support vector classifier

classifier = svm.SVC(gamma=0.001)



classifier.fit(train_val_x, train_val_y)



predicted = classifier.predict(test_val_x)



print(metrics.accuracy_score(test_val_y, predicted))
classifier.fit(train_all_x, train_all_y)



predicted = classifier.predict(test_x)



submit_csv['Label'] = predicted

submit_csv['Label'] = submit_csv['Label'].astype("int")

submit_csv.to_csv("/kaggle/working/submit.csv", index=False)