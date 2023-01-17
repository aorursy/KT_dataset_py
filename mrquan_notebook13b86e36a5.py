# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler
#read data set

train_file = "../input/train.csv"

test_file = "../input/test.csv"

train = pd.read_csv(train_file)

test = pd.read_csv(test_file)
x_train = train.drop(['species', 'id'], axis=1).values

le = LabelEncoder().fit(train['species'])

y_train = le.transform(train['species'])



x_test = test.drop(['id'], axis=1).values



scaler = StandardScaler().fit(x_train)

x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)
# Build 3 layer DNN with 1024, 512, 256 units respectively.

classifier = tf.contrib.learn.DNNClassifier(hidden_units=[1024,512,256],

n_classes=99)
# Fit model.

classifier.fit(x=x_train, y=y_train, steps = 20)
# Make prediction for test data

y = classifier.predict(x_test)

y_prob = classifier.predict_proba(x_test)
# prepare csv for submission

test_ids = test.pop('id')

submission = pd.DataFrame(y_prob, index=test_ids, columns=le.classes_)

submission.to_csv('submission_log_reg.csv')
submission.head()