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
import math

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
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
train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")
train_data.head()
test_data.head()
n= len(train_data)

n_test = len(test_data)
m= len(train_data.iloc[0]) - 1 #since first column is label

train_label = train_data[[0]]

train_label.head()
train = train_data.ix[:,1:785]

train.head()
train = train_data.ix[:,1:785]
forest = RandomForestClassifier(criterion = 'entropy',

                                    max_features = 'auto',

                                    n_estimators = 10,

                                    min_samples_split = 2,

                                    min_samples_leaf = 1,

                                    class_weight = 'auto',

                                    random_state = 1,

                                    n_jobs = 8)
forest.fit(train,train_label)
train_predict = forest.predict(train)
train_pred = forest.predict(train)
training_error_list = list()

error = (train_label != train_pred).sum()*100.00/n

training_error_list.append(error)

print('Misclassification error for Training: %0.1f %%' % error) 