import numpy as np
import pandas as pd
from numpy import genfromtxt
from sklearn.neighbors import KDTree

import numpy as np 
import pandas as pd 

from subprocess import check_output
print(check_output(["ls", "../input"]).decode('utf8'))

data_sample = "../input/sample_submission.csv"
test_sample = "../input/test.csv"
train_sample = "../input/train.csv"

train = genfromtxt(train_sample, delimiter=',')
test_data = genfromtxt(test_sample, delimiter=',')
submssion = genfromtxt(data_sample, delimiter=',')
file = pd.read_csv(data_sample)
data = pd.read_csv(train_sample)

print('Shape of train: ', np.shape(train))
print('Shape of test_data: ', np.shape(test_data))
print('Shape of submission: ', np.shape(submssion))
train_data = train[1:,1:]
train_labels = train[1:,0:1]
test_data = test_data[1:,:] 
kd_tree = KDTree(train_data)
test_neighbors = np.squeeze(kd_tree.query(test_data, k=1, return_distance=False))
kd_tree_predictions = train_labels[test_neighbors]
kd_tree_predictions = kd_tree_predictions.astype(int)
del file['Label']
file.insert(1, 'Label', kd_tree_predictions )
file.to_csv('Kaggle.csv', index=False)