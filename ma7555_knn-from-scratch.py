# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import pairwise_distances

from scipy.spatial import distance

from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
train_data.head()
train_data.describe()
train_labels = train_data.Cover_Type.values

test_id = test_data.Id.values



train_data.drop(['Soil_Type7', 'Soil_Type15', 'Id', 'Cover_Type'], axis=1, inplace=True)

test_data.drop(['Soil_Type7', 'Soil_Type15', 'Id'], axis=1, inplace=True)

print(train_data.shape, test_data.shape)
%timeit pairwise_distances(train_data[:150], metric = 'euclidean')

%timeit distance.cdist(train_data[:150], train_data[:150], 'euclidean')
min_max_scaler = MinMaxScaler() # If you did not use the scaler, you will get higher accuracy

train_data = min_max_scaler.fit_transform(train_data)

test_data = min_max_scaler.fit_transform(test_data)



distance_matrix = pairwise_distances(train_data, metric = 'euclidean')

print(distance_matrix.shape)
sorted_distance_index = np.argsort(distance_matrix, axis=1).astype(np.uint16)

print(sorted_distance_index)
sorted_distance_labels = train_labels[sorted_distance_index].astype(np.uint8)

print(sorted_distance_labels)
max_k = 100

k_matrix = np.empty((len(sorted_distance_labels), 0), dtype=np.uint8)

for k in range (1, max_k+1):

    k_along_rows = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=sorted_distance_labels[:, 1:k+1]).reshape(len(sorted_distance_labels), -1)

    k_matrix = np.hstack((k_matrix, k_along_rows))

print(k_matrix)
k_truth_table = np.where(k_matrix == train_labels[:, None], 1, 0)

print(k_truth_table)

print(k_truth_table.shape)
accuracy_per_k = np.sum(k_truth_table, axis=0)/len(k_truth_table)

best_accuracy = np.amax(accuracy_per_k)

best_k = np.argmax(accuracy_per_k) + 1 # real k = index + 1

print('Best K: {0}, Best Accuracy: {1:4.2f}%'.format(best_k, best_accuracy*100))

plt.plot(range(1, max_k+1), accuracy_per_k)

plt.title('Classification accuracy vs Choice of K')

plt.xlabel('K')

plt.ylabel('Classification Accuracy')

plt.show()
print("RAM needed for the distance matrix = {:.2f} GB".format(len(train_data)*len(test_data) * 64 / (8 * 1024 * 1024 * 1024)))
# Those variables are no longer needed, Free up some RAM instead

del k_truth_table

del k_matrix

del sorted_distance_labels

del sorted_distance_index

del distance_matrix
# ALERT: This code takes some time, it took 8 minutes on a powerful PC but with relatively low RAM usage (around 6.8G)

def classify(unknown, dataset, labels, k):

    classify_distance_matrix = pairwise_distances(unknown, dataset, metric='euclidean')

    nearest_images = np.argsort(classify_distance_matrix)[:, :k]

    nearest_images_labels = labels[nearest_images]

    classification = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=nearest_images_labels[:, :k])

    return classification.astype(np.uint8).reshape(-1, 1)



predict = np.empty((0, 1), dtype=np.uint8)

chunks = 15

last_chunk_index = 0

for i in range(1, chunks+1):

    new_chunk_index = int(i * len(test_data) / chunks)

    predict = np.concatenate((predict, classify(test_data[last_chunk_index : new_chunk_index], train_data, train_labels, best_k)))

    last_chunk_index = new_chunk_index

    print("Progress = {:.2f}%".format(i * 100 / chunks))
submission = pd.DataFrame({"Id": test_id, "Cover_Type": predict.ravel()})

submission.to_csv('submission.csv', index=False)