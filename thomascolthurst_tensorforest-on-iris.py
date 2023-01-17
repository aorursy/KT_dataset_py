import tensorflow as tf

import numpy as np

import pandas as pd

import math

import os

from glob import glob



tf.logging.set_verbosity(tf.logging.DEBUG)



all_data = pd.read_csv("../input/Iris.csv")



train = all_data[::2]

test = all_data[1::2]



print("Training = ")

print(train[:5])



print("Test = ")

print(test[:5])
x_train = train.drop(['Species', 'Id'], axis=1).astype(np.float32).values

label_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

y_train = train['Species'].map(label_map).astype(np.float32).values



print("Training features =")

print(x_train[:5])

print("Training labels =")

print(y_train[:5])
params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(

  num_classes=3, num_features=4, num_trees=50, max_nodes=1000, split_after_samples=50).fill()



print("Params =")

print(vars(params))
# Remove previous checkpoints so that we can re-run this step if necessary.

for f in glob("./*"):

    os.remove(f)

classifier = tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator(

    params, model_dir="./")

classifier.fit(x=x_train, y=y_train)

x_test = test.drop(['Species', 'Id'], axis=1).astype(np.float32).values

y_test = test['Species'].map(label_map).astype(np.float32).values



print("x_test = ")

print(x_test[:5])



print("test labels =")

print(y_test[:5])



y_out = list(classifier.predict(x=x_test))



print(y_out[:5])
n = len(y_test)

out_soft = list(y['classes'] for y in y_out)

out_hard = list(y['probabilities'] for y in y_out)



print("Soft predictions:")

print(out_soft[:5])

print("Hard predictions:")

print(out_hard[:5])



soft_zipped = zip(y_test, out_soft)

hard_zipped = list(zip(y_test, out_hard))



num_correct = sum(1 for p in hard_zipped if p[0] == p[1])

print("Accuracy = %s" % (num_correct / n))
test_ps = list(p[1][int(p[0])] for p in soft_zipped)

print("Probs of real label:")

print(test_ps[:5])

total_log_loss = sum(math.log(p) for p in test_ps)

print("Average log loss = %s" % (total_log_loss / n))
confusion = {x: hard_zipped.count(x) for x in set(hard_zipped)}

print ("Confusion matrix:")

print (confusion)