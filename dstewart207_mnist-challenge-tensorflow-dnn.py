import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
%matplotlib inline
# Load the training and test sets with Pandas
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# Look at the training data
train.head()
train.describe()
plt.figure(figsize=(20,10))

plt.hist(train['label'], color='green', alpha=0.4)
plt.xlabel('Digit Label')
plt.ylabel('Training examples')
plt.grid(True)
plt.axis([0, 9, 0, 5000])
# Collect the training data and target. Scale the training data for better results.
train_data = train[train.columns[1:784+1]] # selection is exclusive
train_target = train[train.columns[0]]
train_data_scaled = train_data / 255
# Test data has no labels so can be used on its own
test.head()
# Scale the test data by the same amount.
test_scaled = test / 255
# Let's look at a random training example. Change n to see differet samples
n=0
train_sample = train_data.iloc[n].values.reshape(28,28)
plt.imshow(train_sample, cmap='gray')
print(train_target.iloc[n])
# Let's look at a random training example. Change n to see differet samples
n=10
train_sample = train_data.iloc[n].values.reshape(28,28)
plt.imshow(train_sample, cmap='gray')
print(train_target.iloc[n])
n=1001
train_sample = train_data.iloc[n].values.reshape(28,28)
plt.imshow(train_sample, cmap='gray')
print(train_target.iloc[n])
# Modifiy N to get small subset of the training data to create a test/train split.
N=5000
print(N)
train_data_small = train_data[0:N]
train_data_small_scaled = train_data_scaled[0:N]
train_target_small = train_target[0:N]
train_data_small_split, train_data_small_cv, train_target_small_split, train_target_small_cv = train_test_split(train_data_small_scaled, train_target_small)
def get_train_inputs():
    x = tf.constant(train_data)
    y = tf.constant(train_target)
    return x, y
def get_train_inputs_small():
    x = tf.constant(train_data_small_split)
    y = tf.constant(train_target_small_split)
    return x, y
def get_cv_inputs_small():
    x = tf.constant(train_data_small_cv)
    y = tf.constant(train_target_small_cv)
    return x, y
def get_test_inputs():
    x = tf.constant(test)
    return x
# Create the feature column with tensorflow
feature_columns = [tf.contrib.layers.real_valued_column('', dimension=784)]
#Build our classifier
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, 
                                            hidden_units=[200, 100, 60, 30], 
                                            n_classes=10)
# Fit to a small set of data in the interest of time
classifier.fit(x=np.array(train_data_small_split), y=np.array(train_target_small_split), steps=1000)
accuracy_score = classifier.evaluate(input_fn=get_cv_inputs_small, steps=1)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))
classifier_final = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, 
                                                  hidden_units=[200, 100, 60, 30], 
                                                  n_classes=10)
classifier_final.fit(x=np.array(train_data_scaled), y=np.array(train_target), steps=2000)
y_pred = list(classifier_final.predict(test_scaled))
y_pred
test_label = np.arange(1,len(test)+1)

pd.DataFrame({'ImageId': test_label, 'Label':y_pred}).set_index('ImageId').to_csv('mnist_tf_sub.csv')