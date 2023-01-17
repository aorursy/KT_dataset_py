import numpy as np
import pandas as pd
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from imblearn.combine import SMOTEENN 
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
df = pd.read_csv("../input/creditcard.csv")
df.head()
df.describe()
# visualize histograms of distribution of each feature, comparing fraud (red) vs genuine (blue)
for column in df.iloc[:,1:29].columns:
    plt.figure(figsize=(16,4))
    sns.distplot(df[column][df.Class == 1], bins=60, color= '#FF2731')
    sns.distplot(df[column][df.Class == 0], bins=60, color = '#349EB8')
    plt.grid(True)
    plt.title('histogram of feature: ' + str(column))
    
    plt.show()
    
    
# separate the data into fraud and genuine and then again into test sets and training sets
X = df.loc[:, df.columns != 'Class']
y = df.Class
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size =.2, random_state=42)
type(X_test)

# use SMOTEENN from imblear
sme = SMOTEENN(random_state=33)
X_train, y_train = sme.fit_sample(X_train, y_train)
print(type(X_train))
print(y_train.shape)
# preprocess the data to have std = 1 and mean = 0 for each feature. This helps with tensorflow models. 
X_train[:,0] = (X_train[:,0] - X_train[:,0].mean()) / X_train[:,0].std()
X_test = X_test.values
X_test[:,0] = (X_test[:,0] - X_test[:,0].mean()) / X_test[:,0].std()

X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

features_temp = df.columns.values
features_temp = np.delete(features_temp, 30)
X_train = pd.DataFrame(data = X_train, columns=features_temp)
X_test = pd.DataFrame(data = X_test, columns=features_temp)
y_train = pd.DataFrame(data = y_train, columns = ["Class"])
X_train.describe()
X_test.describe()
my_feature_columns = []
for key in X_train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
# Build 2 hidden layer DNN with 10, 10 units respectively.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 10 nodes each.
    hidden_units=[10, 10],
    # The model must choose between 3 classes.
    n_classes=3)
# Parameters
train_steps = 500 
batch_size = 1000

def train_input_fn(features, labels, batch_size):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(500000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

classifier.train(input_fn=lambda:train_input_fn(X_train, y_train, batch_size), steps =train_steps)
batch_size = 100
def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

# Evaluate the model.
eval_result = classifier.evaluate(input_fn=lambda:eval_input_fn(X_test, y_test, batch_size))
eval_result
