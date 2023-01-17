# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow.python.data import Dataset
from sklearn import preprocessing
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train.csv")
df = df.reindex(np.random.permutation(df.index))
df = df.fillna(value = df['Age'].mean())

input_features = df[['Pclass','Age','Sex','Fare','Parch','SibSp']]
input_features.head()
targets = df['Survived']
sex_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key='Sex',
        vocabulary_list=["male", "female"])
pclass_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key='Pclass',
        vocabulary_list=[1, 2, 3])
indicator_cols = [tf.feature_column.indicator_column(sex_col), tf.feature_column.indicator_column(pclass_col)]

numeric_features = ['Age','Fare','Parch','SibSp']
numeric_cols = [tf.feature_column.numeric_column(my_feature) for my_feature in numeric_features]
feature_cols = indicator_cols + numeric_cols
def my_input_fn(features, targets, batch_size=10, shuffle=True, num_epochs=None):

    if targets is None:
        features = {key:np.array(value) for key,value in dict(features).items()} 
        ds = Dataset.from_tensor_slices(features)
        ds = ds.batch(batch_size)
        return ds
    else:
        
        # Convert pandas data into a dict of np arrays.
        features = {key:np.array(value) for key,value in dict(features).items()}                                            

        # Construct a dataset, and configure batching/repeating.
        ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
        ds = ds.batch(batch_size).repeat(num_epochs)

        # Shuffle the data, if specified.
        if shuffle:
          ds = ds.shuffle(10000)

        # Return the next batch of data.
        features, labels = ds.make_one_shot_iterator().get_next()
        return features, labels
my_optimizer = tf.train.AdamOptimizer(learning_rate = 0.005)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
classifier = tf.estimator.DNNClassifier(
    feature_columns = feature_cols,
    optimizer = my_optimizer,
    hidden_units = [10, 10, 10, 10, 10],)
classifier.train(
    input_fn = lambda: my_input_fn(input_features, targets),
    steps = 5000)
evaluation_metrics = classifier.evaluate(input_fn = lambda: my_input_fn(input_features, targets, shuffle = False, num_epochs = 1))

print("AUC on the validation set: %0.2f" % evaluation_metrics['auc'])
print("Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy'])
test_df = pd.read_csv("../input/test.csv")
test_df = test_df.fillna(value = test_df['Age'].mean())
test_features = test_df[['Pclass','Age','Sex','Fare','Parch','SibSp']]
test_features
predictions = classifier.predict(
    input_fn = lambda: my_input_fn(test_features, None, shuffle = False, num_epochs = 1, batch_size = 1))

export_df = pd.DataFrame([item['probabilities'][1] for item in predictions])

export_df = export_df.round(0)

#pd.options.display.float_format = '{:,.0f}'.format
export_df.reset_index(level=0, inplace=True)
export_df = export_df.rename({"index": "PassengerId", 0: 'Survived'}, axis='columns')
export_df['Survived'] = pd.to_numeric(export_df['Survived'], errors='raise', downcast='integer')
export_df.to_csv(header = ["PassengerId",'Survived'],index = False, float_format = '{:,.0f}')