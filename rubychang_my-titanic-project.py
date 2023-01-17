# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
train_file_path = '../input/train.csv'
test_file_path = '../input/test.csv'

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)


# Any results you write to the current directory are saved as output.
train_data.describe()
train_data.head(5)
print("missing values: (train_data)")
print(train_data.isnull().sum().sort_values(ascending=False))

print("\nmissing values: (test_data)")
print(test_data.isnull().sum().sort_values(ascending=False))
def distplot(titanic_dataframe, cols, filter_zero=False):
    num_col = 3
    num_row = math.ceil(len(cols) / num_col)
    plt.figure(figsize=(6 * num_col, 5 * num_row))
    for i, col in enumerate(cols, 1):
        plt.subplot(num_row, num_col, i)
        if filter_zero:
            sns.distplot(titanic_dataframe[titanic_dataframe[col] > 0][col], fit=norm);
        else:
            sns.distplot(titanic_dataframe[col], fit=norm);

distplot(train_data.fillna(0), ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'])
def process_age(df,cut_points,label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_cat"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

cut_points = [-1,0,5,16,22,35,60,100]
label_names = ["Missing","Infant","Child","Teenager","YoungAdult","Adult","Senior"]

train_data = process_age(train_data,cut_points,label_names)
test_data = process_age(test_data,cut_points,label_names)


train_data['Fare'] = np.log1p(train_data['Fare'])
test_data['Fare'] = np.log1p(test_data['Fare'])

train_x = train_data[['Pclass', 'Sex', 'Age_cat', 'SibSp', 'Parch', 'Fare','Embarked']]
train_y = train_data[['Survived']]
test_x = test_data[['Pclass', 'Sex', 'Age_cat', 'SibSp', 'Parch', 'Fare','Embarked']]

train_x = pd.get_dummies(train_x)
test_x = pd.get_dummies(test_x)

train_x.head(10)
num_sample = len(train_x)
print(num_sample)
#split training samples v.s. validation samples
num_sample = len(train_x)
train_num = int(num_sample * 0.7)
validate_num = num_sample - train_num

train_examples = train_x.head(train_num)
train_targets = train_y.head(train_num)


validate_examples = train_x.tail(validate_num)
validate_targets = train_y.tail(validate_num)
from tensorflow.python.data import Dataset
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(100)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels
import tensorflow as tf

def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features])

my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

feature_columns = construct_feature_columns(train_examples)

classifier = tf.estimator.LinearClassifier(
  feature_columns = feature_columns,
  optimizer=my_optimizer,
)

classifier.train(
  input_fn=lambda: my_input_fn(train_examples,train_targets),
  steps=1000)

evaluation_metrics = classifier.evaluate(
  input_fn=lambda: my_input_fn(train_examples,train_targets),
  steps=1000)
print("Training set metrics:")
for m in evaluation_metrics:
  print(m, evaluation_metrics[m])
print("---")

evaluation_metrics = classifier.evaluate(
  input_fn=lambda: my_input_fn(validate_examples,validate_targets),
  steps=1000)
print("Validation set metrics:")
for m in evaluation_metrics:
  print(m, evaluation_metrics[m])
print("---")

pred_li = classifier.predict(lambda: my_input_fn(test_x,test_x["Fare"],1,True,1))
pred_li = np.array([item['class_ids'][0] for item in pred_li])
linear_result = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': pred_li})
linear_result.head(5)

linear_result.to_csv('Titanic_Linear.csv', index=False)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestClassifier(max_depth=8,random_state=50)
forest_model.fit(train_examples, train_targets)
melb_preds = forest_model.predict(validate_examples)
print(mean_absolute_error(validate_targets, melb_preds))
pred = forest_model.predict(test_x.fillna(0))
solution = pd.DataFrame({"id":test_data.PassengerId, "Survived":pred})
solution.head(5)

solution.to_csv("Titanic_Dicision.csv", index = False)
my_optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

estimator = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[20, 10],
    optimizer=my_optimizer
    )

estimator.train(
  input_fn=lambda: my_input_fn(train_examples,train_targets),
  steps=1000)

evaluation_metrics = estimator.evaluate(
  input_fn=lambda: my_input_fn(train_examples,train_targets),
  steps=1000)
print("Training set metrics:")
for m in evaluation_metrics:
  print(m, evaluation_metrics[m])
print("---")

evaluation_metrics = estimator.evaluate(
  input_fn=lambda: my_input_fn(validate_examples,validate_targets),
  steps=1000)
print("Validation set metrics:")
for m in evaluation_metrics:
  print(m, evaluation_metrics[m])
print("---")
nn_result = estimator.predict(lambda: my_input_fn(test_x,test_x["Fare"],1,True,1))
nn_result = np.array([prediction["class_ids"][0] for prediction in nn_result])
neural_result = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': nn_result})
neural_result.head(5)

neural_result.to_csv('Titanic_Neural.csv', index=False)