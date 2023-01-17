from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
import seaborn as sns
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

data = pd.read_csv("../input/train.csv")
data = data.reindex(
    np.random.permutation(data.index))
data.head()
sns.heatmap(data.corr(),annot=True)
data.isnull().sum()
data['Outcome'].hist(bins = 20)
sns.pairplot(data,hue='Outcome')
data[['Pregnancies',
 'Glucose',
 'BloodPressure',
 'SkinThickness',
 'Insulin',
 'BMI',
 'DiabetesPedigreeFunction',
 'Age']].hist(figsize=(16, 10), bins=50, xlabelsize=8, ylabelsize=8);
data=data[data['Pregnancies']<=11]
data=data[data['BMI']>=12]
data=data[data['BloodPressure']>40]
data=data[data['Glucose']>40]
data=data[data['SkinThickness']<60]
data.describe()
def preprocess_features(data):
    
    selected_features = data[
    ["Pregnancies",
     "Glucose",
     "Insulin", 
     "BMI",
     "DiabetesPedigreeFunction",
     "Age"]]
    
    processed_features = selected_features.copy()
    return processed_features

def preprocess_targets(data):
    output_targets = pd.DataFrame()
    # Scale the target to be in units of thousands of dollars.
    output_targets["Outcome"] = (data["Outcome"])
    return output_targets
# Choose the first 12000 (out of 17000) examples for training.
training_examples = preprocess_features(data.head(303))
training_targets = preprocess_targets(data.head(303))

# Choose the last 5000 (out of 17000) examples for validation.
validation_examples = preprocess_features(data.tail(202))
validation_targets = preprocess_targets(data.tail(202))

Complete_examples = preprocess_features(data)
Complete_targets = preprocess_targets(data)

# Double-check that we've done the right thing.
print("Training examples summary:")
display.display(training_examples.describe())
print("Validation examples summary:")
display.display(validation_examples.describe())

print("Training targets summary:")
display.display(training_targets.describe())
print("Validation targets summary:")
display.display(validation_targets.describe())
def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
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
def train_linear_classifier_model(
    learning_rate,
    steps,
    hidden_units,
    batch_size,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
    periods = 45
    steps_per_period = steps / periods
    # Create a linear classifier object.
    my_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)  
    DNN_classifier = tf.estimator.DNNClassifier(
      feature_columns=construct_feature_columns(training_examples),
      hidden_units=hidden_units,
      optimizer=my_optimizer
  )
    # Create input functions.
    training_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets["Outcome"], 
                                          batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets["Outcome"], 
                                                  num_epochs=1, 
                                                  shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                    validation_targets["Outcome"], 
                                                    num_epochs=1, 
                                                    shuffle=False)
    
    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("LogLoss (on training data):")
    training_log_losses = []
    validation_log_losses = []
    for period in range (0, periods):
        # Train the model, starting from the prior state.
        DNN_classifier.train(
        input_fn=training_input_fn,
        steps=steps_per_period
         )
        # Take a break and compute predictions.    
        training_probabilities = DNN_classifier.predict(input_fn=predict_training_input_fn)
        training_probabilities = np.array([item['probabilities'] for item in training_probabilities])
    
        validation_probabilities = DNN_classifier.predict(input_fn=predict_validation_input_fn)
        validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])
    
        training_log_loss = metrics.log_loss(training_targets, training_probabilities)
        validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, training_log_loss))
        # Add the loss metrics from this period to our list.
        training_log_losses.append(training_log_loss)
        validation_log_losses.append(validation_log_loss)
    print("Model training finished.")
    # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.tight_layout()
    plt.plot(training_log_losses, label="training")
    plt.plot(validation_log_losses, label="validation")
    plt.legend()
    return DNN_classifier
DNN_classifier = train_linear_classifier_model(
    learning_rate=0.001,
    steps=800,
    batch_size=80,
    hidden_units=[10, 10,10],
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)
predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                  validation_targets["Outcome"], 
                                                  num_epochs=1, 
                                                  shuffle=False)

validation_predictions = DNN_classifier.predict(input_fn=predict_validation_input_fn)
validation_predictions = np.array([item['probabilities'] for item in validation_predictions])

_ = plt.hist(validation_predictions)
evaluation_metrics = DNN_classifier.evaluate(input_fn=predict_validation_input_fn)

print("AUC on the validation set: %0.2f" % evaluation_metrics['auc'])
print("Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy'])

evaluation_metrics['precision']
evaluation_metrics['recall']
testData = pd.read_csv("../input/test.csv")
testData.head()
testData.isna().sum()
testData['Outcome'] =  0
test_examples = preprocess_features(testData)

test_examples.head()
test_validations = preprocess_targets(testData)
predict_test_input_fn = lambda: my_input_fn(test_examples, 
                                                  test_validations["Outcome"], 
                                                  num_epochs=1, 
                                                  shuffle=False)

test_predictions = DNN_classifier.predict(input_fn=predict_test_input_fn)
test_predictions = np.array([item['probabilities'][1] for item in test_predictions])

_ = plt.hist(test_predictions)
test_predictions
for i in range(len(test_predictions)):
    if test_predictions[i] > 0.50 :
        test_predictions[i] = 1
    if test_predictions[i] < 0.50:
        test_predictions[i] = 0
    if test_predictions[i] == 0.50 :
        test_predictions[i] = 0
test_predictions
testData.head()
#convert float into Int
testData['Outcome'] = test_predictions

testData['Outcome']= testData['Outcome'].apply(np.int)
testData[['Id','Outcome']].to_csv('Submit.csv', index = False)
