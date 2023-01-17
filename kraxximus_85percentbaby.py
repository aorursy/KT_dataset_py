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

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

titanic_df = pd.read_csv("../input/train.csv")
titanic_df = titanic_df.reindex(np.random.permutation(titanic_df.index)) #randomize dat shi
# titanic_df
# titanic_df["Parch"].hist()
def preprocess_features(titanic_df):
    selected_features = titanic_df[[
        "Pclass",
#         "Name",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
#         "Ticket",
        "Fare",
#         "Cabin",
        "Embarked",
    ]]
    
    #PROCESS EM
    processed_features = selected_features.copy()
    processed_features["Age"] = processed_features["Age"].fillna(
        value=titanic_df["Age"].mean()
    ).astype(float)
    processed_features["Embarked"] = processed_features["Embarked"].replace(
        np.nan, "S", regex=True
    )
    processed_features["Family"] = (
        processed_features["SibSp"] + processed_features["Parch"]
    )
    return processed_features

def preprocess_targets(titanic_df):
    output_targets = pd.DataFrame()
    output_targets["Survived"] = titanic_df["Survived"]
    return output_targets

def construct_feature_columns(input_features):

    sibsp = tf.feature_column.numeric_column("SibSp")
    parch = tf.feature_column.numeric_column("Parch")
    family = tf.feature_column.numeric_column("Family")
    
    pclass = tf.feature_column.categorical_column_with_identity(
        key="Pclass",
        num_buckets=3,
        default_value=0
    )
    sex = tf.feature_column.categorical_column_with_vocabulary_list(
        key="Sex",
        vocabulary_list=["male", "female"],
    )
    fare = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("Fare"),
        boundaries=[10, 50],
    )     
    age = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("Age"),
#         boundaries=[15, 35, 50],
        boundaries=[10,20,30,40]
    )
    embarked = tf.feature_column.categorical_column_with_vocabulary_list(
        key="Embarked",
        vocabulary_list=["C", "S", "Q"],
    )
    sexage = tf.feature_column.crossed_column(
        set([sex, age]),
        hash_bucket_size=8
    )
    sexfare = tf.feature_column.crossed_column(
        set([sex, fare]),
        hash_bucket_size=6
    )
    feature_columns = set([
        pclass,
        parch,
        sibsp,
#         family, # grouped parch & sibsp
        sex,
        fare,
        age,
#         sexage, # crossed sex & age
#         sexfare, # crossed sex & fare
#         embarked
    ])
    
    return feature_columns
training_examples = preprocess_features(titanic_df.head(670))
training_targets = preprocess_targets(titanic_df.head(670))

validation_examples = preprocess_features(titanic_df.tail(220))
validation_targets = preprocess_targets(titanic_df.tail(220))

# # Double-check that we've done the right thing.
# print("Training examples summary:")
# display.display(training_examples.describe())
# print("Validation examples summary:")
# display.display(validation_examples.describe())

# print("Training targets summary:")
# display.display(training_targets.describe())
# print("Validation targets summary:")
# display.display(validation_targets.describe())

correlation_dataframe = training_examples.copy()
correlation_dataframe["target"] = training_targets["Survived"]
# correlation_dataframe.corr()
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model.
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
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
    batch_size,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
    
    periods = 10
    steps_per_period = steps / periods
    
    # Create a linear classifier object.
#     my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)  
    linear_classifier = tf.estimator.LinearClassifier(
        feature_columns=construct_feature_columns(training_examples),
        optimizer=my_optimizer
    )
    # Create input functions.
    training_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets["Survived"], 
                                          batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets["Survived"], 
                                                  num_epochs=1, 
                                                  shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                    validation_targets["Survived"], 
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
        linear_classifier.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        # Take a break and compute predictions.    
        training_probabilities = linear_classifier.predict(input_fn=predict_training_input_fn)
        training_probabilities = np.array([item['probabilities'] for item in training_probabilities])

        validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
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

    return linear_classifier
linear_classifier = train_linear_classifier_model(
    learning_rate=0.0075,
    steps=20000,
#     batch_size=65,
    batch_size=20,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                  validation_targets["Survived"], 
                                                  num_epochs=1, 
                                                  shuffle=False)

evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)
print("AUC on the validation set: %0.2f" % evaluation_metrics['auc'])
print("Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy'])

test_data = pd.read_csv("../input/test.csv")
test_data["Survived"] = pd.Series([])

test_examples = preprocess_features(test_data)
test_targets = test_data["Survived"]

predict_testing_input_fn = lambda: my_input_fn(test_examples, 
                                               test_targets, 
                                               num_epochs=1, 
                                               shuffle=False
                                              )
# test_predictions = linear_regressor.predict(input_fn=predict_testing_input_fn)
test_predictions = linear_classifier.predict(input_fn=predict_testing_input_fn)
# print([list(test_predictions)])
# test_predictions = np.array([int(item['classes']) for item in test_predictions])
# print(test_data["Survived"])
test_data["Survived"] = np.array([int(item['classes']) for item in test_predictions])
print(test_data["Survived"])
test_data.to_csv(
    path_or_buf="answer.csv",
    index=False,
    columns=["PassengerId", "Survived"]
)

# print(test_predictions)

# root_mean_squared_error = math.sqrt(
#     metrics.mean_squared_error(test_predictions, test_targets))

# print("Final RMSE (on test data): %0.2f") % root_mean_squared_error
