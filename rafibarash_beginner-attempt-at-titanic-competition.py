# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# import libraries
import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import sklearn
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt

# Get data
train = pd.read_csv('../input/train.csv')
train = train.reindex(
    np.random.permutation(train.index)
)
test = pd.read_csv('../input/test.csv')
# display.display(train)
# display.display(train.info())
# display.display(test.info())
# display.display(train.head())
# display.display(train.corr())
# _ = display.display(train.hist())
# display.display(train.Pclass)
def eval_input_fn(features, labels=None, batch_size=1):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)
    # convert inputs to DataSet
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    # shuffle, repeat, and batch examples
    return dataset.repeat(1).batch(batch_size)

def my_input_fn(features, targets=None, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a neural network model.
  
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
    features = dict(features)
    if targets is None:
        inputs = features
    else:
        inputs = (features, targets)

    # Construct a dataset, and configure batching/repeating.
    ds = tf.data.Dataset.from_tensor_slices(inputs) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels
def linear_scale(series):
    min_val = series.min()
    max_val = series.max()
    scale = (max_val - min_val) / 2.0
    return series.apply(lambda x:((x - min_val) / scale) - 1.0)

def log_normalize(series):
    return series.apply(lambda x:math.log(x+1.0))

def clip(series, clip_to_min, clip_to_max):
    return series.apply(lambda x:(
        min(max(x, clip_to_min), clip_to_max)))

def z_score_normalize(series):
    mean = series.mean()
    std_dv = series.std()
    return series.apply(lambda x:(x - mean) / std_dv)

def binary_threshold(series, threshold):
    return series.apply(lambda x:(1 if x > threshold else 0))
def preprocess_features(df):
    x = pd.DataFrame(df, columns=['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
    # Fill missing values
    x['Age'].fillna(x['Age'].median(), inplace=True)
    x['Fare'].fillna(x['Fare'].median(), inplace=True)
    x['Embarked'].fillna(x['Embarked'].mode()[0], inplace=True)
    # Add some synthetic features
    x['FamilySize'] = x['Parch'] + x['SibSp']
    x['IsAlone'] = x['FamilySize'].apply(lambda n: 1 if n == 0 else 0)
    x['Title'] = x['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0] # get prefix title from name
    x['Title'] = x['Title'].apply(lambda s: 'Misc' if x.Title.value_counts()[s] < 10 else s) # make prefix = 'Misc' if >10
    # Bin a couple features too
    bucket_labels=['0', '1', '2', '3', '4', '5']
    x['AgeBin'] = pd.cut(x['Age'].astype(int), 6, labels=bucket_labels)
    x['FareBin'] = pd.qcut(x['Fare'].astype(int), 6, labels=bucket_labels)
    x['FamilySizeBin'] = pd.cut(x['FamilySize'].astype(int), 6, labels=bucket_labels)
    # One-hot encode some features
    x = pd.get_dummies(x, columns=['Sex', 'Embarked', 'AgeBin', 'FareBin', 'FamilySizeBin'])
    # Drop bad features
    x.drop(['Name', 'SibSp', 'Parch', 'Age', 'Fare', 'Title'], axis=1, inplace=True)
    return x

def preprocess_targets(df):
    return df['Survived']

def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features])
def train_dnn_classifier(
    steps,
    batch_size,
    hidden_units,
    train_x,
    train_y,
    validate_x=None,
    validate_y=None,
    validate=False,
    dropout=0):
    '''Trains neural network regression model'''
    
    periods = 10
    steps_per_period = steps / periods
    
    # Create DNN classifier object
#     my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    my_optimizer = tf.train.AdamOptimizer(learning_rate=0.0007)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    classifier = tf.estimator.DNNClassifier(
        feature_columns = construct_feature_columns(train_x),
        hidden_units = hidden_units,
        optimizer = my_optimizer,
        dropout = dropout
    )
    
    # Create input functions
    train_input_fn = lambda: my_input_fn(train_x, train_y, batch_size)
    predict_train_input_fn = lambda: my_input_fn(train_x, train_y, num_epochs=1, shuffle=False)
    predict_validate_input_fn = lambda: my_input_fn(validate_x, validate_y, num_epochs=1, shuffle=False)
    
    # Train model in loop to periodically assess
    print("Training model...")
    print("LogLoss error (on train/validation data):")
    training_log_losses = []
    validation_log_losses = []
    for period in range (0, periods):
        # Train the model, starting from the prior state.
        classifier.train(
            input_fn=train_input_fn,
            steps=steps_per_period
        )

        # Take a break and compute probabilities.
        training_probabilities = classifier.predict(input_fn=predict_train_input_fn)
        training_probabilities = np.array([item['probabilities'] for item in training_probabilities])
        training_log_loss = metrics.log_loss(train_y, training_probabilities)
        training_log_losses.append(training_log_loss)
        print("  train period %02d : %0.3f" % (period, training_log_loss))
        
        if validate:
            validation_probabilities = classifier.predict(input_fn=predict_validate_input_fn)
            validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])
            validation_log_loss = metrics.log_loss(validate_y, validation_probabilities)
            validation_log_losses.append(validation_log_loss)
            print("  validation period %02d : %0.3f" % (period, validation_log_loss))
    print("Model training finished.")
    
    # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.tight_layout()
    plt.plot(training_log_losses, label="training")
    plt.plot(validation_log_losses, label="validation")
    plt.legend()

    return classifier
n = len(train.index)
classifier = train_dnn_classifier(1000, 20, [10, 10], preprocess_features(train.head(int(n*0.8))), preprocess_targets(train.head(int(n*0.8))), preprocess_features(train.tail(int(n*0.2))), preprocess_targets(train.tail(int(n*0.2))), validate=True, dropout=0.1)

def main_evaluation(hidden=[80, 80], drop=0.2):
    # Choose the first 12000 (out of 17000) examples for training.
    n = len(train.index)
    train_x = preprocess_features(train.head(int(n*0.8)))
    train_y = preprocess_targets(train.head(int(n*0.8)))

    # Choose the last 5000 (out of 17000) examples for validation.
    validate_x = preprocess_features(train.tail(int(n*0.2)))
    validate_y = preprocess_targets(train.tail(int(n*0.2)))

    # instantiate classifier
    classifier = train_dnn_classifier(
        steps=1300,
        batch_size=50,
        hidden_units=hidden,
        train_x=train_x,
        train_y=train_y,
        validate_x=validate_x,
        validate_y=validate_y,
        validate=True,
        dropout=drop)
    
    # evaluate model
    train_eval_metrics = classifier.evaluate(input_fn = lambda: my_input_fn(train_x, train_y, num_epochs=1, shuffle=False))
    validate_eval_metrics = classifier.evaluate(input_fn = lambda: my_input_fn(validate_x, validate_y, num_epochs=1, shuffle=False))
    print("AUC on the training set: %0.5f" % train_eval_metrics['auc'])
    print("Accuracy on the training set: %0.5f" % train_eval_metrics['accuracy'])
    print("AUC on the validation set: %0.2f" % validate_eval_metrics['auc'])
    print("Accuracy on the validation set: %0.2f" % validate_eval_metrics['accuracy'])
    
    # plot AUC
    validation_probabilities = classifier.predict(input_fn = lambda: my_input_fn(validate_x, validate_y, num_epochs=1, shuffle=False))
    # Get just the probabilities for the positive class.
    validation_probabilities = np.array([item['probabilities'][1] for item in validation_probabilities])
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
        validate_y, validation_probabilities)
    plt.figure()
    plt.plot(false_positive_rate, true_positive_rate, label="our model")
    plt.plot([0, 1], [0, 1], label="random classifier")
    _ = plt.legend(loc=2)
    
    return [hidden, drop, train_eval_metrics['auc'], train_eval_metrics['accuracy'], validate_eval_metrics['auc'], validate_eval_metrics['accuracy']]
# hidden = [80, 80]
# drop = .1
# configs = []
# for i in range(6):
#     for j in range(4):
#         x = main_evaluation(hidden, drop)
#         print(x)
#         configs += x
#         drop += 0.1
#     hidden[0] += 10
#     hidden[1] += 10
#     drop = 0.1
# def sort_config(configs):
#     new_config = []
#     for config in configs:
#         new_config += tuple(config)
#     return sorted(new_config, key = lambda config: config[3] + config[5])

# s = sort_config(configs)
# for config in s:
#     print(config)
def get_submission(classifier, test_x):
    '''Export predictions on test features to csv'''
#     predictions = classifier.predict(
#         input_fn = lambda: my_input_fn(test_x, num_epochs=1, shuffle=False)
#     )
    predictions = classifier.predict(
        input_fn = lambda: eval_input_fn(test_x)
    )
    # group survive prediction with ids in submission dataframe
    submission = pd.DataFrame()
    ids = test['PassengerId']
    for pred, p_id in zip(predictions, ids):
        cur = pd.Series()
        class_id = pred['class_ids'][0]
        cur['Survived'] = class_id
        cur['PassengerId'] = p_id
        submission = submission.append(cur, ignore_index=True)

    submission['PassengerId'] = submission['PassengerId'].astype(int)
    submission['Survived'] = submission['Survived'].astype(int)

    return submission

def main():
    train_x = preprocess_features(train)
    train_y = preprocess_targets(train)

    # instantiate classifier
    classifier = train_dnn_classifier(
        steps=1300,
        batch_size=50,
        hidden_units=[80, 80],
        train_x=train_x,
        train_y=train_y,
        dropout=0.20)
    
    test_x = preprocess_features(test)
    submission = get_submission(classifier, test_x)
    # download csv
    submission.to_csv('titanic_submission.csv', index=False)
main()