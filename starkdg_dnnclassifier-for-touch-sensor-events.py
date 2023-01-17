# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import ast
from sklearn import metrics
from matplotlib import pyplot as plt
from pandas.api.types import CategoricalDtype
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def process_features(dataframe):
    '''process pandas dataframe: turn main 'event' column
    into numpy array and split into 800 separate columns
    '''
    df = pd.DataFrame()
    df['event'] = dataframe['event'].apply(lambda x: np.array(ast.literal_eval(x)).flatten())

    new_columns = ["Input" + str(x) for x in range(800)]

    df2 = pd.DataFrame()
    df2[new_columns] = pd.DataFrame(df['event'].values.tolist(), columns=new_columns)
    return df2

def process_targets(frame):
    '''extract target class from pandas dataframe as integer value
    '''
    cat_type = CategoricalDtype(categories=['b', 'k', 'r'], ordered=True)
    targets = pd.DataFrame()
    targets['class'] = frame['class'].astype(cat_type).cat.codes.astype(int)

    return targets

def split_dataframe(df):
    ''' split data frame into training, validation and test sets 
    '''
    src_df = df.reindex(np.random.permutation(df.index))
    training_df = src_df.iloc[:50, :]
    validation_df = src_df.iloc[50:90, :]
    testing_df = src_df.iloc[90:, :]

    return training_df, validation_df, testing_df

def create_training_input_fn(
    
    features, labels, batch_size, num_epochs=None, shuffle=True):

    def _input_fn():
        raw_features = {"sensordata": features.values}
        raw_labels = np.array(labels)
        ds = tf.data.Dataset.from_tensor_slices((raw_features, raw_labels))
        ds = ds.batch(batch_size).repeat(num_epochs)
        if shuffle:
            ds = ds.shuffle(100)
        feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
        return feature_batch, label_batch

    return _input_fn


def create_predict_input_fn(features, labels, batch_size):

    def _input_fn():
        raw_features = {"sensordata": features.values}
        raw_labels = np.array(labels)

        ds = tf.data.Dataset.from_tensor_slices((raw_features, raw_labels))
        ds = ds. batch(batch_size)
        feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
        return feature_batch, label_batch

    return _input_fn

def construct_feature_columns():
    ''' one feature column for 800 columns of data '''
    return set([tf.feature_column.numeric_column('sensordata', shape=800)])


def train_model(learning_rate,
                steps,
                batch_size,
                hidden_units,
                training_examples,
                training_targets,
                validation_examples,
                validation_targets):

    periods = 10
    steps_per_period = steps / periods

    training_input_fn = create_training_input_fn(
        training_examples, training_targets, batch_size)
    predict_training_input_fn = create_predict_input_fn(
        training_examples, training_targets, batch_size)
    predict_validation_input_fn = create_predict_input_fn(
        validation_examples, validation_targets, batch_size)

    my_optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(
        my_optimizer, 5.0)
    dnn_classifier = tf.estimator.DNNClassifier(
        feature_columns=construct_feature_columns(),
        n_classes=3,
        hidden_units=hidden_units,
        optimizer=my_optimizer)
    
    training_losses = []
    validation_losses = []
    print("Train Model")
    for period in range(0, periods):
        dnn_classifier.train(
            input_fn=training_input_fn, steps=steps_per_period)

        training_predictions = dnn_classifier.predict(
            input_fn=predict_training_input_fn)
        training_pred_class_id = np.array(
            [item['class_ids'][0] for item in training_predictions])
        training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id, 3)
        
        validation_predictions = dnn_classifier.predict(
            input_fn=predict_validation_input_fn)
        validation_pred_class_id = np.array(
            [item['class_ids'][0] for item in validation_predictions])
        validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id, 3)

        training_loss = metrics.log_loss(training_targets, training_pred_one_hot)
        validation_loss = metrics.log_loss(validation_targets, validation_pred_one_hot)
        print("    {0} : {1}".format(period, validation_loss))
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)

    final_predictions = dnn_classifier.predict(
        input_fn=predict_validation_input_fn)
    final_predictions = np.array(
        [item['class_ids'][0] for item in final_predictions])
    accuracy = metrics.accuracy_score(validation_targets, final_predictions)
    print("Final accuracy score (on validation data) ", accuracy)

    plt.ylabel("Loss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.plot(training_losses, label="training")
    plt.plot(validation_losses, label="validation")
    plt.legend()
   
    return dnn_classifier

def evaluate_classifier(classifier, examples, targets):
    eval_input_fn = create_predict_input_fn(examples, targets, 1)
    results = classifier.evaluate(eval_input_fn, steps=100, )
    print("Evaluate classifier:")
    for key in results:
        print("  {0} : {1}".format(key, results[key]))

def run_tests(classifier, examples, targets):
    predict_test_input_fn = create_predict_input_fn(examples, targets, 1)
    final_predictions = classifier.predict(input_fn=predict_test_input_fn)
    predicted_classes = np.array([item['class_ids'][0] for item in final_predictions])

    cm = metrics.confusion_matrix(targets.values, predicted_classes)
    cm

    accuracy = metrics.accuracy_score(targets.values, predicted_classes)
    accuracy

    target_names = ['b', 'k', 'r']
    metrics.classification_report(targets.values, predicted_classes, target_names=target_names)

    return accuracy

src_df = pd.read_csv("../input/touch_events.csv", sep=",")

# breakdown of event types
src_df.groupby(['class']).count()

#split into data sets
traindf, valid_df, test_df = split_dataframe(src_df)

training_examples = process_features(traindf)
training_targets = process_targets(traindf)

validation_examples = process_features(valid_df)
validation_targets = process_targets(valid_df)

test_examples = process_features(test_df)
test_targets = process_targets(test_df)
learning_rate = 0.0001
steps = 200
batch_size = 5
hidden_units = [500, 200]
classifier = train_model(learning_rate,
                         steps,
                         batch_size,
                         hidden_units,
                         training_examples,
                         training_targets,
                         validation_examples,
                         validation_targets)

evaluate_classifier(classifier, validation_examples, validation_targets)

run_tests(classifier, validation_examples, validation_targets)
run_tests(classifier, test_examples, test_targets)