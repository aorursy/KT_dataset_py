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
pd.options.display.max_rows = 50
pd.options.display.float_format = '{:.3f}'.format

File = pd.read_csv('../input/model.csv')
File['POP_2009'] = File['POP_2009']/1000
File['Per_Cap_2009'] = File['Per_Cap_2009']/1000
File = File.fillna(0.0)
File.head()
File = File.reindex(
    np.random.permutation(File.index))
File.head()
def preprocess_features(File):
    selected_features = np.log1p(File[
    ['GROC09','GROCPTH09','SUPERC09','SUPERCPTH09','CONVS09','CONVSPTH09','SPECS09','SPECSPTH09','SNAPS12','SNAPSPTH12','LACCESS_POP10','LACCESS_LOWI10','LACCESS_HHNV10','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS08','RECFAC09','MEDHHINC15','METRO13','POP_2009','Per_Cap_2009','Personal_income_2009','pop_per_store','CONVS09_with_Snap_benefits','total_stores_snap_benefits']])
    processed_features = selected_features.copy()
    return processed_features

def preprocess_targets(File):
    output_targets = pd.DataFrame()
    output_targets["is_store_decline"] =File["is_store_decline"]
    return output_targets
# Choose the first 12000 (out of 17000) examples for training.
training_examples = preprocess_features(File.head(2200))
training_targets = preprocess_targets(File.head(2200))

# Choose the last 5000 (out of 17000) examples for validation.
validation_examples = preprocess_features(File.tail(943))
validation_targets = preprocess_targets(File.tail(943))

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
#File.isna().sum()
def train_linear_classifier_model(
    learning_rate,
    steps,
    batch_size,
    hidden_units,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
    periods = 10
    steps_per_period = steps / periods
    # Create a linear classifier object.
    my_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_classifier = tf.estimator.DNNClassifier(
      feature_columns=construct_feature_columns(training_examples),
      hidden_units=hidden_units,
      optimizer=my_optimizer)
    # Create input functions.
    training_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets["is_store_decline"], 
                                          batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets["is_store_decline"], 
                                                  num_epochs=1, 
                                                  shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                    validation_targets["is_store_decline"], 
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
        steps=steps_per_period)
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
    learning_rate=0.001,
    steps=450,
    batch_size=250,
    hidden_units=[40,60,20,20],
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)
predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                    validation_targets["is_store_decline"], 
                                                    num_epochs=1, 
                                                    shuffle=False)

evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)

print("AUC on the validation set: %0.2f" % evaluation_metrics['auc'])
print("Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy'])
validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
# Get just the probabilities for the positive class.
validation_probabilities = np.array([item['probabilities'][1] for item in validation_probabilities])

false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
    validation_targets, validation_probabilities)
#plt.plot(false_positive_rate, true_positive_rate, label="our model")
#plt.plot([0, 1], [0, 1], label="random classifier")
#_ = plt.legend(loc=2)

plt.style.use('seaborn-white')
plt.figure(figsize=[11,9])
plt.plot(false_positive_rate, true_positive_rate, label='ROC curve (area = %0.2f)' % evaluation_metrics['auc'], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('ROC Curve', fontsize=18)
plt.legend(loc="lower right", fontsize=18);
plt.savefig('rocauc.png', dpi=300, bbox_inches='tight')

#prepare features of 2014 to predict 
selected_features = np.log1p(File[
   ['GROC09','GROCPTH09','SUPERC09','SUPERCPTH09','CONVS09','CONVSPTH09','SPECS09','SPECSPTH09','SNAPS12','SNAPSPTH12','LACCESS_POP10','LACCESS_LOWI10','LACCESS_HHNV10','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS08','RECFAC09','MEDHHINC15','METRO13','POP_2009','Per_Cap_2009','Personal_income_2009','pop_per_store','CONVS09_with_Snap_benefits','total_stores_snap_benefits']])
test_df=pd.read_csv('../input/test.csv')
test_df = test_df.rename(columns={'GROC14': 'GROC09', 'GROCPTH14': 'GROCPTH09','SUPERC14': 'SUPERC09','SUPERCPTH14': 'SUPERCPTH09','CONVS14': 'CONVS09','CONVSPTH14': 'CONVSPTH09',
                                 'SPECS14':'SPECS09','SPECSPTH14':'SPECSPTH09','SNAPS16':'SNAPS12','SNAPSPTH16':'SNAPSPTH12','LACCESS_POP15':'LACCESS_POP10',
                                 'LACCESS_LOWI15':'LACCESS_LOWI10','LACCESS_HHNV15':'LACCESS_HHNV10','PCT_DIABETES_ADULTS13':'PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS13':'PCT_OBESE_ADULTS08',
                                 'RECFAC14':'RECFAC09','Pop_2014':'POP_2009','Per_Cap_2014':'Per_Cap_2009','Personal_income_2014':'Personal_income_2009','CONVS14_with_Snap_benefits':'CONVS09_with_Snap_benefits'})
test_df['is_store_decline']=0
#Preprocess data 
test_df['POP_2009'] = test_df['POP_2009']/1000
test_df['Per_Cap_2009'] = test_df['Per_Cap_2009']/1000
test_input = preprocess_features(test_df)
test_targets = preprocess_targets(test_df)
predict_test_input_fn = lambda: my_input_fn(test_input, 
                                                    test_targets["is_store_decline"], 
                                                    num_epochs=1, 
                                                    shuffle=False)
test_predictions = linear_classifier.predict(input_fn=predict_test_input_fn)
test_predictions = np.array([item['probabilities'][1] for item in test_predictions])
test_predictions
test_df.head()
# Make sure the all zip codes are 5 digits long (some of them start with 00)
test_df['County FIPS Code'] = test_df['FIPS'].apply(lambda x: str(x).zfill(5))
# Confirm that worked okay.
test_df[['County FIPS Code', 'FIPS']].head(3)
# create a new dataset with only the columns we need
submission=pd.DataFrame(list(zip(test_df['County FIPS Code'], test_predictions)), columns=['FIPS','Probability'])
# convert probabilities to percentages
submission['Probability'] = submission['Probability'].apply(lambda x: round(x*100, 1))
submission.head()
submission.to_csv('predict_proba.csv')
