import numpy as np
import pandas as pd
import tensorflow as tf

# make TensorFlow less verbose
tf.logging.set_verbosity(tf.logging.ERROR)

# read the dataset
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")

# drop unused columns
UNUSED_COLUMNS = ["Name", "Ticket", "Age", "Cabin", "Embarked", "Fare"]
train_data = train_data.drop(UNUSED_COLUMNS, axis=1)
test_data = test_data.drop(UNUSED_COLUMNS, axis=1)
# sample 80% for train data
train_set = train_data.sample(frac=0.8, replace=False, random_state=42)
# the other 20% is reserved for cross validation
cv_set = train_data.loc[ set(train_data.index) - set(train_set.index)]

# define features
sex_feature = tf.feature_column.categorical_column_with_vocabulary_list(
    'Sex', ['female','male']
)

feature_columns = [ sex_feature ]

estimator = tf.estimator.LinearClassifier(
    feature_columns=feature_columns)

# train input function
train_input_fn = tf.estimator.inputs.pandas_input_fn(
      x=train_set.drop('Survived', axis=1),
      y=train_set.Survived,
      num_epochs=None, # for training, use as many epochs as necessary
      shuffle=True,
      target_column='target',
)

cv_input_fn = tf.estimator.inputs.pandas_input_fn(
      x=cv_set.drop('Survived', axis=1),
      y=cv_set.Survived,
      num_epochs=1, # only to score
      shuffle=False
)

estimator.train(input_fn=train_input_fn, steps=10)

scores = estimator.evaluate(input_fn=cv_input_fn)
print("\nTest Accuracy: {0:f}\n".format(scores['accuracy']))
test_input_fn = tf.estimator.inputs.pandas_input_fn(
      x=test_data,
      num_epochs=1, # only to predict
      shuffle=False 
)

predictions = list(estimator.predict(input_fn=test_input_fn))
predicted_classes = [prediction['class_ids'][0] for prediction in predictions]
evaluation = test_data['PassengerId'].copy().to_frame()
evaluation["Survived"] = predicted_classes
evaluation.to_csv("evaluation_submission.csv", index=False) # Public Score: 0.76555
evaluation.head()
