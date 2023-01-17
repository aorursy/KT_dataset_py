import tensorflow as tf

import numpy as np

import pandas as pd
train = pd.read_csv('../input/digit-recognizer/train.csv') 

test = pd.read_csv('../input/digit-recognizer/test.csv')

train.info()
#Decrease arrays sizes

X = train.drop('label', axis=1).astype(np.int32)

y = train['label'].astype(np.int32)
#define tensors

feat_cols = [tf.feature_column.numeric_column("x", shape=[28, 28])]
classifier = tf.estimator.DNNClassifier(

    feature_columns=feat_cols,

    hidden_units=[256,32],

    optimizer=tf.train.AdamOptimizer(1e-4),

    n_classes=10,

    dropout=0.1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
train_input_fn = tf.estimator.inputs.numpy_input_fn(

    x={"x": X_train.values},

    y=y_train.values,

    num_epochs=None,

    batch_size=50,

    shuffle=True)
classifier.train(input_fn=train_input_fn, steps=100000)
test_input_fn = tf.estimator.inputs.numpy_input_fn(

    x={"x": X_test.values},

    num_epochs=1,

    shuffle=False)
note_predictions = list(classifier.predict(input_fn=test_input_fn))
#Check, which parameter to extract from predictions

note_predictions[0]
preds  = []

for pred in note_predictions:

    preds.append(pred['class_ids'][0])
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,preds))
fin_test_input_fn = tf.estimator.inputs.numpy_input_fn(

    x={"x": test.values},

    num_epochs=1,

    shuffle=False)

note_fin_predictions = list(classifier.predict(input_fn=fin_test_input_fn))
final_preds  = []

for pred in note_fin_predictions:

    final_preds.append(pred['class_ids'][0])
submission = pd.DataFrame({'ImageId':range(1, len(final_preds)+1), 'Label':final_preds})

submission.to_csv('submission.csv', index = False)