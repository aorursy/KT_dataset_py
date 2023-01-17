import numpy as np

import pandas as pd

import tensorflow as tf
data = pd.read_csv('../input/mushrooms.csv')
data.head()
data['class'].unique()
data['class'] = data['class'].apply(lambda label: int(label == 'p'))
data = data.drop('veil-type', axis=1)
data.head()
from sklearn.model_selection import train_test_split
x_data = data.drop('class', axis=1)

y_data = data['class']

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)
data.columns
cap_shape = tf.feature_column.categorical_column_with_hash_bucket('cap-shape', hash_bucket_size=6)

cap_surface = tf.feature_column.categorical_column_with_hash_bucket('cap-surface', hash_bucket_size=4)

cap_color = tf.feature_column.categorical_column_with_hash_bucket('cap-color', hash_bucket_size=10)

bruises = tf.feature_column.categorical_column_with_hash_bucket('bruises', hash_bucket_size=2)

odor = tf.feature_column.categorical_column_with_hash_bucket('odor', hash_bucket_size=9)

gill_attachment = tf.feature_column.categorical_column_with_hash_bucket('gill-attachment', hash_bucket_size=2)

gill_spacing = tf.feature_column.categorical_column_with_hash_bucket('gill-spacing', hash_bucket_size=2)

gill_size = tf.feature_column.categorical_column_with_hash_bucket('gill-size', hash_bucket_size=2)

gill_color = tf.feature_column.categorical_column_with_hash_bucket('gill-color', hash_bucket_size=12)

stalk_shape = tf.feature_column.categorical_column_with_hash_bucket('stalk-shape', hash_bucket_size=2)

stalk_root = tf.feature_column.categorical_column_with_hash_bucket('stalk-root', hash_bucket_size=5)

stalk_surface_above_ring = tf.feature_column.categorical_column_with_hash_bucket('stalk-surface-above-ring', hash_bucket_size=4)

stalk_surface_below_ring = tf.feature_column.categorical_column_with_hash_bucket('stalk-surface-below-ring', hash_bucket_size=4)

stalk_color_above_ring = tf.feature_column.categorical_column_with_hash_bucket('stalk-color-above-ring', hash_bucket_size=9)

stalk_color_below_ring = tf.feature_column.categorical_column_with_hash_bucket('stalk-color-below-ring', hash_bucket_size=9)

veil_color = tf.feature_column.categorical_column_with_hash_bucket('veil-color', hash_bucket_size=4)

ring_number = tf.feature_column.categorical_column_with_hash_bucket('ring-number', hash_bucket_size=3)

ring_type = tf.feature_column.categorical_column_with_hash_bucket('ring-type', hash_bucket_size=5)

spore_print_color = tf.feature_column.categorical_column_with_hash_bucket('spore-print-color', hash_bucket_size=9)

population = tf.feature_column.categorical_column_with_hash_bucket('population', hash_bucket_size=6)

habitat = tf.feature_column.categorical_column_with_hash_bucket('habitat', hash_bucket_size=7)
feat_cols = [cap_shape, cap_surface, cap_color, bruises, odor, gill_attachment, gill_spacing, 

             gill_size, gill_color, stalk_shape, stalk_root, stalk_surface_above_ring, 

             stalk_surface_below_ring, stalk_color_above_ring, stalk_color_below_ring, 

             veil_color, ring_number, ring_type, spore_print_color, population, habitat]

sizes = [6, 4, 10, 2, 9, 2, 2, 2, 12, 2, 5, 4, 4, 9, 9, 4, 3, 5, 9, 6, 7]
embedded_feat_cols = []

i = 0

for label in feat_cols:

    label = tf.feature_column.embedding_column(label, dimension=sizes[i])

    i += 1

    embedded_feat_cols.append(label)
embedded_feat_cols
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=100, num_epochs=None, shuffle=True)
dnn_model = tf.estimator.DNNClassifier(hidden_units=[5, 5], feature_columns=embedded_feat_cols, n_classes=2)
dnn_model.train(input_fn=input_func, steps=1000)
pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=len(X_test), shuffle=False)
predictions = list(dnn_model.predict(input_fn=pred_input_func))
predictions[0]
y_pred = []



for pred in predictions:

    y_pred.append(pred['class_ids'][0])
y_pred[:10]
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
acc = tf.metrics.accuracy(y_test, y_pred)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

tf.local_variables_initializer().run()
sess.run([acc])