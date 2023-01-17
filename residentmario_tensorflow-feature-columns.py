import tensorflow as tf
numeric_feature_column = tf.feature_column.numeric_column(key="SepalLength", dtype=tf.float32)
matrix_feature_column = tf.feature_column.numeric_column(key="MyMatrix", shape=[10,5])
bucketized_feature_column = tf.feature_column.bucketized_column(
    source_column = numeric_feature_column,
    boundaries = [1960, 1980, 2000])
tf.feature_column.categorical_column_with_identity
tf.feature_column.categorical_column_with_vocabulary_file
# Cross the bucketized columns, using 5000 hash bins.
crossed_lat_lon_fc = tf.feature_column.crossed_column([bucketized_feature_column, bucketized_feature_column], 5000)