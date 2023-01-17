import tensorflow as tf





def f2_score(y_true, y_pred):

    y_true = tf.cast(y_true, "int32")

    y_pred = tf.cast(tf.round(y_pred), "int32") # implicit 0.5 threshold via tf.round

    y_correct = y_true * y_pred

    sum_true = tf.reduce_sum(y_true, axis=1)

    sum_pred = tf.reduce_sum(y_pred, axis=1)

    sum_correct = tf.reduce_sum(y_correct, axis=1)

    precision = sum_correct / sum_pred

    recall = sum_correct / sum_true

    f_score = 5 * precision * recall / (4 * precision + recall)

    f_score = tf.where(tf.is_nan(f_score), tf.zeros_like(f_score), f_score)

    return tf.reduce_mean(f_score)





# Ex. usage

# model.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=[f2_score]) 