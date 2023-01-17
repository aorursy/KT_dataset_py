import numpy as np

from sklearn.metrics import cohen_kappa_score



# import tensorflow.keras.backend as K

import tensorflow as tf
def kappa_keras(y_true, y_pred):



    y_true = tf.cast(tf.math.argmax(y_true, axis=-1), dtype='int32')

    y_pred = tf.cast(tf.math.argmax(y_pred, axis=-1), dtype='int32')



    # Figure out normalized expected values

    min_rating = tf.minimum(tf.math.reduce_min(y_true), tf.math.reduce_min(y_pred))

    max_rating = tf.maximum(tf.math.reduce_max(y_true), tf.math.reduce_max(y_pred))



    # shift the values so that the lowest value is 0

    # (to support scales that include negative values)

    y_true = tf.map_fn(lambda y: y - min_rating, y_true, dtype='int32')

    y_pred = tf.map_fn(lambda y: y - min_rating, y_pred, dtype='int32')



    # Build the observed/confusion matrix

    num_ratings = max_rating - min_rating + 1

    observed = tf.math.confusion_matrix(y_true, y_pred,

                                        num_classes=num_ratings)

    num_scored_items = y_true.shape[0]



    weights = tf.expand_dims(tf.range(num_ratings), axis=-1) - tf.expand_dims(tf.range(num_ratings), axis=0)

    weights = tf.cast(tf.math.pow(weights, 2), dtype=tf.float32)



    hist_true = tf.math.bincount(y_true, minlength=num_ratings)

    hist_true = hist_true[:num_ratings] / num_scored_items

    hist_pred = tf.math.bincount(y_pred, minlength=num_ratings)

    hist_pred = hist_pred[:num_ratings] / num_scored_items

    expected = tf.cast(tf.tensordot(tf.expand_dims(hist_true, axis=-1), tf.expand_dims(hist_pred, axis=0), axes=1),

                       dtype=tf.float32)



    # Normalize observed array

    observed = tf.cast(observed / num_scored_items, dtype=tf.float32)



    # If all weights are zero, that means no disagreements matter.

    score = tf.where(tf.math.reduce_any(tf.math.not_equal(weights, 0)),

                     tf.math.reduce_sum(weights * observed) / tf.math.reduce_sum(weights * expected),

                     0)



    return 1. - score
def test_kappa_keras():



    LABELS = [0, 1, 2]

    N = 10

    TRIALS = 100



    for _ in range(TRIALS):

        y_true = np.random.choice(LABELS, size=N, replace=True)

        y_pred = np.random.choice(LABELS, size=N, replace=True)



        # Calculating QWK score with scikit-learn

        skl_score = cohen_kappa_score(y_true, y_pred, weights='quadratic')



        # Keras implementation of QWK work with one hot encoding labels and predictions (also it works with softmax probabilities)

        # Converting arrays to one hot encoded representation

        shape = (y_true.shape[0], np.maximum(y_true.max(), y_pred.max()) + 1)



        y_true_ohe = np.zeros(shape)

        y_true_ohe[np.arange(shape[0]), y_true] = 1



        y_pred_ohe = np.zeros(shape)

        y_pred_ohe[np.arange(shape[0]), y_pred] = 1



        keras_score = kappa_keras(y_true_ohe, y_pred_ohe).numpy()



        if not np.isclose(skl_score, keras_score):

            print(skl_score, keras_score)
# some errors due to casting to float32

test_kappa_keras()