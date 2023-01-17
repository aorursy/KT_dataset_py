import tensorflow as tf
import numpy as np

def mean_score(y_true, y_pred):
    """
    Calculate mean score for batch images

    :param y_true: 4-D Tensor of ground truth, such as [NHWC]. Should have numeric or boolean type.
    :param y_pred: 4-D Tensor of prediction, such as [NHWC]. Should have numeric or boolean type.
    :return: 0-D Tensor of score
    """
    y_true_ = tf.cast(tf.round(y_true), tf.bool)
    y_pred_ = tf.cast(tf.round(y_pred), tf.bool)

    # Flatten
    y_true_ = tf.reshape(y_true_, shape=[tf.shape(y_true_)[0], -1])
    y_pred_ = tf.reshape(y_pred_, shape=[tf.shape(y_pred_)[0], -1])
    threasholds_iou = tf.constant(np.arange(0.5, 1.0, 0.05), dtype=tf.float32)

    def _mean_score(y):
        """Calculate score per image"""
        y0, y1 = y[0], y[1]
        total_cm = tf.confusion_matrix(y0, y1, num_classes=2)
        total_cm = tf.Print(total_cm, [total_cm])
        sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
        sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
        cm_diag = tf.to_float(tf.diag_part(total_cm))
        denominator = sum_over_row + sum_over_col - cm_diag
        denominator = tf.where(tf.greater(denominator, 0), denominator, tf.ones_like(denominator))
        # iou[0]: IoU of Background
        # iou[1]: IoU of Foreground
        iou = tf.div(cm_diag, denominator)
        iou_fg = iou[1]
        greater = tf.greater(iou_fg, threasholds_iou)
        score_per_image = tf.reduce_mean(tf.cast(greater, tf.float32))
        # Both predicted object and ground truth are empty, score is 1.
        score_per_image = tf.where(
            tf.logical_and(
                tf.equal(tf.reduce_any(y0), False), tf.equal(tf.reduce_any(y1), False)),
            1., score_per_image)
        return score_per_image

    elems = (y_true_, y_pred_)
    scores_per_image = tf.map_fn(_mean_score, elems, dtype=tf.float32)
    return tf.reduce_mean(scores_per_image)

y_true = tf.placeholder(dtype=tf.int32, shape=[2, 2])
y_pred = tf.placeholder(dtype=tf.int32, shape=[2, 2])
_y_true = tf.reshape(y_true, [1, 2, 2, 1])
_y_pred = tf.reshape(y_pred, [1, 2, 2, 1])

score = mean_score(_y_true, _y_pred)

sess = tf.Session()
y_true_val = np.array([[1, 1],
                       [0, 0]])
y_pred_val = np.array([[1, 1],
                       [1, 0]])

print("Score is {}".format(sess.run(score, feed_dict={y_true: y_true_val, y_pred: y_pred_val})))
y_true_val = np.array([[1, 1],
                       [0, 0]])
y_pred_val = np.array([[0, 0],
                       [0, 0]])

print("Score is {}".format(sess.run(score, feed_dict={y_true: y_true_val, y_pred: y_pred_val})))
y_true_val = np.array([[0, 0],
                       [0, 0]])
y_pred_val = np.array([[0, 0],
                       [0, 0]])

print("Score is {}".format(sess.run(score, feed_dict={y_true: y_true_val, y_pred: y_pred_val})))