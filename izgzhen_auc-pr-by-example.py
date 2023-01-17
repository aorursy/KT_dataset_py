import sklearn.metrics as skmetrics

import tensorflow.keras.metrics as tf2metrics

import tensorflow.compat.v1 as tf1

import numpy as np

import pandas as pd
def sk_auc_pr(y_true, y_prob):

  precisions, recalls, thresholds = skmetrics.precision_recall_curve(y_true, y_prob)

  return skmetrics.auc(recalls, precisions)
def tf1_auc_pr_careful(y_true, y_prob):

  with tf1.Session() as sess:

    metric_val, update_op = tf1.metrics.auc(y_true, y_prob, curve="PR",

                                            summation_method="careful_interpolation")

    sess.run(tf1.local_variables_initializer())

    sess.run(update_op)

    return sess.run(metric_val)



def tf1_auc_pr_trapezoidal(y_true, y_prob):

  with tf1.Session() as sess:

    metric_val, update_op = tf1.metrics.auc(y_true, y_prob, curve="PR")

    sess.run(tf1.local_variables_initializer())

    sess.run(update_op)

    return sess.run(metric_val)
def tf2_auc_pr(y_true, y_prob):

  m = tf2metrics.AUC(curve="PR")

  m.update_state(y_true, y_prob)

  return m.result().numpy()
y_true = np.array([0, 0, 1, 1])

y_prob = np.array([0.1, 0.4, 0.35, 0.8])
df_stats = pd.DataFrame()



for f in [sk_auc_pr, tf1_auc_pr_careful, tf1_auc_pr_trapezoidal, tf2_auc_pr]:

  df_stats = df_stats.append({

      "Method": f.__name__,

      "AUC PR": f(y_true, y_prob)

  }, ignore_index=True)



df_stats[["Method", "AUC PR"]]