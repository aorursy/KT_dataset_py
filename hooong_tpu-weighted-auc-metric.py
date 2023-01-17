# good code to test my bad code 



from sklearn import metrics

import numpy as np



def alaska_weighted_auc(y_true, y_valid):

    tpr_thresholds = [0.0, 0.4, 1.0]

    weights =        [       2,   1]

    

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)

    

    # size of subsets

    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

    

    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

    normalization = np.dot(areas, weights)



    competition_metric = 0

    

    for idx, weight in enumerate(weights):

        y_min = tpr_thresholds[idx]

        y_max = tpr_thresholds[idx + 1]

        mask = (y_min < tpr) & (tpr < y_max)



        x_padding = np.linspace(fpr[mask][-1], 1, 100)



        x = np.concatenate([fpr[mask], x_padding])

        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])

        y = y - y_min # normalize such that curve starts at y=0

        score = metrics.auc(x, y)

        submetric = score * weight

        best_subscore = (y_max - y_min) * weight

        competition_metric += submetric

        

    return competition_metric / normalization
# bad code, but it works? 



import tensorflow as tf 

from tensorflow.python.ops import math_ops

import types

    

def fix_auc(binary = False):

    # only tested with default parameters for the AUC class

    # do not run this again in the same session 

    

    def update_state(self, y_true, y_pred, sample_weight=None):

        # for sparse categorical accuracy 

        

        y_true = tf.cast(y_true != 0, tf.int64)

        y_pred = 1 - y_pred[:, 0]

        return self._update_state(y_true, y_pred, sample_weight)



    def result(self):



        normalization = 1.4



        recall = math_ops.div_no_nan(self.true_positives,

                                     self.true_positives + self.false_negatives)



        fp_rate = math_ops.div_no_nan(self.false_positives,

                                    self.false_positives + self.true_negatives)

        x = fp_rate

        y = recall



        heights = (y[:self.num_thresholds - 1] + y[1:]) / 2.



        regular_auc = math_ops.reduce_sum(math_ops.multiply(x[:self.num_thresholds - 1] - x[1:], heights), name=self.name)



        under40_auc = math_ops.reduce_sum(math_ops.multiply(x[:self.num_thresholds - 1] - x[1:], tf.clip_by_value (heights, 0, 0.4)), name=self.name)



        return (regular_auc + under40_auc) / 1.4

    

    if not binary: 

        if not hasattr(tf.keras.metrics.AUC, '_update_state'):

            tf.keras.metrics.AUC._update_state = tf.keras.metrics.AUC.update_state

        tf.keras.metrics.AUC.update_state = update_state

    

    tf.keras.metrics.AUC.result = result

    

    return tf.keras.metrics.AUC



# check if it actually works

# differences may be due to thresholds or estimation method





tf.keras.metrics.AUC = fix_auc(binary = True)



for i in range(10):

    signal = np.random.random(1000)

    labels = (signal > 0.5).astype(int)

    preds = np.random.random(1000) + (signal - 0.5) * i * 0.1

    preds = np.clip(preds, 0, 1)

    

    

    auc = metrics.roc_auc_score(labels, preds)

    weighted_auc = alaska_weighted_auc(labels, preds)

    accuracy = (labels == (preds > 0.5)).mean()

    

    print(auc, weighted_auc, tf.keras.metrics.AUC()(labels, preds))







# usage. remember to initialize the metric class instance in the scope 



tf.keras.metrics.AUC = fix_auc(binary = False)



'''

with strategy.scope():

    model = your_model

    

    metrics = [tf.keras.metrics.AUC()]

    

    model.compile(

        optimizer='adam',

        loss = losses,

        metrics= metrics        

    )



'''