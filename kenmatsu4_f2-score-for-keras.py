import numpy as np



import keras

from keras import backend as K

from keras import metrics



from sklearn.metrics import fbeta_score
# ref: https://github.com/keras-team/keras/blob/ac1a09c787b3968b277e577a3709cd3b6c931aa5/tests/keras/test_metrics.py

def f2_score(y_true, y_pred):

    beta = 2

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=1)

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=1)

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=1)

    

    precision = true_positives / (predicted_positives + K.epsilon())

    recall = true_positives / (possible_positives + K.epsilon())

    

    return K.mean(((1+beta**2)*precision*recall) / ((beta**2)*precision+recall+K.epsilon()))
# Test Data

y_true_np = np.array([[0, 1, 1, 1, 0],

                      [0, 1, 1, 1, 0],

                      [0, 0, 0, 1, 1],

                      [1, 0, 1, 0, 0],

                      [1, 1, 0, 0, 0],

                      [1, 1, 0, 0, 0],

                      [1, 1, 0, 0, 0]])



y_pred_np = np.array([[0.6, 0.6, 0.6, 0.1, 0.1],

                      [0.1, 0.6, 0.6, 0.6, 0.1],

                      [0.1, 0.1, 0.6, 0.1, 0.6],

                      [0.6, 0.1, 0.1, 0.1, 0.6],

                      [0.6, 0.6, 0.6, 0.1, 0.1],

                      [0.6, 0.6, 0.6, 0.6, 0.6],

                      [0.1, 0.1, 0.1, 0.1, 0.1]])



y_true = K.variable(y_true_np)

y_pred = K.variable(y_pred_np)
K.eval(f2_score(y_true, y_pred),)
scores = []

for yt, yp in zip(y_true_np, y_pred_np):

    scores.append(fbeta_score(yt, (yp>0.5).astype(int), beta=2))

print(f"mean scores: {np.mean(scores)}")

scores