import numpy as np

import keras.backend as K



def QWKloss(y_true, y_pred):

    N = K.sum(y_true)

    n_classes = 5



    # K.expand_dims(y_true, -1)  # N * K * 1

    # K.expand_dims(y_pred, 1)  # N * 1 * K

    C = K.batch_dot(K.expand_dims(y_true, -1), K.expand_dims(y_pred, 1))  # N * K * K

    C = K.sum(C, axis=0)

    

    sum0 = K.sum(C, axis=0)

    sum1 = K.sum(C, axis=1)

    E = K.dot(K.reshape(sum0, (n_classes,1)), K.reshape(sum1, (1,n_classes))) / N

    

    W = np.zeros([n_classes, n_classes], dtype=np.int)

    W += np.arange(n_classes)

    W = (W - W.T) ** 2

    

    k = K.sum(W*C) / K.sum(W*E)

    k = k

    return 1-k
# check ~ comparing sklearn

for _ in range(10):

    N = 1000

    y_true = np.eye(5)[np.random.randint(5, size=(N))]

    y_pred = np.random.uniform(size=(N, 5))

    y_pred /= y_pred.sum(axis=1, keepdims=True)



    from sklearn.metrics import cohen_kappa_score

    # sklearn can not treat probability vector.

    skl = cohen_kappa_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1), weights='quadratic')



    s = QWKloss(y_true, np.eye(5)[np.argmax(y_pred, axis=1)])

    org = K.get_value(s)

    

    print(skl, org)