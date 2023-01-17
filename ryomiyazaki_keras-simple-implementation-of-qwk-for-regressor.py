import keras.backend as K

# For scoring coding on continuous manner.

def QWKloss_score(y_true, y_pred):

    N = K.cast(K.shape(y_true)[0], 'float32')

    

    WC = (y_pred - y_true)**2 / N

    WE = (y_pred - K.transpose(y_true))**2 / (N**2)

    

    k = K.sum(WC) / K.sum(WE)

    

    return 1-k
import numpy as np

from sklearn.metrics import cohen_kappa_score



# check ~ comparing sklearn

for _ in range(10):

    N = 1000

    y_true = np.random.randint(5, size=(N))

    y_pred = np.random.randint(5, size=(N))  # cast to int is necessary for sklearn. but youcan use float in Keras use.



    skl = cohen_kappa_score(y_true, y_pred, weights='quadratic')



    s = QWKloss_score(K.variable(y_true.reshape(-1, 1)), K.variable(y_pred.reshape(-1, 1)))

    org = K.get_value(s)

    

    print(skl, org)