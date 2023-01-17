import numpy as np

from sklearn.metrics import r2_score



def r_score(y_true, y_pred, sample_weight=None, multioutput=None):

    r2 = r2_score(y_true, y_pred, sample_weight=sample_weight,

                  multioutput=multioutput)

    r = (np.sign(r2)*np.sqrt(np.abs(r2)))

    if r <= -1:

        return -1

    else:

        return r
import matplotlib.pyplot as plt

import seaborn as sns

count = 10000

v = np.random.normal(0, 1, count)



s = [(x, r_score(v, np.ones(count) * x + v)) for x in np.linspace(-2, 2, 1001)]

max(s, key = lambda x: x[1]), v.mean()



_ = plt.plot(s)
count = 10000

v = np.random.normal(0, 1.5, count)



s = [(x, r_score(v, np.ones(count) * x + v.mean())) for x in np.linspace(-2, 2, 1001)]

max(s, key = lambda x: x[1]), v.mean()



_ = plt.plot(s)