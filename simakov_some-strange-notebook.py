import numpy as np

import statsmodels.formula.api as smf

import statsmodels.tsa.api as smt

import statsmodels.api as sm

import scipy.stats as scs

import matplotlib.pyplot as plt

%pylab inline

import pandas as pd

from tqdm import tqdm_notebook

from itertools import product
train = np.loadtxt('../input/ts-data/sig1.train')
test = np.loadtxt('../input/ts-data/sig1.test.public')
X_train = train[:, 0]

ind_train = train[:, 1]

plt.hist(X_train[ind_train == 1], bins=20)

# plt.gca().set_xscale('log')
changepoints = np.where(np.diff(ind_train) != 0)[0] + 1
# логарифм отношения правдоподобий

def normal_likelihood(value, mean_0, mean_8, std):

    return np.log(scs.norm.pdf(value, mean_0, std) / 

                  scs.norm.pdf(value, mean_8, std))
# Базовый алгоритм для статистик

class Stat(object):

    def __init__(self, threshold, direction="unknown", init_stat=0.0):

        self._direction = str(direction)

        self._threshold = float(threshold)

        self._stat = float(init_stat)

        self._alarm = self._stat / self._threshold

    

    @property

    def direction(self):

        return self._direction



    @property

    def stat(self):

        return self._stat

        

    @property

    def alarm(self):

        return self._alarm

        

    @property

    def threshold(self):

        return self._threshold

    

    def update(self, **kwargs):

        # Statistics may use any of the following kwargs:

        #   ts - timestamp for the value

        #   value - original value

        #   mean - current estimated mean

        #   std - current estimated std

        #   adjusted_value - usually (value - mean) / std

        # Statistics call this after updating '_stat'

        self._alarm = self._stat / self._threshold
class MeanExpNoDataException(Exception):

    pass



# Класс, реализующий вычисление скользящего среднего

class MeanExp(object):

    def __init__(self, new_value_weight, load_function=median):

        self._load_function = load_function

        self._new_value_weight = new_value_weight

        self.load([])



    @property

    def value(self):

        if self._weights_sum <= 1:

            raise MeanExpNoDataException('self._weights_sum <= 1')

        return self._values_sum / self._weights_sum



    def update(self, new_value, **kwargs):

        self._values_sum = (1 - self._new_value_weight) * self._values_sum + self._new_value_weight * new_value

        self._weights_sum = (1 - self._new_value_weight) * self._weights_sum + 1.0



    def load(self, old_values):

        if old_values:

            old_values = [value for ts, value in old_values]

            mean = float(self._load_function(old_values))

            self._weights_sum = min(float(len(old_values)), 1.0 / self._new_value_weight)

            self._values_sum = mean * self._weights_sum

        else:

            self._values_sum = 0.0

            self._weights_sum = 0.0
class AdjustedCusum(Stat):

    def __init__(self, mean_diff,

                 threshold, direction="unknown", init_stat=0.0):

        self.mean_diff = mean_diff

        super(AdjustedCusum, self).__init__(threshold, direction, init_stat)

        

    def update(self, value):

        zeta_k = normal_likelihood(value, mean_diff, 0., 1.)

        self._stat = max(0, self._stat + zeta_k)

        super(AdjustedCusum, self).update()
X_train_fl = X_train.copy()
def clip_by_abs(X, threshold=5):

    for i, x in enumerate(X):

        if abs(x) > threshold:

            X[i] = np.mean([X[i - 1], X[i + 1]])

    X_max = max(abs(X))

    if abs(X_max) > threshold:

        return clip_by_abs(X, threshold)

    else:

        return X
X_train_fl = clip_by_abs(X_train_fl, threshold=8)
mean = 0.

var = 1.

alpha = 0.05

beta = 0.05

mean_diff = 1.0



X = X_train_fl



stat_trajectory, mean_values, var_values = [], [], []



mean_exp = MeanExp(new_value_weight=alpha)

var_exp = MeanExp(new_value_weight=beta)

cusum = AdjustedCusum(mean_diff, 30.)

for k, x_k in enumerate(X):

    try:

        mean_estimate = mean_exp.value

    except MeanExpNoDataException:

        mean_estimate = 0.

    

    try:

        var_estimate = var_exp.value

    except MeanExpNoDataException:

        var_estimate = 1.

    

    adjusted_value = (x_k - mean_estimate) / np.sqrt(var_estimate)

    cusum.update(adjusted_value)

    

    mean_exp.update(x_k)

    diff_value = (x_k - mean) ** 2

    var_exp.update(diff_value)

    

    stat_trajectory.append(cusum._stat)

    mean_values.append(mean_estimate)

    var_values.append(np.sqrt(var_estimate))



plot(stat_trajectory)

grid()

title('Траектория статистики для обучающей выборки\nВертикальными линиями показаны точки разладки')

axvline(x=changepoints[0], color='r')

axvline(x=changepoints[1], color='k')



figure()

plot(X)

plot(ind_train * 8)



plot(np.array(mean_values), 'k')

plot(np.array(mean_values) + np.sqrt(var_values), 'k')

plot(np.array(mean_values) - np.sqrt(var_values), 'k')

axvline(x=changepoints[0], color='r')

axvline(x=changepoints[1], color='k')

grid()
test_fl = test.copy()

test_fl = clip_by_abs(test_fl, threshold=8)




X = test_fl



stat_trajectory, mean_values, var_values = [], [], []



mean_exp = MeanExp(new_value_weight=alpha)

var_exp = MeanExp(new_value_weight=beta)

cusum = AdjustedCusum(mean_diff, 30.)

for k, x_k in enumerate(X):

    try:

        mean_estimate = mean_exp.value

    except MeanExpNoDataException:

        mean_estimate = 0.

    

    try:

        var_estimate = var_exp.value

    except MeanExpNoDataException:

        var_estimate = 1.

    

    adjusted_value = (x_k - mean_estimate) / np.sqrt(var_estimate)

    cusum.update(adjusted_value)

    

    mean_exp.update(x_k)

    diff_value = (x_k - mean) ** 2

    var_exp.update(diff_value)

    

    stat_trajectory.append(cusum._stat)

    mean_values.append(mean_estimate)

    var_values.append(np.sqrt(var_estimate))



plot(stat_trajectory)

grid()

title('Траектория статистики для тестовой выборки')





figure()

plot(X)



grid()