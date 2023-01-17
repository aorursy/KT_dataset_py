import numpy as np

import pymc3 as pm

from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

%matplotlib inline
size = 500

true_intercept = 1

true_slope = 2

x = np.linspace(0, 1, size)

# y = a + b*x

true_regression_line = true_intercept + true_slope * x

# add noise

model1 = true_regression_line + np.random.normal(scale=.5, size=size) #Noisy

model2 = true_regression_line + np.random.normal(scale=.2, size=size) #Less Noisy
np.random.seed = 0

permutation_set = np.random.permutation(size)

train_set = permutation_set[0:size//2]

test_set = permutation_set[size//2:size]
print(mean_absolute_error(true_regression_line[test_set],model1[test_set]))

print(mean_absolute_error(true_regression_line[test_set],model2[test_set]))
print(mean_absolute_error(true_regression_line[test_set],(model1*.5+model2*.5)[test_set]))
data = dict(x1=model1[train_set], x2=model2[train_set], y=true_regression_line[train_set])

with pm.Model() as model:

    # specify glm and pass in data. The resulting linear model, its likelihood and 

    # and all its parameters are automatically added to our model.

    pm.glm.glm('y ~ x1 + x2', data)

    step = pm.NUTS() # Instantiate MCMC sampling algorithm

    trace = pm.sample(2000, step, progressbar=False)
pm.traceplot(trace, figsize=(7,7))

plt.tight_layout();
intercept = np.median(trace.Intercept)

print(intercept)

x1param = np.median(trace.x1)

print(x1param)

x2param = np.median(trace.x2)

print(x2param)
model1_train = model1[train_set]

model2_train = model2[train_set]

x_train = np.vstack((model1_train, model2_train)).T



model1_test = model1[test_set].T

model2_test = model2[test_set].T

x_test = np.vstack((model1_test, model2_test)).T



y = true_regression_line[train_set]
from sklearn.linear_model import LinearRegression

clfLR = LinearRegression()

clfLR.fit(x_train, y)

y_pred_LR = clfLR.predict(x_test)

print(clfLR.intercept_)

print(clfLR.coef_[0])

print(clfLR.coef_[1])
from sklearn.neural_network import MLPRegressor

clfMLP = MLPRegressor()

clfMLP.fit(x_train, y)

y_pred_MLP = clfMLP.predict(x_test)
from sklearn.ensemble import GradientBoostingRegressor

clfGBR = GradientBoostingRegressor(random_state=0)

clfGBR.fit(x_train, y)

y_pred_GBR = clfGBR.predict(x_test)
print('Model 1:',mean_absolute_error(true_regression_line[test_set],model1[test_set]))

print('Model 2:', mean_absolute_error(true_regression_line[test_set],model2[test_set]))

print('Average:',mean_absolute_error(true_regression_line[test_set],(model1*.5+model2*.5)[test_set]))

print('MCMC:',mean_absolute_error(true_regression_line[test_set],

                                  (intercept+x1param*model1+x2param*model2)[test_set]))

print('LR:',mean_absolute_error(true_regression_line[test_set], y_pred_LR))

print('MLP:',mean_absolute_error(true_regression_line[test_set], y_pred_MLP))

print('GBM:',mean_absolute_error(true_regression_line[test_set], y_pred_GBR))