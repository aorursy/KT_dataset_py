import numpy as np

from pymc3 import *

from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

%matplotlib inline
size = 200

true_intercept = 1

true_slope = 2

x = np.linspace(0, 1, size)

# y = a + b*x

true_regression_line = true_intercept + true_slope * x

# add noise

model1 = true_regression_line + np.random.normal(scale=.5, size=size) #Noisy

model2 = true_regression_line + np.random.normal(scale=.2, size=size) #Less Noisy
print(mean_absolute_error(true_regression_line,model1))

print(mean_absolute_error(true_regression_line,model2))
print(mean_absolute_error(true_regression_line,model1*.5+model2*.5))
data = dict(x1=model1, x2=model2, y=true_regression_line)

with Model() as model:

    # specify glm and pass in data. The resulting linear model, its likelihood and 

    # and all its parameters are automatically added to our model.

    glm.glm('y ~ x1 + x2', data)

    step = NUTS() # Instantiate MCMC sampling algorithm

    trace = sample(2000, step, progressbar=False)
plt.figure(figsize=(7, 7))

traceplot(trace)

plt.tight_layout();
intercept = np.median(trace.Intercept)

print(intercept)

x1param = np.median(trace.x1)

print(x1param)

x2param = np.median(trace.x2)

print(x2param)
print('Model 1:',mean_absolute_error(true_regression_line,model1))

print('Model 2:', mean_absolute_error(true_regression_line,model2))

print('Average:',mean_absolute_error(true_regression_line,model1*.5+model2*.5))

print('MCMC:',mean_absolute_error(true_regression_line,intercept+x1param*model1+x2param*model2))