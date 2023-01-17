#import libraries
import math
import numpy as np
import random
from scipy import integrate
import matplotlib.pyplot as plt
%matplotlib inline

#define function
def function(x):
    return np.log(x) / x

#create visualization
X = np.arange(0,12,0.0001)
plt.axis([0, 12, 0, 0.5])
plt.plot(X,function(X))
section = np.arange(1, 10, 1/20.)
plt.fill_between(section,function(section), fc="lightsteelblue")
plt.axes().set_aspect(aspect=12)
X = np.arange(0,12,0.0001)

plt.axis([0, 12, 0, 0.5])
plt.plot(X,function(X))
section = np.arange(1, 10, 1/20.)
plt.fill_between(section,function(section), fc="lightsteelblue")
plt.axes().set_aspect(aspect=12)
rectangle = plt.Rectangle((1,0), 9, max(function(X)), fill=None, ec="red", alpha=2)
plt.gca().add_patch(rectangle)
import random

counter = 0
area_rectangle = 9 * max(function(X))

number_trials = 100000
for trial in range(number_trials):
    x_coord = random.uniform(1, 10)
    y_coord = random.uniform(0, max(function(X)))
    
    if y_coord < function(x_coord):
        counter+=1
area_under_curve = (counter/number_trials)* area_rectangle
error = 1/np.sqrt(number_trials)
print("Area under the curve is {:f} (+-{:g})".format(area_under_curve, error))
from scipy.integrate import quad

trials = list(range(number_trials))
# call quad to integrate function() from -1 to 10
res, err = quad(function, 1, 10)

print("The numerical result is {:f} (+-{:g})".format(res, err))
