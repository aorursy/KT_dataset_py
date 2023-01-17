from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3,3,100)
y = np.array([norm.pdf(m) for m in x])

plt.scatter(x, y)
import math

sigma = 4
n = 15

# c such that: P(Z < c) = 0.975
X = np.random.normal(50, 4, 15)
c = norm.ppf(0.975)

# lower and upper bounds
L = np.mean(X) - c*sigma/math.sqrt(n)
U = np.mean(X) + c*sigma/math.sqrt(n)

print("The 95% confidence interval for the mean: [{:.3f},{:.3f}]".format(L, U))
# we sample the data 10000 times and each sample has the length of 15
number_of_experiments = 10000
n = 15

# we will count the number of times the real mean is in the interval
counter = 0

#

X = np.random.normal(50, 4, (number_of_experiments, n))

for i in range(number_of_experiments):
    sample = X[i,:]
    # we construct the lower and upper bounds
    lower = np.mean(sample) - c*sigma/math.sqrt(n)
    upper = np.mean(sample) + c*sigma/math.sqrt(n)
    if (lower < 50) & (50 < upper):
        counter += 1
        
print("The real value of the mean appears in {:.2f}% of all confidence intervals constructed.".format(100*counter/number_of_experiments))
print("The length of the interval is: {:.2f}.".format(upper-lower))
variance = np.linspace(0, 30, 100)
n_needed = np.array([4*c*c*m for m in variance])

plt.scatter(variance, n_needed)

print("The slope of the graph is: {:.2f}".format(4*c*c))
n1 = math.ceil(4*c*c*16)

X = np.random.normal(50, 4, n1)

lower = np.mean(X)-c*4/math.sqrt(n1)
upper = np.mean(X)+c*4/math.sqrt(n1)

print("VARIANCE: 16")
print("The 95% confidence interval: ({:.2f}, {:.2f})".format(lower, upper))
print("The length of the interval: {:.2f}".format(upper-lower))
print("Sample size: {}".format(n1))

print("\nVARIANCE: 32")
X = np.random.normal(50, math.sqrt(2)*4, 2*n1)
lower = np.mean(X)-c*math.sqrt(2)*4/math.sqrt(2*n1)
upper = np.mean(X)+c*math.sqrt(2)*4/math.sqrt(2*n1)
print("The 95% confidence interval: ({:.2f}, {:.2f})".format(lower, upper))
print("The length of the interval: {:.2f}".format(upper-lower))
print("Sample size: {}".format(2*n1))