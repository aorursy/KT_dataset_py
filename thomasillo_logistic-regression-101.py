# Here's the data
Hours = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]
Pass = [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
p = np.linspace(0,1, num=100)
l = lambda p: np.log(p/(1-p))
ax=plt.gca()
ax.plot(p, l(p))
ax.axvline(0.5, color='k', alpha=0.5, ls=':')
ax.grid(True)
ax.set_xlabel('$P$')
ax.set_ylabel('$L$')
ax.text(0.1, 3, "$\\log \\frac {P(x)} {1-P(x)}$");
h_theta = lambda hours, th0, th1: 1./(1+np.exp(-th0 + th1 * hours))
data = np.array([Hours, Pass]).T

def prob(datum, th0, th1):
    h = h_theta(datum[0], th0, th1)
    return h**datum[1] * (1-h)**(1-datum[1])
    
def likelihood(th0, th1):
    l = 1
    for datum in data:
        l *= prob(datum, th0, th1)
    return l

log_likelihood = lambda th0, th1:  np.log(likelihood(th0, th1))
# Optimize
import scipy.optimize
f = lambda th: -log_likelihood(th[0],th[1])
sol = scipy.optimize.minimize(f, (0,0))
print(sol)
t0min, t0max = -10, 2
t1min, t1max = -4, 2
tt0, tt1 = np.mgrid[t0min:t0max:50j, t1min:t1max:50j]
plt.imshow(log_likelihood(tt0, tt1).T, origin='lower', 
           extent=(t0min,t0max,t1min, t1max), aspect='auto')
plt.plot(*sol.x, 'rx', ms=8)
plt.axvline(sol.x[0], color='r', alpha=0.5, ls=':')
plt.axhline(sol.x[1], color='r', alpha=0.5, ls=':')
plt.colorbar(label='$-\\log(\\Lambda)$')
plt.gca().set_xlabel('$\\theta_0$')
_=plt.gca().set_ylabel('$\\theta_1$')
import sklearn.linear_model
# Tune C value (inverse of regularization strength) to have quasi no regularization.
model = sklearn.linear_model.LogisticRegression(C=1E6, solver='lbfgs')
model.fit(data.T[0].reshape(-1,1), data.T[1])
model.coef_, model.intercept_
import scipy.stats
xx = np.linspace(-5,5, num=100)
plt.plot(xx, scipy.stats.logistic(scale=np.sqrt(3/np.pi**2)).pdf(xx), label='logistic')
plt.plot(xx, scipy.stats.norm().pdf(xx), label='normal')
plt.legend();
