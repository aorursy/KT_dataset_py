import numpy as np
from scipy.special import legendre
def polify(weight, monic=False):
    '''
    Returns a polynomial which is the sum of Legendre polynomials weighted by `weight`
    '''
    return sum(float(weight[i]) * legendre(i, monic=monic) for i in range(len(weight)))

def polymean(poly):
    '''
    Returns the mean value of a polynomial assuming x is uniform between -1 and 1
    '''
    return np.polyint(poly).c[-2::-2].sum()

def get_target(cplx):
    '''
    Generates target function of order `cplx` - 1
    '''
    return (polify(np.ones(cplx), monic=True)
            / np.sqrt(sum(polymean(legendre(i, monic=True) ** 2) for i in range(cplx))))

def get_legtrans(cplx):
    '''
    Generates Legendre transformation of order `cplx` - 1
    '''
    def legtrans(x):
        return np.concatenate([legendre(i)(x) for i in range(cplx)], axis=1)
    return legtrans
def data(target, noise, size):
    '''
    Generates data of size `size` from target function `target`
    and normal random variable with
    mean zero and standard deviation `noise`
    '''
    x = np.random.uniform(-1, 1, size)
    y = target(x) + np.random.normal(0., noise, size)
    return x, y
def predict(x, weights, cplx):
    '''
    Predicts y by applying Legendre transformation of order `cplx`
    and taking dot products with weights `weights`
    '''
    return get_legtrans(cplx)(x[:, None]) @ weights
def train(x, y, cplx):
    '''
    Trains data `x` and `y` by applying Legendre transformation of order `cplx`
    and linear regression and returns the weights
    '''
    z = get_legtrans(cplx)(x[:, None])
    return np.linalg.solve(z.T @ z, z.T @ y[:, None]).T[0]
def mean_eout(target, noise, cplx, ntrain, niter=1500):
    '''
    Returns the mean out-of-sample error for learning `target` and `noise`
    using a polynomial fit of order `cplx` - 1
    `ntrain` specifies the number of training points
    `ntest` specifies the number of testing points
    `niter` specifies the number of simulations to perform
    '''
    return sum(polymean((target - polify(train(*data(target, noise, ntrain), cplx))) ** 2)
               for i in range(niter)) / niter  + noise ** 2
def get_errors(cplx):
    '''
    Returns the mean errors for (`cplx` - 1)-th order regression
    '''
    return np.array([[mean_eout(target, 0.1, cplx, ntrain)
                      for ntrain in range(80, 120)]
                     for target in map(get_target, (range(2, 42)))])
errors1 = get_errors(3)
import matplotlib.pyplot as plt

plt.imshow(np.flip(errors1, 0), cmap='hot', extent=[80, 119, 2, 41])
plt.colorbar()
plt.show()
errors2 = get_errors(11)
plt.imshow(np.flip(errors2, 0), cmap='hot', extent=[80, 119, 2, 41])
plt.colorbar()
plt.show()
plt.imshow(np.flip(errors2 - errors1, 0), cmap='coolwarm', extent=[80, 119, 2, 41])
plt.colorbar()
plt.show()