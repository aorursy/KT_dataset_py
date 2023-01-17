# Exercise 10.2

%matplotlib inline

import numpy as np

from matplotlib import pyplot
x = np.arange(0, 1, 0.02)

fig = pyplot.plot(x, 4*np.sqrt(1-x**2))
# Hit and miss Monte Carlo integration

ngroups = 16

N = np.zeros(ngroups)

I = np.zeros(ngroups)

E = np.zeros(ngroups)



n0 = 100

for i in range(ngroups):

    N[i] = n0

    x = np.random.random(n0)

    y = np.random.random(n0)

    I[i] = 0

    Nin = 0

    for j in range(n0):

        if(y[j]<np.sqrt(1-x[j]**2)):

            Nin += 1

    I[i] = 4*float(Nin)/float(n0)

    E[i] = abs(I[i] - np.pi)

    print(n0, Nin, I[i], E[i])

    n0 *= 2



pyplot.plot(N, E, ls='-', c='red',lw =3);

pyplot.plot(N, 0.8/np.sqrt(N), ls='-', c='blue', lw=3)

pyplot.xscale('log')

pyplot.yscale('log')
# Simple Monte Carlo Integration

ngroups = 16

N = np.zeros(ngroups)

I = np.zeros(ngroups)

E = np.zeros(ngroups)



n0 = 100

for i in range(ngroups):

    N[i] = n0

    r = np.random.random(n0)

    I[i] = 0.

    for j in range(n0):

        x = r[j]

        I[i] += np.sqrt(1-x**2)

        

    I[i] *=  4./float(n0)

    E[i] = abs(I[i] - np.pi)

    print(n0, I[i], E[i])

    n0 *= 2



pyplot.plot(N, E, ls='-', c='red', lw = 3)

pyplot.plot(N, 0.8/np.sqrt(N), ls='-', c='blue', lw=3)

pyplot.xscale('log')

pyplot.yscale('log')
n0 = 100000

I = np.zeros(n0)

r = np.random.random(n0)

for j in range(n0):

    x = r[j]

    I[j] = 4 * np.sqrt(1-x**2)

    

def group_measurements(ngroups):

    global I, n0

    nmeasurements = n0//ngroups;

    for n in range(ngroups):

        Ig = 0.

        Ig2 = 0.

        for i in range (n*nmeasurements, (n+1)*nmeasurements):

            Ig += I[i]

            Ig2 += I[i]**2

        Ig /= nmeasurements

        Ig2 /= nmeasurements

        sigma = Ig2 - Ig**2

        print(Ig, Ig2, sigma)

group_measurements(10)

print("-----------------------------------")

group_measurements(20)

print("-----------------------------------")

group_measurements(1)
pyplot.xlim(0, 10)

pyplot.ylim(0, 1)

x = np.arange(0, 10, 0.1)

pyplot.plot(x, np.exp(-x), label='e^{-x}')

pyplot.plot(x, np.exp(-x**2), label='e^{-x^2}')

pyplot.plot(x, x**1.5*np.exp(-x), label='x^{3/2}e^{-x}')

pyplot.legend()
# Trapezoidal integration

def trapezoids(func, xmin, xmax, nmax):

    Isim = func(xmin) + func(xmax)

    h = (xmax-xmin)/nmax

    for i in range(1, nmax):

        x = xmin + i *h

        Isim += 2*func(x)

    Isim *= h/2

    return Isim
def f(x):

    return x**1.5*np.exp(-x)
print("Trapezoids: ", trapezoids(f, 0., 20., 100000))
# Simple Monte arlo integration

n0 = 100000

r = np.random.random(n0)



Itot = np.sum(r**1.5*np.exp(-r))

print("Simple Monte Carlo: ", Itot/n0)
# Importance sampling

x = -np.log(r)

Itot = np.sum(x**1.5)

print("Importance Sampling: ", Itot/n0)
pyplot.xlim(0, np.pi)

x = np.arange(0, np.pi, 0.05)

pyplot.plot(x, 1./(x**2 + np.cos(x)**2), label='one')

pyplot.plot(x, np.exp(-x), label='two')

pyplot.plot(x, np.exp(-2*x), label='three')

pyplot.plot(x, np.exp(-0.2*x), label='four')

pyplot.legend()
# Trapezoidal integration

def g(x):

    return 1./(x**2+np.cos(x)**2)



print("Trapezoids: ", trapezoids(g, 0., np.pi, 1000000))



# Simple Monte Carlo integeration

n0 = 1000000

a = np.arange(0.1, 2.1, 0.1)

I = np.arange(0.1, 2.1, 0.1)



r = np.random.random(n0)

I0 = np.sum(1./((r*np.pi)**2 +np.cos(r*np.pi)**2))

print("Simple Monte Carlo: ", I0/n0*np.pi)
# importance sampling

print("Importance Sampling:")

x = -np.log(r)

i = 0

for ai in a:

    norm = (1.-np.exp(-ai*np.pi))/ai

    x1 = norm*x/ai

    Itot = 0.

    Nin = 0

    I2 = 0.

    for xi in x1:

        if(xi <= np.pi):

            Nin += 1

            Itot += g(xi)*np.exp(xi*ai)

            I2 += (g(xi)*np.exp(xi*ai))**2

    Itot *= norm

    I2 *= norm

    

    I[i] = Itot/Nin

    i+=1

    print(ai, Itot/Nin, np.sqrt(abs(Itot**2/Nin**2-I2/Nin))/np.sqrt(Nin))

pyplot.plot(a, I, ls='-', marker='o', c = 'red', lw=3)
delta = 2

xmin = 0.

xmax = 4.0

def f(x):

    return x**2*np.exp(-x)



def P(x):

    global xmin, xmax

    if(x<xmin or x> xmax):

        return 0.

    return np.exp(-x)



def metropolis(xold):

    global delta

    xtrial = np.random.random()

    xtrial = xold +(2*xtrial-1)*delta

    weight = P(xtrial)/P(xold)

    xnew = xold

    if weight >= 1:

        xnew = xtrial

    elif(weight != 0 ):

        r = np.random.random()

        if(r<=weight):

            xnew = xtrial

    return xnew

xwalker = (xmax + xmin)/2.

for i in range(100000):

    xwalker = metropolis(xwalker)



I0 = 0.

N = 300000

x = np.zeros(N)

x[0] = xwalker

for i in range(1, N):

    for j in range(20):

        xwalker = metropolis(xwalker)

    x[i] = xwalker

    I0 += x[i]**2



binwidth = 0.1

pyplot.hist(x, bins= np.arange(xmin-1, xmax+1, 0.1), normed= True)



print("Trapezoids: ", trapezoids(f, xmin, xmax, 100000))

print("Metropolis: ", I0*(1.0-np.exp(-4.0))/N)
fig=pyplot.hist(x**2, bins=np.arange(xmin**2-1, xmax**2+1, 0.1), normed=True)
import numpy as np

import math

import random

from matplotlib import pyplot as plt

from IPython.display import clear_output

PI = 3.1415926

e = 2.71828
# uniform random value from a range

def get_rand_number(min_value, max_value):

    range_value = max_value - min_value

    choice = random.uniform(0, 1)

    return min_value + range_value*choice
def f_of_x(x):

    """

    The function that we want to integerate over

    """

    return (e**(-1*x))/(1+(x-1)**2)
def crude_monte_carlo(num_samples=5000):

    lower_bound = 0

    upper_bound = 50

    

    sum_of_samples = 0

    for i in range(num_samples):

        x = get_rand_number(lower_bound, upper_bound)

        sum_of_samples += f_of_x(x)

    return (upper_bound - lower_bound)*float(sum_of_samples/num_samples)
print(crude_monte_carlo(100000))
def get_crude_MC_variance(num_samples):

    """

    This function returns the variance of the Crude Monte Carlo.

    Note that the inputed number of samples does not necessarily need to correpsond to the number of samples used in the Monte Carlo Simulation.

    Args:

    - num_samples (int)

    Return:

    - Variance for Crude Monte Carlo approximation of f(x) (float)

    """

    int_max = 5 # this is th emax of our integration range

    # get the average of squares

    running_total = 0

    for i in range(num_samples):

        x = get_rand_number(0, int_max)

        running_total += f_of_x(x)**2

    sum_of_sqs = running_total*int_max/num_samples

    

    # get square of average

    running_total = 0

    for i in range(num_samples):

        x = get_rand_number(0, int_max)

        running_total += f_of_x(x)

    sq_ave = (int_max*running_total/num_samples)**2

    

    return sum_of_sqs - sq_ave
def step_f(x):

    return 1 if x<=2 else 0

for i in range(10):

    x = get_rand_number(0, 6)

    print(x, step_f(x))
(4.0/10.0)*6
xs = [float(i/50) for i in range(int(50*PI*2))]

ys = [f_of_x(x) for x in xs]

plt.xlim([0, 6])

plt.ylim([0, 0.5])

plt.plot(xs, ys)

plt.title("f(x)");
# finding the optimal lambda

def g_of_x(x, A, lamda):

    e = 2.71828

    return A*math.pow(e, -1*lamda*x)



def inverse_G_of_r(r, lamda):

    return (-1 * math.log(float(r)))/lamda



def get_IS_variance(lamda, num_samples):

    """

    This function calculates the variance if a Monte Carlo using importance sampling.

    Args:

    - lamda (float): lambda value of g(x) being tested

    Return:

    - Variance

    """

    A = lamda

    int_max = 5

    # get sum of squares

    running_total = 0

    for i in range(num_samples):

        x = get_rand_number(0, int_max)

        running_total += (f_of_x(x)/g_of_x(x, A, lamda)/g_of_x(x, A, lamda))**2

        

    sum_of_sqs = running_total/num_samples

    

    # get squared average

    running_total = 0

    for i in range(num_samples):

        x = get_rand_number(0, int_max)

        running_total += f_of_x(x)/g_of_x(x, A, lamda)

    sq_ave = (running_total/num_samples)**2

    

    return sum_of_sqs - sq_ave



# get variance as a funciton of lambda by testing many different lambdas

test_lambdas = [i*.05 for i in range (1, 61)]

variances = []

for i, lamda in enumerate(test_lambdas):

    print(f"lambda {i+1}/{len(test_lambdas)}: {lamda}")

    A = lamda

    variances.append(get_IS_variance(lamda, 10000))

    clear_output(wait=True)



optimal_lambda = test_lambdas[np.argmin(np.asarray(variances))]

IS_variance = variances[np.argmin(np.asarray(variances))]



print(f"Optimal Lambda: {optimal_lambda}")

print(f"Optimal Variance: {IS_variance}")

print(f"Error: {(IS_variance/10000)**0.5}")
def importance_sampling_MC(lamda, num_samples):

    A  = lamda

    running_total = 0

    for i in range(num_samples):

        r = get_rand_number(0, 1)

        running_total += f_of_x(inverse_G_of_r(r, lamda=lamda))/g_of_x(inverse_G_of_r(r, lamda=lamda), A, lamda)

    approximation = float(running_total/num_samples)

    return approximation
# run simulation

num_samples = 10000

approx = importance_sampling_MC(optimal_lambda, num_samples)

variance = get_IS_variance(optimal_lambda, num_samples)

error = (variance/num_samples)**0.5



# Display results

print(f"Importance samping approximation: {approx}")

print(f"Variance: {variance}")

print(f"Error: {error}")