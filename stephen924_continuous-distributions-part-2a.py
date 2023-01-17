import math

import numpy as np

import pandas as pd

import holoviews as hv

from holoviews import opts

from bokeh.io import output_file, save, show

import scipy

import scipy.special

from scipy.special import gamma

import scipy.stats as stats

hv.extension('bokeh')
def histogram(hist, x, pdf, cdf, label):

    pdf = hv.Curve((x, pdf), label='PDF')

    cdf = hv.Curve((x, cdf), label='CDF')

    return (hv.Histogram(hist, vdims='P(r)').opts(fill_color="gray") * pdf * cdf).relabel(label)
label = "Normal Distribution (μ=0, σ=0.5)"

mu, sigma = 0, 0.5



measured = np.random.normal(mu, sigma, 10000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(-2, 2, 1000)

pdf = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2 / (2*sigma**2))

cdf = (1+scipy.special.erf((x-mu)/np.sqrt(2*sigma**2)))/2

norm = histogram(hist, x, pdf, cdf, label)
norm.opts(width = 800, height = 700 , show_grid=True)
def hist(mu, sigma):

    data = np.random.normal(mu, sigma, 1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['mu', 'sigma'])

hmap.redim.range(mu = (0,2), sigma = (0.5,1)).opts(width = 700, height = 600 , show_grid=True).relabel('Normal Histogram')
def pdf(mu, sigma):

    xs = np.linspace(mu-10, mu + 10, 1000)

    ys = [1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2 / (2*sigma**2)) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['mu', 'sigma'])

hmap1.redim.range(mu = (0,4), sigma = (0.5,5)).opts(width = 700, height = 600 , show_grid=True).relabel('Normal PDF')
def cdf(mu, sigma):

    xs = np.linspace(mu -2, mu + 2, 1000)

    return hv.Curve((xs, [(1+scipy.special.erf((x-mu)/np.sqrt(2*sigma**2)))/2 for x in xs]))



hmap2 = hv.DynamicMap(cdf, kdims=['mu', 'sigma'])

hmap2.redim.range(mu = (0,4), sigma = (0.5,1)).opts(width = 700, height = 600 , show_grid=True).relabel('Normal CDF')
label = "Exponential Distribution (lambda=1.5)"

scale, size = 1.5, 1000



measured = np.random.exponential(scale, size)

hist = np.histogram(measured,density=True, bins=30)



x = np.linspace(0, 8, 1000)

pdf = scale * np.e**(-scale * x)

cdf = 1 - np.e **(-scale * x)

exp = histogram(hist, x, pdf, cdf, label)
exp.opts(width = 800, height = 700 , show_grid=True)
def hist(lamda):

    # lambda is a key word so it will be written as lamda

    data = np.random.exponential(lamda, 1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['lamda'])

hmap.redim.range(lamda = (0.5,1.5)).opts(width = 700, height = 600 , show_grid=True).relabel('Exponential Histogram')
def pdf(lamda):

    xs = np.linspace(lamda-10, lamda + 10, 1000)

    return hv.Curve((xs, [lamda * np.e**(-lamda * x) for x in xs]))



hmap1 = hv.DynamicMap(pdf, kdims=['lamda',])

hmap1.redim.range(lamda = (0.2,2)).opts(width = 700, height = 600 , show_grid=True).relabel('Exponential PDF')
def cdf(lamda):

    xs = np.linspace(lamda -2, lamda + 2, 1000)

    return hv.Curve((xs, [1 - np.e**(-lamda *x) for x in xs]))



hmap2 = hv.DynamicMap(cdf, kdims=['lamda'])

hmap2.redim.range(lamda = (0.2,2)).opts(width = 700, height = 600 , show_grid=True).relabel('Exponential CDF')
label = "Weibull Distribution (λ=1, k=1.25)"

lam, k = 1, 1.25



measured = lam*(-np.log(np.random.uniform(0, 1, 1000)))**(1/k)

hist = np.histogram(measured, density=True, bins=40)



x = np.linspace(0, 8, 1000)

pdf = (k/lam)*(x/lam)**(k-1) * np.exp(-(x/lam)**k)

cdf = 1 - np.exp(-(x/lam)**k)

weibull = histogram(hist, x, pdf, cdf, label)
weibull.opts(width = 800, height = 700 , show_grid=True)
def hist(lam, k):

    data = lam*(-np.log(np.random.uniform(0, 1, 1000)))**(1/k)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['lam', 'k'])

hmap.redim.range(lam = (1,4), k = (1,5)).opts(width = 700, height = 600 , show_grid=True).relabel('Weibull Histogram')
def pdf(lam, k):

    xs = np.linspace(0, 8, 1000)

    ys = [(k/lam)*(x/lam)**(k-1) * np.exp(-(x/lam)**k) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['lam', 'k'])

hmap1.redim.range(lam = (1,4), k = (1,5)).opts(width = 700, height = 600 , show_grid=True).relabel('Weibull PDF')
def cdf(lam, k):

    xs = np.linspace(0, 8, 1000)

    ys = [(1 - np.exp(-(x/lam)**k)) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['lam', 'k'])

hmap1.redim.range(lam = (1,4), k = (1,5)).opts(width = 700, height = 600 , show_grid=True).relabel('Weibull CDF')
label = "Log Normal Distribution (μ=0, σ=0.5)"

mu, sigma = 0, 0.5



measured = np.random.lognormal(mu, sigma, 1000)

hist = np.histogram(measured, density=True, bins=50)



x = np.linspace(0.001, 8.0, 1000)

# there is a devided by zero error, when we start from 0 at x

pdf = 1/(x* sigma * np.sqrt(2*np.pi)) * np.exp(-(np.log(x)-mu)**2 / (2*sigma**2))

cdf = (1+scipy.special.erf((np.log(x)-mu)/(np.sqrt(2)*sigma)))/2

lognorm = histogram(hist, x, pdf, cdf, label)
lognorm.opts(width = 800, height = 700 , show_grid=True)
def hist(mu, sigma):

    data = np.random.lognormal(mu, sigma, 1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['mu', 'sigma'])

hmap.redim.range(mu = (1,4), sigma = (0.5,1.5)).opts(width = 700, height = 600 , show_grid=True).relabel('Log Normal Histogram')
def pdf(mu, sigma):

    xs = np.linspace(0, 8, 1000)

    ys = [(1/(x* sigma * np.sqrt(2*np.pi)) * np.exp(-(np.log(x)-mu)**2 / (2*sigma**2))) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['mu', 'sigma'])

hmap1.redim.range(mu = (1,4), sigma = (0.5,1.5)).opts(width = 700, height = 600 , show_grid=True).relabel('Log Normal PDF')
def cdf(mu, sigma):

    xs = np.linspace(0, 8, 1000)

    ys = [(1+scipy.special.erf((np.log(x)-mu)/(np.sqrt(2)*sigma)))/2 for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['mu', 'sigma'])

hmap1.redim.range(mu = (1,4), sigma = (0.5,1.5)).opts(width = 700, height = 600 , show_grid=True).relabel('Log Normal CDF')
label = "Gamma Distribution (k=1, θ=2)"

k, theta = 1.0, 2.0



measured = np.random.gamma(k, theta, 1000)

hist = np.histogram(measured, density=True, bins=50)



x = np.linspace(0, 10, 1000)

pdf = x**(k-1) * np.exp(-x/theta) / (theta**k * scipy.special.gamma(k))

cdf = scipy.special.gammainc(k, x/theta) / scipy.special.gamma(k)

gamma = histogram(hist, x, pdf, cdf, label)
gamma.opts(width = 800, height = 700 , show_grid=True)
def hist(k, theta):

    data = np.random.gamma(k, theta, 1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['k', 'theta'])

hmap.redim.range(k = (0.5,5), theta = (0.5,2.5)).opts(width = 700, height = 600 , show_grid=True).relabel('Gamma Histogram')
def pdf(k, theta):

    xs = np.linspace(0, 10, 1000)

    ys = [x**(k-1) * np.exp(-x/theta) / (theta**k * scipy.special.gamma(k)) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['k', 'theta'])

hmap1.redim.range(k = (0.5,5), theta = (0.5,2.5)).opts(width = 700, height = 600 , show_grid=True).relabel('Gamma PDF')
def cdf(k, theta):

    xs = np.linspace(0, 10, 1000)

    ys = [scipy.special.gammainc(k, x/theta) / scipy.special.gamma(k) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['k', 'theta'])

hmap1.redim.range(k = (0.5,5), theta = (0.5,2.5)).opts(width = 700, height = 600 , show_grid=True).relabel('Gamma CDF')
label = "Beta Distribution (α=2, β=2)"

alpha, beta = 2.0, 2.0



measured = np.random.beta(alpha, beta, 1000)

hist = np.histogram(measured, density=True, bins=50)



x = np.linspace(0, 1, 1000)

pdf = x**(alpha-1) * (1-x)**(beta-1) / scipy.special.beta(alpha, beta)

cdf = scipy.special.btdtr(alpha, beta, x)

beta = histogram(hist, x, pdf, cdf, label)
beta.opts(width = 800, height = 700 , show_grid=True)
def hist(alpha, beta):

    data = np.random.beta(alpha, beta, 1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['alpha', 'beta'])

hmap.redim.range(alpha = (0.5,5), beta = (0.5,5)).opts(width = 700, height = 600 , show_grid=True).relabel('Beta Histogram')
def pdf(alpha, beta):

    xs = np.linspace(0, 2.5, 1000)

    ys = [x**(alpha-1) * (1-x)**(beta-1) / scipy.special.beta(alpha, beta) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['alpha', 'beta'])

hmap1.redim.range(alpha = (0.5,5), beta = (0.5,5)).opts(width = 700, height = 600 , show_grid=True).relabel('Beta PDF')
def cdf(alpha, beta):

    xs = np.linspace(0, 1, 1000)

    ys = [scipy.special.btdtr(alpha, beta, x) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['alpha', 'beta'])

hmap1.redim.range(alpha = (0.5,5), beta = (0.5,5)).opts(width = 700, height = 600 , show_grid=True).relabel('Beta CDF')
label = "Chu-Squared Distribution (k = 3)"

k = 3



measured = np.random.chisquare(k,1000)

hist = np.histogram(measured, density=True, bins=50)



x = np.linspace(0, 8, 1000)

#pdf = (x**((k/2)-1) * np.exp**(-x/2))/(2**(k/2)*gamma(k/2))

pdf= stats.chi2.pdf(x, k)

cdf = stats.chi2.cdf(x, k, loc=0, scale=1)

chi = histogram(hist, x, pdf, cdf, label)
chi.opts(width = 800, height = 700 , show_grid=True)
def hist(k):

    data = np.random.chisquare(k, 1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['k'])

hmap.redim.range(k = (1,9)).opts(width = 700, height = 600 , show_grid=True).relabel('Chi-Squared Histogram')
def pdf(k):

    xs = np.linspace(0, 8, 1000)

    ys = [stats.chi2.pdf(x, k) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['k'])

hmap1.redim.range(k = (1,9)).opts(width = 700, height = 600 , show_grid=True).relabel('Chi-Squared PDF')
def cdf(k):

    xs = np.linspace(0, 8, 1000)

    ys = [stats.chi2.cdf(x, k, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['k'])

hmap1.redim.range(k = (1,9)).opts(width = 700, height = 600 , show_grid=True).relabel('Chi-Squared CDF')
label = "F Distribution (dfnum, dfden = 5, 2)"

dfnum, dfden = 5, 2



measured = np.random.f(dfnum, dfden, 30)

hist = np.histogram(measured, density=True, bins=25)



x = np.linspace(0.01, 25, 1000)

pdf= stats.f.pdf(x, dfnum, dfden, loc=0, scale=1)

cdf = stats.f.cdf(x, dfnum, dfden, loc=0, scale=1)

f = histogram(hist, x, pdf, cdf, label)
f.opts(width = 800, height = 700 , show_grid=True)
def hist(dfnum, dfden):

    data = np.random.f(dfnum, dfden, 50)

    frequencies, edges = np.histogram(data, 20)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['dfnum', 'dfden'])

hmap.redim.range(dfnum = (1,100), dfden = (1,100)).opts(width = 700, height = 600 , show_grid=True).relabel('F Histogram')
def pdf(dfnum, dfden):

    xs = np.linspace(0, 5, 1000)

    ys = [stats.f.pdf(x, dfnum, dfden, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['dfnum', 'dfden'])

hmap1.redim.range(dfnum = (5,100), dfden = (1,100)).opts(width = 700, height = 600 , show_grid=True).relabel('F PDF')
def cdf(dfnum, dfden):

    xs = np.linspace(0, 5, 1000)

    ys = [stats.f.cdf(x, dfnum, dfden, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['dfnum', 'dfden'])

hmap1.redim.range(dfnum = (5,100), dfden = (1,100)).opts(width = 700, height = 600 , show_grid=True).relabel('F CDF')
label = "Gumbel Distribution (mu, beta = 1, 2)"

mu, beta = 1, 2



measured = np.random.gumbel(mu, beta, 1000)

hist = np.histogram(measured, density=True, bins=40)



x = np.linspace(-5, 20, 1000)

pdf = stats.gumbel_l.pdf(x, mu, beta)

cdf = stats.gumbel_l.cdf(x, mu, beta)

gum = histogram(hist, x, pdf, cdf, label)
gum.opts(width = 800, height = 700 , show_grid=True)
def hist(mu, beta):

    data = np.random.gumbel(mu, beta, 1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['mu', 'beta'])

hmap.redim.range(mu = (1,3), beta = (2,4)).opts(width = 700, height = 600 , show_grid=True).relabel('Gumbel Histogram')
def pdf(mu, beta):

    xs = np.linspace(-7, 10, 1000)

    ys = [stats.gumbel_l.pdf(x, mu, beta) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['mu','beta'])

hmap1.redim.range(mu = (1,3),beta=(2,4)).opts(width = 700, height = 600 , show_grid=True).relabel('Gumbel Right PDF')
def pdf(mu, beta):

    xs = np.linspace(-7, 10, 1000)

    ys = [stats.gumbel_l.pdf(x, mu, beta) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['mu','beta'])

hmap1.redim.range(mu = (1,3),beta=(1,4)).opts(width = 700, height = 600 , show_grid=True).relabel('Gumbel Left PDF')
def cdf(mu, beta):

    xs = np.linspace(-5, 7, 1000)

    ys = [stats.gumbel_l.cdf(x, mu, beta) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['mu','beta'])

hmap1.redim.range(mu = (1,3),beta=(1,4)).opts(width = 700, height = 600 , show_grid=True).relabel('Gumbel Left CDF')
def cdf(mu, beta):

    xs = np.linspace(-5, 7, 1000)

    ys = [stats.gumbel_r.cdf(x, mu, beta) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['mu','beta'])

hmap1.redim.range(mu = (1,3),beta=(1,4)).opts(width = 700, height = 600 , show_grid=True).relabel('Gumbel Right CDF')
label = "Pareto Distribution (b = 3)"

b = 3



measured = stats.pareto.rvs(b, size=1000)

hist = np.histogram(measured, density=True, bins=50)



x = np.linspace(0.001, 10, 1000)

# divided be zero exception

pdf = stats.pareto.pdf(x, b, loc=0, scale=1)

cdf = stats.pareto.cdf(x, b, loc=0, scale=1)

pareto = histogram(hist, x, pdf, cdf, label)
pareto.opts(width = 800, height = 700 , show_grid=True)
def hist(b):

    data = stats.pareto.rvs(b, size=100)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['b'])

hmap.redim.range(b = (1,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Pareto Histogram')
def pdf(b):

    xs = np.linspace(0, 5, 1000)

    ys = [stats.pareto.pdf(x, b, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['b'])

hmap1.redim.range(b = (2,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Pareto PDF')
def cdf(b):

    xs = np.linspace(0, 5, 1000)

    ys = [(stats.pareto.cdf(x, b, loc=0, scale=1)) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['b'])

hmap1.redim.range(b = (2,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Pareto CDF')
label = "Alpha Continuous Random Variable Distribution (a = 4)"

a = 4



measured = stats.alpha.rvs(a, size=1000)

hist = np.histogram(measured, density=True, bins=50)



x = np.linspace(0.001, 1, 1000)

pdf = stats.alpha.pdf(x, a, loc=0, scale=1)

cdf = stats.alpha.cdf(x, a, loc=0, scale=1)

alpha = histogram(hist, x, pdf, cdf, label)

alpha.opts(width = 800, height = 700 , show_grid=True)
def hist(a):

    data = stats.alpha.rvs(a, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['a'])

hmap.redim.range(a = (2,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Alpha Continuous Random Variable Histogram')
def pdf(a):

    xs = np.linspace(0, 2, 1000)

    ys = [stats.alpha.pdf(x, a, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['a'])

hmap1.redim.range(a = (2,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Alpha Continuous Random Variable PDF')
def cdf(a):

    xs = np.linspace(0, 2, 1000)

    ys = [stats.alpha.cdf(x, a, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['a'])

hmap1.redim.range(a = (2,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Alpha Continuous Random Variable CDF')
label = "Anglit Continuous Random Variable Distribution"



measured = stats.anglit.rvs(size=1000)

hist = np.histogram(measured, density=True, bins=50)



x = np.linspace(-1, 1, 1000)

pdf = stats.anglit.pdf(x, loc=0, scale=1)

cdf = stats.anglit.cdf(x, loc=0, scale=1)

alpha = histogram(hist, x, pdf, cdf, label)

alpha.opts(width = 800, height = 700 , show_grid=True)
label = "Arcsine Continuous Random Variable Distribution"



measured = stats.arcsine.rvs(size=1000)

hist = np.histogram(measured, density=True, bins=50)



x = np.linspace(0, 1, 1000)

pdf = stats.arcsine.pdf(x, loc=0, scale=1)

cdf = stats.arcsine.cdf(x, loc=0, scale=1)

arc = histogram(hist, x, pdf, cdf, label)

arc.opts(width = 800, height = 700 , show_grid=True)
label = "ARGUS Distribution (chi = 2)"

chi = 2



measured = stats.argus.rvs(chi, size=1000)

hist = np.histogram(measured, density=True, bins=50)



x = np.linspace(0, 1, 1000)

pdf = stats.argus.pdf(x,chi, loc=0, scale=1)

cdf = stats.argus.cdf(x,chi, loc=0, scale=1)

argus = histogram(hist, x, pdf, cdf, label)

argus.opts(width = 800, height = 700 , show_grid=True)
def hist(chi):

    data = stats.argus.rvs(chi, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['chi'])

hmap.redim.range(chi = (1,10)).opts(width = 700, height = 600 , show_grid=True).relabel('ARGUS Distribution Histogram')
def pdf(chi):

    xs = np.linspace(0, 1, 1000)

    ys = [stats.argus.pdf(x,chi, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['chi'])

hmap1.redim.range(chi = (1,10)).opts(width = 700, height = 600 , show_grid=True).relabel('ARGUS Distribution PDF')
def cdf(chi):

    xs = np.linspace(0, 1, 1000)

    ys = [stats.argus.cdf(x,chi, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['chi'])

hmap1.redim.range(chi = (1,10)).opts(width = 700, height = 600 , show_grid=True).relabel('ARGUS Distribution CDF')
label = "Beta Prime Distribution (a = 4, b =6)"

a = 4

b = 6



measured = stats.betaprime.rvs(a, b, size=1000)

hist = np.histogram(measured, density=True, bins=50)



x = np.linspace(0, 8, 1000)

pdf = stats.betaprime.pdf(x, a, b, loc=0, scale=1)

cdf = stats.betaprime.pdf(x, a, b, loc=0, scale=1)

betap = histogram(hist, x, pdf, cdf, label)

betap.opts(width = 800, height = 700 , show_grid=True)
def hist(a, b):

    data = stats.betaprime.rvs(a, b, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['a', 'b'])

hmap.redim.range(a = (1,10), b=(1,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Beta Prime Distribution Histogram')
def pdf(a, b):

    xs = np.linspace(0, 2, 1000)

    ys = [stats.betaprime.pdf(x, a, b, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['a', 'b'])

hmap1.redim.range(a = (1,10), b=(1,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Beta Prime Distribution PDF')
def cdf(a, b):

    xs = np.linspace(0, 2, 1000)

    ys = [stats.betaprime.cdf(x, a, b, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['a', 'b'])

hmap1.redim.range(a = (1,10), b=(1,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Beta Prime Distribution CDF')
label = "Bradford Distribution (c = 0.25)"

c = 0.25



measured = stats.bradford.rvs(c, size=1000)

hist = np.histogram(measured, density=True, bins=50)



x = np.linspace(0, 1, 1000)

pdf = stats.bradford.pdf(x, c, loc=0, scale=1)

cdf = stats.bradford.cdf(x, c, loc=0, scale=1)

brad = histogram(hist, x, pdf, cdf, label)

brad.opts(width = 800, height = 700 , show_grid=True)
def hist(c):

    data = stats.bradford.rvs(c, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['c'])

hmap.redim.range(c = (0.1, 1)).opts(width = 700, height = 600 , show_grid=True).relabel('Bradford Distribution Histogram')
def pdf(c):

    xs = np.linspace(0, 2, 1000)

    ys = [stats.bradford.pdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['c'])

hmap1.redim.range(c = (0.1,1)).opts(width = 700, height = 600 , show_grid=True).relabel('Bradford Distribution PDF')
def cdf(c):

    xs = np.linspace(0, 2, 1000)

    ys = [stats.bradford.cdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['c'])

hmap1.redim.range(c = (0.1,1)).opts(width = 700, height = 600 , show_grid=True).relabel('Bradford Distribution CDF')
label = "Burr (Type III) Continuous Random Variable Distribution (c = 10, d = 4)"

c, d = 10, 4



measured = stats.burr.rvs(c, d, size=1000)

hist = np.histogram(measured, density=True, bins=50)



x = np.linspace(0, 3, 1000)

pdf = stats.burr.pdf(x, c, d, loc=0, scale=1)

cdf = stats.burr.cdf(x, c, d, loc=0, scale=1)

burr = histogram(hist, x, pdf, cdf, label)

burr.opts(width = 800, height = 700 , show_grid=True)
def hist(c, d):

    data = stats.burr.rvs(c, d, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['c', 'd'])

hmap.redim.range(c = (5,10), d=(1,5)).opts(width = 700, height = 600 , show_grid=True).relabel('Burr (Type III) Continuous Random Variable Distribution Histogram')
def pdf(c, d):

    xs = np.linspace(0, 3, 1000)

    ys = [stats.burr.pdf(x, c, d, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['c', 'd'])

hmap1.redim.range(c = (5,10), d=(1,5)).opts(width = 700, height = 600 , show_grid=True).relabel('Burr (Type III) Continuous Random Variable Distribution Distribution PDF')
def cdf(c, d):

    xs = np.linspace(0, 3, 1000)

    ys = [stats.burr.cdf(x, c, d, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['c', 'd'])

hmap1.redim.range(c = (5,10), d=(1,5)).opts(width = 700, height = 600 , show_grid=True).relabel('Burr (Type III) Continuous Random Variable Distribution Distribution CDF')
label = "Burr (Type XII) Continuous Random Variable Distribution (c = 10, d = 4)"

c, d = 10, 4



measured = stats.burr12.rvs(c, d, size=1000)

hist = np.histogram(measured, density=True, bins=50)



x = np.linspace(0, 1.5, 1000)

pdf = stats.burr12.pdf(x, c, d, loc=0, scale=1)

cdf = stats.burr12.cdf(x, c, d, loc=0, scale=1)

burr12 = histogram(hist, x, pdf, cdf, label)

burr12.opts(width = 800, height = 700 , show_grid=True)
def hist(c, d):

    data = stats.burr12.rvs(c, d, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['c', 'd'])

hmap.redim.range(c = (5,10), d=(1,5)).opts(width = 700, height = 600 , show_grid=True).relabel('Burr (Type XII) Continuous Random Variable Distribution Histogram')
def pdf(c, d):

    xs = np.linspace(0, 3, 1000)

    ys = [stats.burr12.pdf(x, c, d, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['c', 'd'])

hmap1.redim.range(c = (5,10), d=(1,5)).opts(width = 700, height = 600 , show_grid=True).relabel('Burr (Type XII) Continuous Random Variable Distribution PDF')
def cdf(c, d):

    xs = np.linspace(0, 3, 1000)

    ys = [stats.burr12.cdf(x, c, d, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['c', 'd'])

hmap1.redim.range(c = (5,10), d=(1,5)).opts(width = 700, height = 600 , show_grid=True).relabel('Burr (Type XII) Continuous Random Variable Distribution CDF')
label = "Cauchy distribution (gamma = 1)"



gamma = 1

measured = stats.cauchy.rvs(gamma,size=250)

hist = np.histogram(measured, density=True, bins=50)



x = np.linspace(-20, 50, 1000)

pdf = stats.cauchy.pdf(x, gamma)

cdf = stats.cauchy.cdf(x, gamma)

chauchy = histogram(hist, x, pdf, cdf, label)

chauchy.opts(width = 800, height = 700 , show_grid=True)
def hist(gamma):

    data = stats.cauchy.rvs(gamma, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['gamma'])

hmap.redim.range(gamma = (0.5, 2)).opts(width = 700, height = 600 , show_grid=True).relabel('Cauchy Distribution Histogram')
def pdf(gamma):

    xs = np.linspace(-5, 5, 1000)

    ys = [stats.cauchy.pdf(x, gamma) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['gamma'])

hmap1.redim.range(gamma = (0.1,2)).opts(width = 700, height = 600 , show_grid=True).relabel('Cauchy Distribution PDF')
def cdf(gamma):

    xs = np.linspace(-5, 5, 1000)

    ys = [stats.cauchy.cdf(x, gamma) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['gamma'])

hmap1.redim.range(gamma = (0.1,2)).opts(width = 700, height = 600 , show_grid=True).relabel('Cauchy Distribution CDF')
label = "Chi Distribution (df = 4)"

df = 4



measured = stats.chi.rvs(df, size=1000)

hist = np.histogram(measured, density=True, bins=50)



x = np.linspace(0, 5, 1000)

pdf = stats.chi.pdf(x, df, loc=0, scale=1)

cdf = stats.chi.cdf(x, df, loc=0, scale=1)

chauchy = histogram(hist, x, pdf, cdf, label)

chauchy.opts(width = 800, height = 700 , show_grid=True)
def hist(df):

    data = stats.chi.rvs(df, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['df'])

hmap.redim.range(df = (1, 10)).opts(width = 700, height = 600 , show_grid=True).relabel('Chi Distribution Histogram')
def pdf(df):

    xs = np.linspace(0, 5, 1000)

    ys = [stats.chi.pdf(x, df, loc=0, scale=1)  for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['df'])

hmap1.redim.range(df = (1,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Chi Distribution PDF')
def cdf(df):

    xs = np.linspace(0, 5, 1000)

    ys = [stats.chi.cdf(x, df, loc=0, scale=1)  for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['df'])

hmap1.redim.range(df = (1,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Chi Distribution CDF')
label = "Cosine Continuous Random Variable Distribution"



measured = stats.cosine.rvs(size=1000)

hist = np.histogram(measured, density=True, bins=50)



x = np.linspace(-5, 5, 1000)

pdf = stats.cosine.pdf(x, loc=0, scale=1)

cdf = stats.cosine.cdf(x, loc=0, scale=1)

chauchy = histogram(hist, x, pdf, cdf, label)

chauchy.opts(width = 800, height = 700 , show_grid=True)