import math

import numpy as np

import pandas as pd

import holoviews as hv

from holoviews import opts

from bokeh.io import output_file, save, show

import scipy

import scipy.special

from scipy.stats import boltzmann

from scipy.special import gamma

import scipy.stats as stats

hv.extension('bokeh')
def histogram(hist, x, pmf, cdf, label):

    pmf = hv.Curve((x, pmf), label='PMF')

    cdf = hv.Curve((x, cdf), label='CDF')

    return (hv.Histogram(hist, vdims='P(r)').opts(fill_color="gray") * pmf * cdf).relabel(label)
label = "Bernoulli Distribution (p=0.6)"

p = 0.6



measured = stats.bernoulli.rvs(size=100,p=0.6)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 2, 100)

pmf = stats.bernoulli.pmf(x, p, loc=0)

cdf = stats.bernoulli.cdf(x, p)

bern = histogram(hist, x, pmf, cdf, label)
bern.opts(width = 800, height = 700 , show_grid=True)
def hist(p):

    data = stats.bernoulli.rvs(size=100,p=0.6)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['p'])

hmap.redim.range(p = (0.1,1)).opts(width = 700, height = 600 , show_grid=True).relabel('Bernoulli Histogram')
def pmf(p):

    xs = np.linspace(0,2,10)

    ys = [stats.bernoulli.pmf(x, p, loc=0) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pmf, kdims=['p'])

hmap1.redim.range(p = (0.1,1)).opts(width = 700, height = 600 , show_grid=True).relabel('Bernoulli PMF')
def cdf(p):

    xs = np.linspace(0, 2, 1000)

    ys = [stats.bernoulli.cdf(x, p, loc=0) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['p'])

hmap1.redim.range(p = (0.1,1)).opts(width = 700, height = 600 , show_grid=True).relabel('Bernoulli CDF')
label = "Binomial Distribution (n = 25, p=0.6)"

n , p = 25, 0.6



measured = stats.binom.rvs(n, p, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 25, 1000)

pmf = stats.binom.pmf(x, n, p, loc=0)

cdf = stats.binom.cdf(x, n, p, loc=0)

binom = histogram(hist, x, pmf, cdf, label)



binom.opts(width = 800, height = 700 , show_grid=True)
def hist(n, p):

    data = stats.binom.rvs(n, p, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['n', 'p'])

hmap.redim.range(n= (20, 50),p = (0.1,1)).opts(width = 700, height = 600 , show_grid=True).relabel('Binomial Histogram')
def pmf(n,p):

    xs = np.arange(0,20)

    ys = [stats.binom.pmf(x,n, p, loc=0) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pmf, kdims=['n','p'])

hmap1.redim.range(n = (5,50),p = (0.1,1)).opts(width = 700, height = 600 , show_grid=True).relabel('Binom PMF')
def cdf(n,p):

    xs = np.linspace(0, 20, 1000)

    ys = [stats.binom.cdf(x,n, p) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['n','p'])

hmap1.redim.range(n = (5,50),p = (0.1,1)).opts(width = 700, height = 600 , show_grid=True).relabel('Binom CDF')
label = "Boltzman Distribution (lambda, N = 1.1, 20)"

lambda_, N = 1.1, 20



measured = boltzmann.rvs(lambda_, N, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 5, 1000)

pmf = stats.boltzmann.pmf(x, lambda_, N, loc=0)

cdf = stats.boltzmann.cdf(x, lambda_, N, loc=0)

bol = histogram(hist, x, pmf, cdf, label)



bol.opts(width = 800, height = 700 , show_grid=True)
def hist(lambda_, N):

    data = boltzmann.rvs(lambda_, N, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['lambda_', 'N'])

hmap.redim.range(lambda_ = (0.5,2), N = (20, 100)).opts(width = 700, height = 600 , show_grid=True).relabel('Bolzmann Histogram')
def pmf(lambda_, N):

    xs = np.arange(0,20)

    ys = [stats.boltzmann.pmf(x,lambda_, N, loc=0) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pmf, kdims=['lambda_', 'N'])

hmap1.redim.range(lambda_ = (0.5,2), N = (20, 100)).opts(width = 700, height = 600 , show_grid=True).relabel('Bolzmann PMF')
def cdf(lambda_, N):

    xs = np.arange(0,20)

    ys = [stats.boltzmann.cdf(x,lambda_, N, loc=0) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['lambda_', 'N'])

hmap1.redim.range(lambda_ = (0.5,2), N = (20, 100)).opts(width = 700, height = 600 , show_grid=True).relabel('Bolzmann CDF')
label = "Discrete Laplace Distribution (a = 0.8)"

a = 0.8



measured = stats.dlaplace.rvs(a, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 5, 1000)

pmf = stats.dlaplace.pmf(x, a, loc=0)

cdf = stats.dlaplace.cdf(x, a, loc=0)

dlap = histogram(hist, x, pmf, cdf, label)



dlap.opts(width = 800, height = 700 , show_grid=True)
def hist(a):

    data = stats.dlaplace.rvs(lambda_, N, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['a'])

hmap.redim.range(a = (0.1,1.5)).opts(width = 700, height = 600 , show_grid=True).relabel('Discrete Laplace Histogram')
def pmf(a):

    xs = np.arange(0,30)

    ys = [stats.dlaplace.pmf(x,a, loc=0) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pmf, kdims=['a'])

hmap1.redim.range(a = (0.1,2)).opts(width = 700, height = 600 , show_grid=True).relabel('Descrete Laplace PMF')
def cdf(a):

    xs = np.arange(0,30)

    ys = [stats.dlaplace.cdf(x,a, loc=0) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['a'])

hmap1.redim.range(a = (0.1,2)).opts(width = 700, height = 600 , show_grid=True).relabel('Descrete Laplace CDF')
label = "Geometric Distribution (p = 0.6)"

p = 0.6



measured = stats.geom.rvs(p, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 5, 1000)

pmf = stats.geom.pmf(x, p, loc=0)

cdf = stats.geom.cdf(x, p, loc=0)

geo = histogram(hist, x, pmf, cdf, label)



geo.opts(width = 800, height = 700 , show_grid=True)
def hist(p):

    data = stats.geom.rvs(p, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['p'])

hmap.redim.range(p = (0.1,1)).opts(width = 700, height = 600 , show_grid=True).relabel('Geometric Histogram')
def pmf(p):

    xs = np.arange(0,30)

    ys = [stats.geom.pmf(x,p, loc=0) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pmf, kdims=['p'])

hmap1.redim.range(p = (0.1,1)).opts(width = 700, height = 600 , show_grid=True).relabel('Geometric PMF')
def cdf(p):

    xs = np.arange(0,30)

    ys = [stats.geom.cdf(x,p, loc=0) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['p'])

hmap1.redim.range(p = (0.1,1)).opts(width = 700, height = 600 , show_grid=True).relabel('Geometric CDF')
label = "Hypergeometric Distribution (M = 50, n = 9, N = 21)"

[M, n, N] = [50, 9, 21]



measured = stats.hypergeom.rvs(M, n, N, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 10, 1000)

pmf = stats.hypergeom.pmf(x, M, n, N)

cdf = stats.hypergeom.cdf(x, M, n, N)

hyp = histogram(hist, x, pmf, cdf, label)



hyp.opts(width = 800, height = 700 , show_grid=True)
def hist(M, n, N):

    data = stats.hypergeom.rvs(M, n, N, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['M', 'n', 'N'])

hmap.redim.range(M = (30,100), n = (5, 20), N = (5, 30)).opts(width = 700, height = 600 , show_grid=True).relabel('Hypergeometric Histogram')
def pmf(M, n, N):

    xs = np.arange(0,10)

    ys = [stats.hypergeom.pmf(x, M, n, N) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pmf, kdims=['M', 'n', 'N'])

hmap1.redim.range(M = (30,100), n = (5, 20), N = (5, 30)).opts(width = 700, height = 600 , show_grid=True).relabel('Hypergeometric PMF')
def cdf(M, n, N):

    xs = np.arange(0,5)

    ys = [stats.hypergeom.cdf(x, M, n, N) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['M', 'n', 'N'])

hmap1.redim.range(M = (30,100), n = (5, 20), N = (5, 30)).opts(width = 700, height = 600 , show_grid=True).relabel('Hypergeometric CDF')
label = "Log Series Distribution (p = 0.6)"

p = 0.6



measured = stats.logser.rvs(p, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 10, 1000)

pmf = stats.logser.pmf(x, p)

cdf = stats.logser.cdf(x, p)

log = histogram(hist, x, pmf, cdf, label)



log.opts(width = 800, height = 700 , show_grid=True)
def hist(p):

    data = stats.logser.rvs(p, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['p'])

hmap.redim.range(p = (0.4,1)).opts(width = 700, height = 600 , show_grid=True).relabel('Log-Series Histogram')
def pmf(p):

    xs = np.arange(0,10)

    ys = [stats.logser.pmf(x,p) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pmf, kdims=['p'])

hmap1.redim.range(p = (0.1,1)).opts(width = 700, height = 600 , show_grid=True).relabel('Log-series PMF')
def cdf(p):

    xs = np.arange(0,10)

    ys = [stats.logser.cdf(x,p) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['p'])

hmap1.redim.range(p = (0.1,1)).opts(width = 700, height = 600 , show_grid=True).relabel('Log-series CDF')
label = "Negative Binomial Distribution (n = 20, p = 0.6)"

n, p = 20, 0.6



measured = stats.nbinom.rvs(n, p, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 30, 1000)

pmf = stats.nbinom.pmf(x,n, p)

cdf = stats.nbinom.cdf(x,n, p)

nb = histogram(hist, x, pmf, cdf, label)



nb.opts(width = 800, height = 700 , show_grid=True)
def hist(n, p):

    data = stats.nbinom.rvs(n, p, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['n','p'])

hmap.redim.range(n = (5, 30), p = (0.1,1)).opts(width = 700, height = 600 , show_grid=True).relabel('Negative Binomial Histogram')
def pmf(n, p):

    xs = np.arange(0,100)

    ys = [stats.nbinom.pmf(x,n,p) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pmf, kdims=['n','p'])

hmap1.redim.range(n = (5, 15), p = (0.1,1)).opts(width = 700, height = 600 , show_grid=True).relabel('Negative Binomial PMF')
def cdf(n, p):

    xs = np.arange(0,100)

    ys = [stats.nbinom.cdf(x,n,p) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pmf, kdims=['n','p'])

hmap1.redim.range(n = (5, 15), p = (0.1,1)).opts(width = 700, height = 600 , show_grid=True).relabel('Negative Binomial CDF')
label = "Planck Discrete Exponential Distribution (lam = 0.6)"

lam = 0.6



measured = stats.planck.rvs(lam, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 10, 1000)

pmf = stats.planck.pmf(x,lam)

cdf = stats.planck.cdf(x,lam)

planck = histogram(hist, x, pmf, cdf, label)



planck.opts(width = 800, height = 700 , show_grid=True)
def hist(lam):

    data = stats.planck.rvs(lam, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['lam'])

hmap.redim.range(lam = (0.1, 1)).opts(width = 700, height = 600 , show_grid=True).relabel('Planck Discrete Exponential Histogram')
def pmf(lam):

    xs = np.arange(0,100)

    ys = [stats.planck.pmf(x,lam) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pmf, kdims=['lam'])

hmap1.redim.range(lam = (0.1, 1)).opts(width = 700, height = 600 , show_grid=True).relabel('Planck Discrete Exponential PMF')
def cdf(lam):

    xs = np.arange(0,100)

    ys = [stats.planck.cdf(x,lam) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['lam'])

hmap1.redim.range(lam = (0.1, 1)).opts(width = 700, height = 600 , show_grid=True).relabel('Planck Discrete Exponential CDF')
label = "Poisson Distribution (mu = 0.6)"

mu = 0.6



measured = stats.poisson.rvs(mu, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 5, 1000)

pmf = stats.poisson.pmf(x,mu)

cdf = stats.poisson.cdf(x, mu)

pois = histogram(hist, x, pmf, cdf, label)



pois.opts(width = 800, height = 700 , show_grid=True)
def hist(mu):

    data = stats.poisson.rvs(mu, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['mu'])

hmap.redim.range(mu = (1, 10)).opts(width = 700, height = 600 , show_grid=True).relabel('Poisson Histogram')
def pmf(mu):

    xs = np.arange(0,7)

    ys = [stats.poisson.pmf(x,mu) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pmf, kdims=['mu'])

hmap1.redim.range(mu = (0.1, 10)).opts(width = 700, height = 600 , show_grid=True).relabel('Poisson PMF')
def cdf(mu):

    xs = np.arange(0,7)

    ys = [stats.poisson.cdf(x,mu) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['mu'])

hmap1.redim.range(mu = (0.1, 10)).opts(width = 700, height = 600 , show_grid=True).relabel('Poisson CDF')
label = "Uniform Distribution (low, high = 5, 10)"

low, high = 5, 10



measured = stats.randint.rvs(low, high, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 10, 1000)

pmf = stats.randint.pmf(x, low, high)

cdf = stats.randint.cdf(x, low, high)

uni = histogram(hist, x, pmf, cdf, label)



uni.opts(width = 800, height = 700 , show_grid=True)
def hist(low, high):

    data = stats.randint.rvs(low, high, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['low', 'high'])

hmap.redim.range(low = (1, 10),high = (11, 20)).opts(width = 700, height = 600 , show_grid=True).relabel('Uniform Histogram')
def pmf(low, high):

    xs = np.arange(0,100)

    ys = [stats.randint.pmf(x, low, high) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pmf, kdims=['low', 'high'])

hmap1.redim.range(low = (1, 49),high = (50, 100)).opts(width = 700, height = 600 , show_grid=True).relabel('Uniform PMF')
def cdf(low, high):

    xs = np.arange(0,100)

    ys = [stats.randint.cdf(x, low, high) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['low', 'high'])

hmap1.redim.range(low = (1, 49),high = (50, 100)).opts(width = 700, height = 600 , show_grid=True).relabel('Uniform CDF')
label = "Skellam Distribution (mu1, mu2 = 15, 8)"

mu1, mu2 = 15, 8



measured = stats.skellam.rvs(mu1, mu2, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 10, 1000)

pmf = stats.skellam.pmf(x, mu1, mu2)

cdf = stats.skellam.cdf(x, mu1, mu2)

uni = histogram(hist, x, pmf, cdf, label)



uni.opts(width = 800, height = 700 , show_grid=True)
def hist(mu1, mu2):

    data = stats.randint.rvs(low, high, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['mu1', 'mu2'])

hmap.redim.range(mu1 = (1, 20),mu2 = (1, 20)).opts(width = 700, height = 600 , show_grid=True).relabel('Skellam Histogram')
def pmf(mu1, mu2):

    xs = np.arange(0,10)

    ys = [stats.skellam.pmf(x, mu1, mu2) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pmf, kdims=['mu1', 'mu2'])

hmap1.redim.range(mu1 = (1, 20),mu2 = (1, 20)).opts(width = 700, height = 600 , show_grid=True).relabel('Skellam PMF')
def cdf(mu1, mu2):

    xs = np.arange(0,10)

    ys = [stats.skellam.cdf(x, mu1, mu2) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['mu1', 'mu2'])

hmap1.redim.range(mu1 = (1, 20),mu2 = (1, 20)).opts(width = 700, height = 600 , show_grid=True).relabel('Skellam CDF')
label = "Zipf Distribution (a = 4)"

a = 4



measured = stats.zipf.rvs(a, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 4, 1000)

pmf = stats.zipf.pmf(x,a)

cdf = stats.zipf.cdf(x,a)

zipf = histogram(hist, x, pmf, cdf, label)



zipf.opts(width = 800, height = 700 , show_grid=True)
def hist(a):

    data = stats.zipf.rvs(a, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['a'])

hmap.redim.range(a = (2, 20)).opts(width = 700, height = 600 , show_grid=True).relabel('Zipf Histogram')
def pmf(a):

    xs = np.arange(0,10)

    ys = [stats.zipf.pmf(x, a) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pmf, kdims=['a'])

hmap1.redim.range(a = (2, 20)).opts(width = 700, height = 600 , show_grid=True).relabel('Zipf PMF')
def cdf(a):

    xs = np.arange(0,100)

    ys = [stats.zipf.cdf(x, a) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['a'])

hmap1.redim.range(a = (2, 20)).opts(width = 700, height = 600 , show_grid=True).relabel('Zipf CDF')