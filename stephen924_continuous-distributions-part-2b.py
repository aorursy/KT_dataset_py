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
label = "Crystalball Distribution (beta, m = 2, 3)"

beta, m = 2, 3



measured = stats.crystalball.rvs(beta, m, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(-4, 4, 1000)

pdf = stats.crystalball.pdf(x, beta, m, loc=0, scale=1)

cdf = stats.crystalball.cdf(x, beta, m, loc=0, scale=1)

cry = histogram(hist, x, pdf, cdf, label)

cry.opts(width = 800, height = 700 , show_grid=True)
def hist(beta, mu):

    data = stats.crystalball.rvs(beta, mu, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['beta', 'mu'])

hmap.redim.range(beta = (2,8), mu = (4,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Crystalball Distribution Histogram')
def pdf(beta, mu):

    xs = np.linspace(-10, 4, 1000)

    ys = [stats.crystalball.pdf(x, beta, mu, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['beta', 'mu'])

hmap1.redim.range(beta = (2,8), mu = (4,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Crystalball Distribution PDF')
def cdf(beta, mu):

    xs = np.linspace(-10, 4, 1000)

    ys = [stats.crystalball.cdf(x, beta, mu, loc=0, scale=1) for x in xs]

    return hv.Curve((xs,ys))



hmap2 = hv.DynamicMap(cdf, kdims=['beta', 'mu'])

hmap2.redim.range(beta = (2,8), mu = (4,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Crystalball Distribution CDF')
label = "Double Gamma Continuous Distribution (a = 2)"

a = 2



measured = stats.dgamma.rvs(a, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(-10, 10, 1000)

pdf = stats.dgamma.pdf(x, a, loc=0, scale=1)

cdf = stats.dgamma.cdf(x, a, loc=0, scale=1)

dg = histogram(hist, x, pdf, cdf, label)

dg.opts(width = 800, height = 700 , show_grid=True)
def hist(a):

    data = stats.dgamma.rvs(a, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['a'])

hmap.redim.range(a = (2,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Double Gamma Continuous Distribution Histogram')
def pdf(a):

    xs = np.linspace(-10, 10, 1000)

    ys = [stats.dgamma.pdf(x, a, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['a'])

hmap1.redim.range(a = (2,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Double Gamma Continuous Distribution PDF')
def cdf(a):

    xs = np.linspace(-10, 10, 1000)

    ys = [stats.dgamma.cdf(x, a, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap2 = hv.DynamicMap(cdf, kdims=['a'])

hmap2.redim.range(a = (2,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Double Gamma Continuous Distribution CDF')
label = "Double Weibull Continuous Distribution (c = 2)"

c = 2



measured = stats.dweibull.rvs(c, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(-4, 4, 1000)

pdf = stats.dweibull.pdf(x, c, loc=0, scale=1)

cdf = stats.dweibull.cdf(x, c, loc=0, scale=1)

dw = histogram(hist, x, pdf, cdf, label)

dw.opts(width = 800, height = 700 , show_grid=True)
def hist(c):

    data = stats.dweibull.rvs(c, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['c'])

hmap.redim.range(c = (2,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Double Weibull Continuous Distribution Histogram')
def pdf(c):

    xs = np.linspace(-4, 4, 1000)

    ys = [stats.dweibull.pdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['c'])

hmap1.redim.range(c = (2,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Double Weibull Continuous Distribution PDF')
def cdf(a):

    xs = np.linspace(-4, 4, 1000)

    ys = [stats.dweibull.cdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['c'])

hmap1.redim.range(c = (2,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Double Weibull Continuous Distribution CDF')
label = "Erlang Distribution (a = 2)"

a = 2



measured = stats.erlang.rvs(a, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(-1, 8, 1000)

pdf = stats.erlang.pdf(x, a, loc=0, scale=1)

cdf = stats.erlang.cdf(x, a, loc=0, scale=1)

er = histogram(hist, x, pdf, cdf, label)

er.opts(width = 800, height = 700 , show_grid=True)
def hist(a):

    data = stats.erlang.rvs(a, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['a'])

hmap.redim.range(a = (2,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Erlang Distribution Histogram')
def pdf(a):

    xs = np.linspace(-1, 8, 1000)

    ys = [stats.erlang.pdf(x, a, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['a'])

hmap1.redim.range(a = (1,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Erlang Distribution PDF')
def cdf(a):

    xs = np.linspace(-1, 8, 1000)

    ys = [stats.erlang.cdf(x, a, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['a'])

hmap1.redim.range(a = (1,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Erlang Distribution CDF')
label = "Exponentially modified Gaussian distribution (K = 0.1)"

K = 0.1



measured = stats.exponnorm.rvs(K, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(-4, 4, 1000)

pdf = stats.exponnorm.pdf(x, K, loc=0, scale=1)

cdf = stats.exponnorm.cdf(x, K, loc=0, scale=1)

ex = histogram(hist, x, pdf, cdf, label)

ex.opts(width = 800, height = 700 , show_grid=True)
def hist(K):

    data = stats.exponnorm.rvs(K, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['K'])

hmap.redim.range(K = (0.1,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Exponentially modified Gaussian Distribution Histogram')
def pdf(K):

    xs = np.linspace(-4, 4, 1000)

    ys = [stats.exponnorm.pdf(x, K, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['K'])

hmap1.redim.range(K = (0.1,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Exponentially modified Gaussian Distribution PDF')
def cdf(K):

    xs = np.linspace(-4, 4, 1000)

    ys = [stats.exponnorm.cdf(x, K, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['K'])

hmap1.redim.range(K = (0.1,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Exponentially modified Gaussian Distribution CDF')
label = "Exponentiated Weibull distribution (a, c = 3, 2)"

a, c = 3, 2



measured = stats.exponweib.rvs(a, c, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 4, 1000)

pdf = stats.exponweib.pdf(x, a, c, loc=0, scale=1)

cdf = stats.exponweib.cdf(x, a, c, loc=0, scale=1)

exw = histogram(hist, x, pdf, cdf, label)

exw.opts(width = 800, height = 700 , show_grid=True)
def hist(a, c):

    data = stats.exponweib.rvs(a, c, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['a', 'c'])

hmap.redim.range(a = (1,10), c=(1,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Exponentiated Weibull distribution Histogram')
def pdf(a, c):

    xs = np.linspace(0, 4, 1000)

    ys = [stats.exponweib.pdf(x, a, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['a', 'c'])

hmap1.redim.range(a = (1,10), c=(1,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Exponentiated Weibull distribution PDF')
def cdf(a, c):

    xs = np.linspace(0, 4, 1000)

    ys = [stats.exponweib.cdf(x, a, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['a', 'c'])

hmap1.redim.range(a = (1,10), c=(1,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Exponentiated Weibull distribution CDF')
label = "Exponential Power Distribution (b = 3)"

b = 3



measured = stats.exponpow.rvs(b, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 2, 1000)

pdf = stats.exponpow.pdf(x, b, loc=0, scale=1)

cdf = stats.exponpow.cdf(x, b, loc=0, scale=1)

exp = histogram(hist, x, pdf, cdf, label)

exp.opts(width = 800, height = 700 , show_grid=True)
def hist(b):

    data = stats.exponpow.rvs(b, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['b'])

hmap.redim.range(b = (1,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Exponential Power Distribution Histogram')
def pdf(b):

    xs = np.linspace(0, 2, 1000)

    ys = [stats.exponpow.pdf(x, b, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['b'])

hmap1.redim.range(b = (1,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Exponential Power Distribution PDF')
def cdf(b):

    xs = np.linspace(0, 2, 1000)

    ys = [stats.exponpow.cdf(x, b, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['b'])

hmap1.redim.range(b = (1,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Exponential Power Distribution CDF')
label = "Fatigue Life Distribution (c = 10)"

c = 10



measured = stats.fatiguelife.rvs(c, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 500, 1000)

pdf = stats.fatiguelife.pdf(x, c, loc=0, scale=1)

cdf = stats.fatiguelife.cdf(x, c, loc=0, scale=1)

fat = histogram(hist, x, pdf, cdf, label)

fat.opts(width = 800, height = 700 , show_grid=True)
def hist(c):

    data = stats.fatiguelife.rvs(c, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['c'])

hmap.redim.range(c = (20,100)).opts(width = 700, height = 600 , show_grid=True).relabel('Fatigue Life Distribution Histogram')
def pdf(c):

    xs = np.linspace(0, 500, 1000)

    ys = [stats.fatiguelife.pdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['c'])

hmap1.redim.range(c = (20,100)).opts(width = 700, height = 600 , show_grid=True).relabel('Fatigue Life Distribution PDF')
def cdf(c):

    xs = np.linspace(0, 500, 1000)

    ys = [stats.fatiguelife.cdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['c'])

hmap1.redim.range(c = (20,100)).opts(width = 700, height = 600 , show_grid=True).relabel('Fatigue Life Distribution CDF')
label = "Log-logistic distribution (c = 4)"

c = 4



measured = stats.fisk.rvs(c, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 4, 1000)

pdf = stats.fisk.pdf(x, c, loc=0, scale=1)

cdf = stats.fisk.cdf(x, c, loc=0, scale=1)

fat = histogram(hist, x, pdf, cdf, label)

fat.opts(width = 800, height = 700 , show_grid=True)
def hist(c):

    data = stats.fisk.rvs(c, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['c'])

hmap.redim.range(c = (2,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Log-logistic distribution Histogram')
def pdf(c):

    xs = np.linspace(0, 4, 1000)

    ys = [stats.fisk.pdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['c'])

hmap1.redim.range(c = (1,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Log-logistic distribution PDF')
def cdf(c):

    xs = np.linspace(0, 4, 1000)

    ys = [stats.fisk.cdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['c'])

hmap1.redim.range(c = (1,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Log-logistic distribution CDF')
label = "Fold Cauchy Distribution (c = 4)"

c = 4



measured = stats.foldcauchy.rvs(c, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 400, 1000)

pdf = stats.foldcauchy.pdf(x, c, loc=0, scale=1)

cdf = stats.foldcauchy.cdf(x, c, loc=0, scale=1)

fat = histogram(hist, x, pdf, cdf, label)

fat.opts(width = 800, height = 700 , show_grid=True)
def hist(c):

    data = stats.foldcauchy.rvs(c, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['c'])

hmap.redim.range(c = (2,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Fold Cauchy Distribution Histogram')
def pdf(c):

    xs = np.linspace(-1, 6, 1000)

    ys = [stats.foldcauchy.pdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['c'])

hmap1.redim.range(c = (2,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Fold Cauchy Distribution PDF')
def cdf(c):

    xs = np.linspace(-1, 6, 1000)

    ys = [stats.foldcauchy.cdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['c'])

hmap1.redim.range(c = (2,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Fold Cauchy Distribution CDF')
label = "Fold Normal Distribution (c = 2)"

c = 2



measured = stats.foldnorm.rvs(c, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(-1, 5, 1000)

pdf = stats.foldnorm.pdf(x, c, loc=0, scale=1)

cdf = stats.foldnorm.cdf(x, c, loc=0, scale=1)

fol = histogram(hist, x, pdf, cdf, label)

fol.opts(width = 800, height = 700 , show_grid=True)
def hist(c):

    data = stats.foldnorm.rvs(c, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['c'])

hmap.redim.range(c = (1,7)).opts(width = 700, height = 600 , show_grid=True).relabel('Fold Normal Distribution Histogram')
def pdf(c):

    xs = np.linspace(-1, 6, 1000)

    ys = [stats.foldnorm.pdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['c'])

hmap1.redim.range(c = (1,7)).opts(width = 700, height = 600 , show_grid=True).relabel('Fold Normal Distribution PDF')
def cdf(c):

    xs = np.linspace(-1, 6, 1000)

    ys = [stats.foldnorm.cdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['c'])

hmap1.redim.range(c = (1,7)).opts(width = 700, height = 600 , show_grid=True).relabel('Fold Normal Distribution CDF')
label = "Fréchet Right Distribution (c = 2)"

c = 2



measured = stats.frechet_r.rvs(c, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(-1, 4, 1000)

pdf = stats.frechet_r.pdf(x, c, loc=0, scale=1)

cdf = stats.frechet_r.cdf(x, c, loc=0, scale=1)

fretr = histogram(hist, x, pdf, cdf, label)

fretr.opts(width = 800, height = 700 , show_grid=True)
label = "Fréchet Left Distribution (c = 2)"

c = 2



measured = stats.frechet_l.rvs(c, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(-1, 4, 1000)

pdf = stats.frechet_l.pdf(x, c, loc=0, scale=1)

cdf = stats.frechet_l.cdf(x, c, loc=0, scale=1)

fretl = histogram(hist, x, pdf, cdf, label)

fretl.opts(width = 800, height = 700 , show_grid=True)
def hist(c):

    data = stats.frechet_r.rvs(c, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['c'])

hmap.redim.range(c = (1,7)).opts(width = 700, height = 600 , show_grid=True).relabel('Fréchet Right Distribution Histogram')
def hist(c):

    data = stats.frechet_l.rvs(c, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['c'])

hmap.redim.range(c = (1,7)).opts(width = 700, height = 600 , show_grid=True).relabel('Fréchet Left Distribution Histogram')
def pdf(c):

    xs = np.linspace(-1, 6, 1000)

    ys = [stats.frechet_r.pdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['c'])

hmap1.redim.range(c = (1,7)).opts(width = 700, height = 600 , show_grid=True).relabel('Fréchet Right Distribution PDF')
def pdf(c):

    xs = np.linspace(-6, 1, 1000)

    ys = [stats.frechet_l.pdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['c'])

hmap1.redim.range(c = (1,7)).opts(width = 700, height = 600 , show_grid=True).relabel('Fréchet Left Distribution PDF')
def cdf(c):

    xs = np.linspace(-1, 6, 1000)

    ys = [stats.frechet_r.cdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['c'])

hmap1.redim.range(c = (1,7)).opts(width = 700, height = 600 , show_grid=True).relabel('Fréchet Right Distribution CDF')
def cdf(c):

    xs = np.linspace(-6, 1, 1000)

    ys = [stats.frechet_l.cdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['c'])

hmap1.redim.range(c = (1,7)).opts(width = 700, height = 600 , show_grid=True).relabel('Fréchet Left Distribution CDF')
label = "Generalized Logistic Distribution (c = 0.5)"

c = 0.5



measured = stats.genlogistic.rvs(c, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(-10, 4, 1000)

pdf = stats.genlogistic.pdf(x, c, loc=0, scale=1)

cdf = stats.genlogistic.cdf(x, c, loc=0, scale=1)

log = histogram(hist, x, pdf, cdf, label)

log.opts(width = 800, height = 700 , show_grid=True)
def hist(c):

    data = stats.genlogistic.rvs(c, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['c'])

hmap.redim.range(c = (0.1,7)).opts(width = 700, height = 600 , show_grid=True).relabel('Generalized Logistic Distribution Histogram')
def pdf(c):

    xs = np.linspace(-6, 6, 1000)

    ys = [stats.genlogistic.pdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['c'])

hmap1.redim.range(c = (1,7)).opts(width = 700, height = 600 , show_grid=True).relabel('Generalized Logistic Distribution PDF')
def cdf(c):

    xs = np.linspace(-6, 6, 1000)

    ys = [stats.genlogistic.cdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['c'])

hmap1.redim.range(c = (1,7)).opts(width = 700, height = 600 , show_grid=True).relabel('Generalized Logistic Distribution CDF')
label = "Generalized Normal Distribution (beta = 1.5)"

beta = 1.5



measured = stats.gennorm.rvs(beta, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(-4, 4, 1000)

pdf = stats.gennorm.pdf(x, beta, loc=0, scale=1)

cdf = stats.gennorm.cdf(x, beta, loc=0, scale=1)

log = histogram(hist, x, pdf, cdf, label)

log.opts(width = 800, height = 700 , show_grid=True)
def hist(beta):

    data = stats.gennorm.rvs(beta, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['beta'])

hmap.redim.range(beta = (0.5,2)).opts(width = 700, height = 600 , show_grid=True).relabel('Generalized Normal Distribution Histogram')
def pdf(beta):

    xs = np.linspace(-7, 6, 1000)

    ys = [stats.genlogistic.pdf(x, beta, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['beta'])

hmap1.redim.range(beta = (0.5,7)).opts(width = 700, height = 600 , show_grid=True).relabel('Generalized Normal Distribution PDF')
def cdf(beta):

    xs = np.linspace(-7, 6, 1000)

    ys = [stats.genlogistic.cdf(x, beta, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['beta'])

hmap1.redim.range(beta = (0.5,7)).opts(width = 700, height = 600 , show_grid=True).relabel('Generalized Normal Distribution CDF')
label = "Generalized Pareto Distribution (c = 0.1)"

c = 0.1



measured = stats.genpareto.rvs(c, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(-1, 8, 1000)

pdf = stats.genpareto.pdf(x, c, loc=0, scale=1)

cdf = stats.genpareto.cdf(x, c, loc=0, scale=1)

par = histogram(hist, x, pdf, cdf, label)

par.opts(width = 800, height = 700 , show_grid=True)
def hist(c):

    data = stats.genpareto.rvs(c, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['c'])

hmap.redim.range(c = (0.1,0.8)).opts(width = 700, height = 600 , show_grid=True).relabel('Generalized Pareto Distribution Histogram')
def pdf(c):

    xs = np.linspace(-1, 6, 1000)

    ys = [stats.genpareto.pdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['c'])

hmap1.redim.range(c = (0.1,0.8)).opts(width = 700, height = 600 , show_grid=True).relabel('Generalized Pareto Distribution PDF')
def cdf(c):

    xs = np.linspace(-1, 6, 1000)

    ys = [stats.genpareto.cdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['c'])

hmap1.redim.range(c = (0.1,0.8)).opts(width = 700, height = 600 , show_grid=True).relabel('Generalized Pareto Distribution CDF')
label = "Generalized Exponential Distribution (a, b, c = 9, 16, 3)"

a, b, c = 9, 16, 3



measured = stats.genexpon.rvs(a, b, c, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 0.5, 1000)

pdf = stats.genexpon.pdf(x, a, b, c, loc=0, scale=1)

cdf = stats.genexpon.cdf(x, a, b, c, loc=0, scale=1)

par = histogram(hist, x, pdf, cdf, label)

par.opts(width = 800, height = 700 , show_grid=True)
def hist(a,b, c):

    data = stats.genexpon.rvs(a, b, c, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['a','b','c'])

hmap.redim.range(a = (1, 20), b=(1,20) ,c = (1,20)).opts(width = 700, height = 600 , show_grid=True).relabel('Generalized Exponential Distribution Histogram')
def pdf(a,b,c):

    xs = np.linspace(0, 4, 1000)

    ys = [stats.genexpon.pdf(x, a, b, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['a','b','c'])

hmap1.redim.range(a = (1, 20), b=(1,20) ,c = (1,20)).opts(width = 700, height = 600 , show_grid=True).relabel('Generalized Exponential Distribution PDF')
def cdf(a,b,c):

    xs = np.linspace(0, 4, 1000)

    ys = [stats.genexpon.cdf(x, a, b, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['a','b','c'])

hmap1.redim.range(a = (1, 20), b=(1,20) ,c = (1,20)).opts(width = 700, height = 600 , show_grid=True).relabel('Generalized Exponential Distribution CDF')
label = "Generalized Extreme Value Distribution (c = 0)"

c = 0



measured = stats.genextreme.rvs(c, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(-2, 10, 1000)

pdf = stats.genextreme.pdf(x, c, loc=0, scale=1)

cdf = stats.genextreme.cdf(x, c, loc=0, scale=1)

ev = histogram(hist, x, pdf, cdf, label)

ev.opts(width = 800, height = 700 , show_grid=True)
def hist(c):

    data = stats.genextreme.rvs(c, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['c'])

hmap.redim.range(c = (0, 2)).opts(width = 700, height = 600 , show_grid=True).relabel('Generalized Extreme Value Distribution Histogram')
def pdf(c):

    xs = np.linspace(-2, 4, 1000)

    ys = [stats.genextreme.pdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['c'])

hmap1.redim.range(c = (0,2)).opts(width = 700, height = 600 , show_grid=True).relabel('Generalized Extreme Value Distribution PDF')
def cdf(c):

    xs = np.linspace(-2, 4, 1000)

    ys = [stats.genextreme.cdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['c'])

hmap1.redim.range(c = (0,2)).opts(width = 700, height = 600 , show_grid=True).relabel('Generalized Extreme Value Distribution CDF')
label = "Gauss Hypergeometric Distribution (a, b, c, z = 14, 3, 2, 5)"

a, b, c, z = 14, 3, 2, 5



measured = stats.gausshyper.rvs(a, b, c, z, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 1, 1000)

pdf = stats.gausshyper.pdf(x, a, b, c, z, loc=0, scale=1)

cdf = stats.gausshyper.cdf(x, a, b, c, z, loc=0, scale=1)

gh = histogram(hist, x, pdf, cdf, label)

gh.opts(width = 800, height = 700 , show_grid=True)
def hist(a,b,c,z):

    data = stats.gausshyper.rvs(a, b, c, z, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['a','b','c','z'])

hmap.redim.range(a = (1, 20), b=(1,20) ,c = (1,20), z =(1,20)).opts(width = 700, height = 600 , show_grid=True).relabel('Gauss Hypergeometric Distribution Histogram')
def pdf(a,b,c, z):

    xs = np.linspace(0, 2, 1000)

    ys = [stats.gausshyper.pdf(x, a, b, c, z, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['a','b','c', 'z'])

hmap1.redim.range(a = (1, 20), b=(1,20) ,c = (1,20), z=(1,20)).opts(width = 700, height = 600 , show_grid=True).relabel('Gauss Hypergeometric Distribution PDF')
def cdf(a,b,c, z):

    xs = np.linspace(0, 2, 1000)

    ys = [stats.gausshyper.cdf(x, a, b, c, z, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['a','b','c', 'z'])

hmap1.redim.range(a = (1, 20), b=(1,20) ,c = (1,20), z=(1,20)).opts(width = 700, height = 600 , show_grid=True).relabel('Gauss Hypergeometric Distribution CDF')
label = "Generalized Gamma Distribution (a, c = 4, 3)"

a, c = 4, 3



measured = stats.gengamma.rvs(a, c, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 3, 1000)

pdf = stats.gengamma.pdf(x, a, c, loc=0, scale=1)

cdf = stats.gengamma.cdf(x, a, c, loc=0, scale=1)

gg = histogram(hist, x, pdf, cdf, label)

gg.opts(width = 800, height = 700 , show_grid=True)
def hist(a,c):

    data = stats.gengamma.rvs(a, c, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['a','c'])

hmap.redim.range(a = (1, 20),c = (1,20)).opts(width = 700, height = 600 , show_grid=True).relabel('Generalized Gamma Distribution Histogram')
def pdf(a,c):

    xs = np.linspace(0, 2, 1000)

    ys = [stats.gengamma.pdf(x, a, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['a','c'])

hmap1.redim.range(a = (1, 5) ,c = (1,20)).opts(width = 700, height = 600 , show_grid=True).relabel('Generalized Gamma Distribution PDF')
def cdf(a,c):

    xs = np.linspace(0, 2, 1000)

    ys = [stats.gengamma.cdf(x, a, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['a','c'])

hmap1.redim.range(a = (1, 5) ,c = (1,20)).opts(width = 700, height = 600 , show_grid=True).relabel('Generalized Gamma Distribution CDF')
label = "Generalized Half-logistic Distribution (c=1)"

c = 1



measured = stats.genhalflogistic.rvs(c, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(-0.5, 1.5, 1000)

pdf = stats.genhalflogistic.pdf(x, c, loc=0, scale=1)

cdf = stats.genhalflogistic.cdf(x, c, loc=0, scale=1)

gg = histogram(hist, x, pdf, cdf, label)

gg.opts(width = 800, height = 700 , show_grid=True)
def hist(c):

    data = stats.genhalflogistic.rvs(c, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['c'])

hmap.redim.range(c = (1, 10)).opts(width = 700, height = 600 , show_grid=True).relabel('Generalized Half-logistic Distribution Histogram')
def pdf(c):

    xs = np.linspace(-0.5, 1.5, 1000)

    ys = [stats.genhalflogistic.pdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['c'])

hmap1.redim.range(c = (1,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Generalized Half-logistic Distribution PDF')
def cdf(c):

    xs = np.linspace(-0.5, 1.5, 1000)

    ys = [stats.genhalflogistic.cdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['c'])

hmap1.redim.range(c = (1,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Generalized Half-logistic Distribution CDF')