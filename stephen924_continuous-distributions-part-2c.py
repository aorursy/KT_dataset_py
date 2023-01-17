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
label = "Gibrat Distribution"



measured = stats.gilbrat.rvs(size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 10, 1000)

pdf = stats.gilbrat.pdf(x, loc=0, scale=1)

cdf = stats.gilbrat.cdf(x, loc=0, scale=1)

cry = histogram(hist, x, pdf, cdf, label)

cry.opts(width = 800, height = 700 , show_grid=True)
label = "Gompertz Distribution (c = 1)"

c = 1



measured = stats.gompertz.rvs(c, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 2, 1000)

pdf = stats.gompertz.pdf(x, c, loc=0, scale=1)

cdf = stats.gompertz.cdf(x, c, loc=0, scale=1)

dg = histogram(hist, x, pdf, cdf, label)

dg.opts(width = 800, height = 700 , show_grid=True)
def hist(c):

    data = stats.gompertz.rvs(c, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['c'])

hmap.redim.range(c = (0.1,5)).opts(width = 700, height = 600 , show_grid=True).relabel('Gompertz Distribution Histogram')
def pdf(c):

    xs = np.linspace(0, 5, 1000)

    ys = [stats.gompertz.pdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['c'])

hmap1.redim.range(c = (0.1,5)).opts(width = 700, height = 600 , show_grid=True).relabel('Gompertz Distribution Distribution PDF')
def cdf(c):

    xs = np.linspace(0, 5, 1000)

    ys = [stats.gompertz.cdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['c'])

hmap1.redim.range(c = (0.1,5)).opts(width = 700, height = 600 , show_grid=True).relabel('Gompertz Distribution Distribution CDF')
label = "Half-Cauchy Distribution"



measured = stats.halfcauchy.rvs(size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 1, 1000)

pdf = stats.halfcauchy.pdf(x, loc=0, scale=1) 

cdf = stats.halfcauchy.cdf(x, loc=0, scale=1)

dw = histogram(hist, x, pdf, cdf, label)

dw.opts(width = 800, height = 700 , show_grid=True)
label = "Half-logistic Distribution"



measured = stats.halflogistic.rvs(size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 8, 1000)

pdf = stats.halflogistic.pdf(x, loc=0, scale=1)

cdf = stats.halflogistic.cdf(x, loc=0, scale=1)

hl = histogram(hist, x, pdf, cdf, label)

hl.opts(width = 800, height = 700 , show_grid=True)
label = "Half-Normal Distribution"



measured = stats.halfnorm.rvs(size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 4, 1000)

pdf = stats.halfnorm.pdf(x,loc=0, scale=1)

cdf = stats.halfnorm.cdf(x,loc=0, scale=1)

hn = histogram(hist, x, pdf, cdf, label)

hn.opts(width = 800, height = 700 , show_grid=True)
label = "Exponential Power Distribution (beta = 1)"

beta = 1



measured = stats.halfgennorm.rvs(beta, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 15, 1000)

pdf = stats.halfgennorm.pdf(x, beta, loc=0, scale=1)

cdf = stats.halfgennorm.cdf(x, beta, loc=0, scale=1)

exw = histogram(hist, x, pdf, cdf, label)

exw.opts(width = 800, height = 700 , show_grid=True)
def hist(beta):

    data = stats.halfgennorm.rvs(beta, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['beta'])

hmap.redim.range(beta=(0.5,5)).opts(width = 700, height = 600 , show_grid=True).relabel('Exponential Power Distribution Histogram')
def pdf(beta):

    xs = np.linspace(0, 4, 1000)

    ys = [stats.halfgennorm.pdf(x, beta, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['beta'])

hmap1.redim.range(beta=(0.5,5)).opts(width = 700, height = 600 , show_grid=True).relabel('Exponential Power Distribution distribution PDF')
def cdf(beta):

    xs = np.linspace(0, 4, 1000)

    ys = [stats.halfgennorm.cdf(x, beta, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['beta'])

hmap1.redim.range(beta=(0.5,5)).opts(width = 700, height = 600 , show_grid=True).relabel('Exponential Power Distribution distribution CDF')
label = "Hyperbolic Secant Distribution"



measured = stats.hypsecant.rvs(size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(-5, 5, 1000)

pdf = stats.hypsecant.pdf(x, loc=0, scale=1)

cdf = stats.hypsecant.cdf(x, loc=0, scale=1)

hs = histogram(hist, x, pdf, cdf, label)

hs.opts(width = 800, height = 700 , show_grid=True)
label = "Inverse-Gamma Distribution (a = 4)"

a = 4



measured = stats.invgamma.rvs(a, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 2, 1000)

pdf = stats.invgamma.pdf(x, a, loc=0, scale=1)

cdf = stats.invgamma.cdf(x, a, loc=0, scale=1)

ig = histogram(hist, x, pdf, cdf, label)

ig.opts(width = 800, height = 700 , show_grid=True)
def hist(a):

    data = stats.invgamma.rvs(a, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['a'])

hmap.redim.range(a = (2,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Inverse-Gamma Distribution Histogram')
def pdf(a):

    xs = np.linspace(0, 5, 1000)

    ys = [stats.invgamma.pdf(x, a, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['a'])

hmap1.redim.range(a = (2,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Inverse-Gamma Distribution PDF')
def cdf(a):

    xs = np.linspace(0, 5, 1000)

    ys = [stats.invgamma.cdf(x, a, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['a'])

hmap1.redim.range(a = (2,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Inverse-Gamma Distribution CDF')
label = "Inverse Gaussian Distribution (mu = 0.1)"

mu = 0.1



measured = stats.invgauss.rvs(mu, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 0.4, 1000)

pdf = stats.invgauss.pdf(x, mu, loc=0, scale=1)

cdf = stats.invgauss.cdf(x, mu, loc=0, scale=1)

ig = histogram(hist, x, pdf, cdf, label)

ig.opts(width = 800, height = 700 , show_grid=True)
def hist(mu):

    data = stats.invgauss.rvs(mu, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['mu'])

hmap.redim.range(mu = (0.01,1)).opts(width = 700, height = 600 , show_grid=True).relabel('Inverse Gaussian Distribution Histogram')
def pdf(mu):

    xs = np.linspace(0, 0.1, 1000)

    ys = [stats.invgauss.pdf(x, mu, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['mu'])

hmap1.redim.range(mu = (0.01,0.1)).opts(width = 700, height = 600 , show_grid=True).relabel('Inverse Gaussian Distribution PDF')
def cdf(mu):

    xs = np.linspace(0, 0.1, 1000)

    ys = [stats.invgauss.cdf(x, mu, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['mu'])

hmap1.redim.range(mu = (0.01,0.1)).opts(width = 700, height = 600 , show_grid=True).relabel('Inverse Gaussian Distribution CDF')
label = "Inverse Weibull Distribution (c = 4)"

c = 4



measured = stats.invweibull.rvs(c, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 4, 1000)

pdf = stats.invweibull.pdf(x, c, loc=0, scale=1)

cdf = stats.invweibull.cdf(x, c, loc=0, scale=1)

iw = histogram(hist, x, pdf, cdf, label)

iw.opts(width = 800, height = 700 , show_grid=True)
def hist(c):

    data = stats.invweibull.rvs(c, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['c'])

hmap.redim.range(c = (4,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Inverse Weibull Distribution Histogram')
def pdf(c):

    xs = np.linspace(0, 6, 1000)

    ys = [stats.invweibull.pdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['c'])

hmap1.redim.range(c = (4,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Inverse Weibull Distribution PDF')
def cdf(c):

    xs = np.linspace(0, 6, 1000)

    ys = [stats.invweibull.cdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['c'])

hmap1.redim.range(c = (4,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Inverse Weibull Distribution CDF')
label = "Johnson's SB-Distribution (a, b = 4, 3)"

a, b = 4, 3



measured = stats.johnsonsb.rvs(a, b, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 0.5, 1000)

pdf = stats.johnsonsb.pdf(x, a, b, loc=0, scale=1)

cdf = stats.johnsonsb.cdf(x, a, b, loc=0, scale=1)

jsb = histogram(hist, x, pdf, cdf, label)

jsb.opts(width = 800, height = 700 , show_grid=True)
def hist(a, b):

    data = stats.johnsonsb.rvs(a, b, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['a', 'b'])

hmap.redim.range(a = (1,5), b=(1,10)).opts(width = 700, height = 600 , show_grid=True).relabel("Johnson's SB-Distribution Histogram")
def pdf(a,b):

    xs = np.linspace(0, 1, 1000)

    ys = [stats.johnsonsb.pdf(x, a, b, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['a', 'b'])

hmap1.redim.range(a = (1,5), b=(1,10)).opts(width = 700, height = 600 , show_grid=True).relabel("Johnson's SB-Distribution PDF")
def cdf(a,b):

    xs = np.linspace(0, 1, 1000)

    ys = [stats.johnsonsb.cdf(x, a, b, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['a', 'b'])

hmap1.redim.range(a = (1,5), b=(1,10)).opts(width = 700, height = 600 , show_grid=True).relabel("Johnson's SB-Distribution CDF")
label = "Johnson's SU-Distribution (a, b = 4, 3)"

a, b = 4, 3



measured = stats.johnsonsu.rvs(a, b, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(-4, 0, 1000)

pdf = stats.johnsonsu.pdf(x, a, b, loc=0, scale=1)

cdf = stats.johnsonsu.cdf(x, a, b, loc=0, scale=1)

jsu = histogram(hist, x, pdf, cdf, label)

jsu.opts(width = 800, height = 700 , show_grid=True)
def hist(a, b):

    data = stats.johnsonsu.rvs(a, b, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['a', 'b'])

hmap.redim.range(a = (1,5), b=(1,10)).opts(width = 700, height = 600 , show_grid=True).relabel("Johnson's SU-Distribution Histogram")
def pdf(a,b):

    xs = np.linspace(-5, 1, 1000)

    ys = [stats.johnsonsu.pdf(x, a, b, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['a', 'b'])

hmap1.redim.range(a = (1,5), b=(1,10)).opts(width = 700, height = 600 , show_grid=True).relabel("Johnson's SU-Distribution PDF")
def cdf(a,b):

    xs = np.linspace(-5, 1, 1000)

    ys = [stats.johnsonsu.cdf(x, a, b, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['a', 'b'])

hmap1.redim.range(a = (1,5), b=(1,10)).opts(width = 700, height = 600 , show_grid=True).relabel("Johnson's SU-Distribution CDF")
label = "Kappa 4 Parameter Distribution (h, k = 0.1, 0)"

h, k = 0.1, 0



measured = stats.kappa4.rvs(h, k, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(-2, 4, 1000)

pdf = stats.kappa4.pdf(x, h, k, loc=0, scale=1)

cdf = stats.kappa4.cdf(x, h, k, loc=0, scale=1)

k4 = histogram(hist, x, pdf, cdf, label)

k4.opts(width = 800, height = 700 , show_grid=True)
def hist(h,k):

    data = stats.kappa4.rvs(h, k, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['h', 'k'])

hmap.redim.range(h = (0.1,1), k=(0,1)).opts(width = 700, height = 600 , show_grid=True).relabel('Kappa 4 Parameter Distribution Histogram')
def pdf(h,k):

    xs = np.linspace(-2, 6, 1000)

    ys = [stats.kappa4.pdf(x, h, k, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['h', 'k'])

hmap1.redim.range(h = (0.1,1), k=(0,1)).opts(width = 700, height = 600 , show_grid=True).relabel('Kappa 4 Parameter Distribution PDF')
def cdf(h,k):

    xs = np.linspace(-2, 6, 1000)

    ys = [stats.kappa4.cdf(x, h, k, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['h', 'k'])

hmap1.redim.range(h = (0.1,1), k=(0,1)).opts(width = 700, height = 600 , show_grid=True).relabel('Kappa 4 Parameter Distribution CDF')
label = "Kappa 3 Parameter Distribution (a = 1)"

a = 1



measured = stats.kappa3.rvs(a, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 1000, 1000)

pdf = stats.kappa3.pdf(x, a, loc=0, scale=1)

cdf = stats.kappa3.cdf(x, a, loc=0, scale=1)

k3 = histogram(hist, x, pdf, cdf, label)

k3.opts(width = 800, height = 700 , show_grid=True)
def hist(a):

    data = stats.kappa3.rvs(a, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['a'])

hmap.redim.range(a = (0.1,1.5)).opts(width = 700, height = 600 , show_grid=True).relabel('Kappa 3 Parameter Distribution Histogram')
def pdf(a):

    xs = np.linspace(0, 1000, 1000)

    ys = [stats.kappa3.pdf(x, a, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['a'])

hmap1.redim.range(a = (0.1,1.5)).opts(width = 700, height = 600 , show_grid=True).relabel('Kappa 3 Parameter Distribution PDF')
def cdf(a):

    xs = np.linspace(0, 1000, 1000)

    ys = [stats.kappa3.cdf(x, a, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['a'])

hmap1.redim.range(a = (0.1,1.5)).opts(width = 700, height = 600 , show_grid=True).relabel('Kappa 3 Parameter Distribution CDF')
label = "General Kolmogorov-Smirnov One-Sided Distribution (n = 1e+03)"

n = 1e+03



measured = stats.ksone.rvs(n, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 0.1, 1000)

pdf = stats.ksone.pdf(x, n, loc=0, scale=1)

cdf = stats.ksone.cdf(x, n, loc=0, scale=1)

kso = histogram(hist, x, pdf, cdf, label)

kso.opts(width = 800, height = 700 , show_grid=True)
def hist(n):

    data = stats.ksone.rvs(n, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['n'])

hmap.redim.range(n = (1e+02,1e+04)).opts(width = 700, height = 600 , show_grid=True).relabel('General Kolmogorov-Smirnov One-Sided Distribution Histogram')
def pdf(n):

    xs = np.linspace(0, 0.2, 1000)

    ys = [stats.ksone.pdf(x, n, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['n'])

hmap1.redim.range(n = (1e+02,1e+04)).opts(width = 700, height = 600 , show_grid=True).relabel('General Kolmogorov-Smirnov One-Sided Distribution PDF')
def cdf(n):

    xs = np.linspace(0, 0.2, 1000)

    ys = [stats.ksone.cdf(x, n, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['n'])

hmap1.redim.range(n = (1e+02,1e+04)).opts(width = 700, height = 600 , show_grid=True).relabel('General Kolmogorov-Smirnov One-Sided Distribution CDF')
label = "Kolmogorov-Smirnov Two-Sided Distribution"



measured = stats.kstwobign.rvs(size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 2, 1000)

pdf = stats.kstwobign.pdf(x, loc=0, scale=1)

cdf = stats.kstwobign.cdf(x, loc=0, scale=1)

ks2 = histogram(hist, x, pdf, cdf, label)

ks2.opts(width = 800, height = 700 , show_grid=True)
label = "Laplace Distribution (loc = 0, scale = 1)"

loc = 0

scale = 1



measured = stats.laplace.rvs(size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(-6, 6, 1000)

pdf = stats.laplace.pdf(x,loc, scale=1)

cdf = stats.laplace.cdf(x,loc, scale=1)

lap = histogram(hist, x, pdf, cdf, label)

lap.opts(width = 800, height = 700 , show_grid=True)
def hist(loc, scale):

    data = stats.laplace.rvs(loc, scale, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['loc', 'scale'])

hmap.redim.range(loc = (0, 2), scale=(1,5)).opts(width = 700, height = 600 , show_grid=True).relabel('Laplace Distribution Histogram')
def pdf(loc, scale):

    xs = np.linspace(-6, 6, 1000)

    ys = [stats.laplace.pdf(x,loc, scale) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['loc', 'scale'])

hmap1.redim.range(loc = (0, 2), scale=(1,5)).opts(width = 700, height = 600 , show_grid=True).relabel('Laplace Distribution PDF')
def cdf(loc, scale):

    xs = np.linspace(-6, 6, 1000)

    ys = [stats.laplace.cdf(x,loc, scale) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['loc', 'scale'])

hmap1.redim.range(loc = (0, 2), scale=(1,5)).opts(width = 700, height = 600 , show_grid=True).relabel('Laplace Distribution CDF')
label = "Lévy Distribution (loc=0, scale=10)"

loc=0

scale=10



measured = stats.levy.rvs(size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 1000, 1000)

pdf = stats.levy.pdf(x, loc=0, scale=10)

cdf = stats.levy.cdf(x, loc=0, scale=10)

lev = histogram(hist, x, pdf, cdf, label)

lev.opts(width = 800, height = 700 , show_grid=True)
def hist(c):

    data = stats.levy.rvs(c,size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['c'])

hmap.redim.range(c = (8,20)).opts(width = 700, height = 600 , show_grid=True).relabel('Lévy Distribution Histogram')
def pdf(c):

    xs = np.linspace(0, 20, 1000)

    ys = [stats.levy.pdf(x, loc=0, scale=c) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['c'])

hmap1.redim.range(c = (2,20)).opts(width = 700, height = 600 , show_grid=True).relabel('Lévy Distribution PDF')
def cdf(c):

    xs = np.linspace(0, 20, 1000)

    ys = [stats.levy.cdf(x, loc=0, scale=c) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['c'])

hmap1.redim.range(c = (2,20)).opts(width = 700, height = 600 , show_grid=True).relabel('Lévy Distribution CDF')
label = "Left-skewed Levy Distribution (loc=0, scale=1)"

loc=0

scale=1



measured = stats.levy_l.rvs(size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(-2, 1, 1000)

pdf = stats.levy_l.pdf(x, loc=0, scale=1)

cdf = stats.levy_l.cdf(x, loc=0, scale=1)

levl = histogram(hist, x, pdf, cdf, label)

levl.opts(width = 800, height = 700 , show_grid=True)
def hist(c):

    data = stats.levy_l.rvs(c,size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['c'])

hmap.redim.range(c = (8,20)).opts(width = 700, height = 600 , show_grid=True).relabel('Left-skewed Levy Distribution Histogram')
def pdf(c):

    xs = np.linspace(-8, 1, 1000)

    ys = [stats.levy_l.pdf(x, loc=0, scale=c) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['c'])

hmap1.redim.range(c = (2,20)).opts(width = 700, height = 600 , show_grid=True).relabel('Left-skewed Levy PDF')
def cdf(c):

    xs = np.linspace(-8, 1, 1000)

    ys = [stats.levy_l.cdf(x, loc=0, scale=c) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['c'])

hmap1.redim.range(c = (2,20)).opts(width = 700, height = 600 , show_grid=True).relabel('Left-skewed Levy CDF')
label = "Levy-stable Distribution (alpha = 2 beta = -1)"

alpha, beta = 2, -1



measured = stats.levy_stable.rvs(alpha, beta, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(-10, 10, 1000)

pdf = stats.levy_stable.pdf(x, alpha, beta, loc=0, scale=1)

cdf = stats.levy_stable.pdf(x, alpha, beta, loc=0, scale=1)

levs = histogram(hist, x, pdf, cdf, label)

levs.opts(width = 800, height = 700 , show_grid=True)
def hist(alpha, beta):

    data = stats.levy_stable.rvs(alpha, beta, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['alpha', 'beta'])

hmap.redim.range(alpha = (1, 2), beta=(1, 2)).opts(width = 700, height = 600 , show_grid=True).relabel('Levy-stable Distribution Histogram')
def pdf(alpha, beta):

    xs = np.linspace(-10, 10, 1000)

    ys = [stats.levy_stable.pdf(x, alpha, beta, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['alpha', 'beta'])

hmap1.redim.range(alpha = (0.5, 3), beta=(-1, 1)).opts(width = 700, height = 600 , show_grid=True).relabel('Levy-stable Distribution PDF')
def cdf(alpha, beta):

    xs = np.linspace(-10, 10, 1000)

    ys = [stats.levy_stable.cdf(x, alpha, beta, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['alpha', 'beta'])

hmap1.redim.range(alpha = (0.5, 3), beta=(-1, 1)).opts(width = 700, height = 600 , show_grid=True).relabel('Levy-stable Distribution CDF')