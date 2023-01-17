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
label = "Semicircle Distribution (r = 0.5)"

r = 0.5



measured = stats.semicircular.rvs(size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(-2, 2, 1000)

pdf = stats.semicircular.pdf(x,loc=0, scale=r)

cdf = stats.semicircular.cdf(x,loc=0, scale=r)

s = histogram(hist, x, pdf, cdf, label)

s.opts(width = 800, height = 700 , show_grid=True)
def hist(r):

    data = stats.semicircular.rvs(size=1000, scale =r)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['r'])

hmap.redim.range(r = (0.5,5)).opts(width = 700, height = 600 , show_grid=True).relabel('Semicircle Distribution Histogram')
def pdf(r):

    xs = np.linspace(-2, 2, 1000)

    ys = [stats.semicircular.pdf(x,loc=0, scale=r) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['r'])

hmap1.redim.range(r = (0.5,5)).opts(width = 700, height = 600 , show_grid=True).relabel('Semicircle Distribution PDF')
def cdf(r):

    xs = np.linspace(-2, 2, 1000)

    ys = [stats.semicircular.cdf(x,loc=0, scale=r) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['r'])

hmap1.redim.range(r = (0.5,5)).opts(width = 700, height = 600 , show_grid=True).relabel('Semicircle Distribution CDF')
label = "Skew Normal Distribution (a = 3)"

a = 3



measured = stats.skewnorm.rvs(a, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(-2, 4, 1000)

pdf = stats.skewnorm.pdf(x, a, loc=0, scale=1)

cdf = stats.skewnorm.cdf(x, a, loc=0, scale=1)

sn = histogram(hist, x, pdf, cdf, label)

sn.opts(width = 800, height = 700 , show_grid=True)
def hist(a):

    data = stats.skewnorm.rvs(a, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['a'])

hmap.redim.range(a = (3,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Skew Normal Distribution Histogram')
def pdf(a):

    xs = np.linspace(-2, 4, 1000)

    ys = [stats.skewnorm.pdf(x, a, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['a'])

hmap1.redim.range(a = (3,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Skew Normal Distribution PDF')
def cdf(a):

    xs = np.linspace(-2, 4, 1000)

    ys = [stats.skewnorm.cdf(x, a, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['a'])

hmap1.redim.range(a = (3,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Skew Normal Distribution CDF')
label = "Student's t-distribution (df = 2)"

df = 2



measured = stats.t.rvs(df, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(-6, 6, 1000)

pdf = stats.t.pdf(x, df, loc=0, scale=1)

cdf = stats.t.cdf(x, df, loc=0, scale=1)

td = histogram(hist, x, pdf, cdf, label)

td.opts(width = 800, height = 700 , show_grid=True)
def hist(df):

    data = stats.t.rvs(df, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['df'])

hmap.redim.range(df = (5,50)).opts(width = 700, height = 600 , show_grid=True).relabel("Student's t-distribution Histogram")
def pdf(df):

    xs = np.linspace(-6, 6, 1000)

    ys = [stats.t.pdf(x, df, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['df'])

hmap1.redim.range(df = (5,50)).opts(width = 700, height = 600 , show_grid=True).relabel("Student's t-distribution PDF")
def cdf(df):

    xs = np.linspace(-6, 6, 1000)

    ys = [stats.t.cdf(x, df, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['df'])

hmap1.redim.range(df = (5,50)).opts(width = 700, height = 600 , show_grid=True).relabel("Student's t-distribution CDF")
label = "Trapezoidal Distribution (c = 0.5, d = 1)"

c, d = 0.5, 1



measured = stats.trapz.rvs(c, d, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 2, 1000)

pdf = stats.trapz.pdf(x, c, d, loc=0, scale=1)

cdf = stats.trapz.pdf(x, c, d, loc=0, scale=1)

trd = histogram(hist, x, pdf, cdf, label)

trd.opts(width = 800, height = 700 , show_grid=True)
def hist(c, d):

    data = stats.trapz.rvs(c, d, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['c', 'd'])

hmap.redim.range(c = (0.01,0.1), d=(0.1, 1)).opts(width = 700, height = 600 , show_grid=True).relabel('Trapezoidal Distribution Histogram')
def pdf(c, d):

    xs = np.linspace(0, 2, 1000)

    ys = [stats.trapz.pdf(x, c, d, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['c', 'd'])

hmap1.redim.range(c = (0.01,0.1), d=(0.1, 1)).opts(width = 700, height = 600 , show_grid=True).relabel('Trapezoidal Distribution PDF')
def cdf(c, d):

    xs = np.linspace(0, 2, 1000)

    ys = [stats.trapz.cdf(x, c, d, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['c', 'd'])

hmap1.redim.range(c = (0.01,0.1), d=(0.1, 1)).opts(width = 700, height = 600 , show_grid=True).relabel('Trapezoidal Distribution CDF')
label = "Triangular Distribution (c = 0.1)"

c = 0.1



measured = stats.triang.rvs(c, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 1, 1000)

pdf = stats.triang.pdf(x, c, loc=0, scale=1)

cdf = stats.triang.cdf(x, c, loc=0, scale=1)

td = histogram(hist, x, pdf, cdf, label)

td.opts(width = 800, height = 700 , show_grid=True)
def hist(c):

    data = stats.triang.rvs(c, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['c'])

hmap.redim.range(c = (0.01,1)).opts(width = 700, height = 600 , show_grid=True).relabel('Triangular Distribution Histogram')
def pdf(c):

    xs = np.linspace(0, 1, 1000)

    ys = [stats.triang.pdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['c'])

hmap1.redim.range(c = (0.01,1)).opts(width = 700, height = 600 , show_grid=True).relabel('Triangular Distribution PDF')
def cdf(c):

    xs = np.linspace(0, 1, 1000)

    ys = [stats.triang.cdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['c'])

hmap1.redim.range(c = (0.01,1)).opts(width = 700, height = 600 , show_grid=True).relabel('Triangular Distribution CDF')
label = "Truncated Exponential Distribution (b = 5)"

b = 5



measured = stats.truncexpon.rvs(b, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 5, 1000)

pdf = stats.truncexpon.pdf(x, b, loc=0, scale=1)

cdf = stats.truncexpon.cdf(x, b, loc=0, scale=1)

te = histogram(hist, x, pdf, cdf, label)

te.opts(width = 800, height = 700 , show_grid=True)
def hist(b):

    data = stats.truncexpon.rvs(b, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['b'])

hmap.redim.range(b = (1,3)).opts(width = 700, height = 600 , show_grid=True).relabel('Truncated Exponential Distribution Histogram')
def pdf(b):

    xs = np.linspace(0, 1, 1000)

    ys = [stats.truncexpon.pdf(x, b, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['b'])

hmap1.redim.range(b = (1,3)).opts(width = 700, height = 600 , show_grid=True).relabel('Truncated Exponential Distribution PDF')
def cdf(b):

    xs = np.linspace(0, 1.5, 1000)

    ys = [stats.truncexpon.cdf(x, b, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['b'])

hmap1.redim.range(b = (1,3)).opts(width = 700, height = 600 , show_grid=True).relabel('Truncated Exponential Distribution CDF')
label = "Truncated Normal Distribution (a = 0.1, b= 2)"

a, b = 0.1, 2



measured = stats.truncnorm.rvs(a, b, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 2, 1000)

pdf = stats.truncnorm.pdf(x, a, b, loc=0, scale=1)

cdf = stats.truncnorm.pdf(x, a, b, loc=0, scale=1)

tn = histogram(hist, x, pdf, cdf, label)

tn.opts(width = 800, height = 700 , show_grid=True)
def hist(a, b):

    data = stats.truncnorm.rvs(a, b, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['a', 'b'])

hmap.redim.range(a = (0.1,1), b=(1,5)).opts(width = 700, height = 600 , show_grid=True).relabel('Truncated Normal Distribution Histogram')
def pdf(a, b):

    xs = np.linspace(0, 2, 1000)

    ys = [stats.truncnorm.pdf(x, a, b, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['a', 'b'])

hmap1.redim.range(a = (0.1,1), b=(1,5)).opts(width = 700, height = 600 , show_grid=True).relabel('Truncated Normal Distribution PDF')
def cdf(a, b):

    xs = np.linspace(0, 2, 1000)

    ys = [stats.truncnorm.cdf(x, a, b, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['a', 'b'])

hmap1.redim.range(a = (0.1,1), b=(1,5)).opts(width = 700, height = 600 , show_grid=True).relabel('Truncated Normal Distribution CDF')
label = "Tukey Lambda Distribution (lam = 3)"

lam = 3



measured = stats.tukeylambda.rvs(lam, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(-0.5, 0.5, 1000)

pdf = stats.tukeylambda.pdf(x, lam, loc=0, scale=1)

cdf = stats.tukeylambda.cdf(x, lam, loc=0, scale=1)

tl = histogram(hist, x, pdf, cdf, label)

tl.opts(width = 800, height = 700 , show_grid=True)
def hist(lam):

    data = stats.tukeylambda.rvs(lam, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['lam'])

hmap.redim.range(lam=(1, 10)).opts(width = 700, height = 600 , show_grid=True).relabel('Tukey Lambda Distribution Histogram')
def pdf(lam):

    xs = np.linspace(-2, 2, 1000)

    ys = [stats.tukeylambda.pdf(x, lam, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['lam'])

hmap1.redim.range(lam=(1, 10)).opts(width = 700, height = 600 , show_grid=True).relabel('Tukey Lambda Distribution PDF')
def cdf(lam):

    xs = np.linspace(-2, 2, 1000)

    ys = [stats.tukeylambda.cdf(x, lam, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['lam'])

hmap1.redim.range(lam=(1, 10)).opts(width = 700, height = 600 , show_grid=True).relabel('Tukey Lambda Distribution CDF')
label = "Uniform Distribution (a = 1, b = 5)"

a = 1

b = 5



measured = stats.uniform.rvs(a, b, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 6, 1000)

pdf = stats.uniform.pdf(x, loc=a, scale=b)

cdf = stats.uniform.cdf(x, loc=a, scale=b)

ud = histogram(hist, x, pdf, cdf, label)

ud.opts(width = 800, height = 700 , show_grid=True)
def hist(a,b):

    data = stats.uniform.rvs(a, b, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['a', 'b'])

hmap.redim.range(a=(1,10), b=(10,20)).opts(width = 700, height = 600 , show_grid=True).relabel('Uniform Distribution Histogram')
def pdf(a,b):

    xs = np.linspace(0,6+b, 1000)

    ys = [stats.uniform.pdf(x, loc=a, scale=b) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['a', 'b'])

hmap1.redim.range(a=(1,10), b=(10,20)).opts(width = 700, height = 600 , show_grid=True).relabel('Uniform Distribution PDF')
def cdf(a,b):

    xs = np.linspace(0,6+b, 1000)

    ys = [stats.uniform.cdf(x,loc=a, scale=b) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['a', 'b'])

hmap1.redim.range(a=(1,10), b=(10,20)).opts(width = 700, height = 600 , show_grid=True).relabel('Uniform Distribution CDF')
label = "von Mises Distribution (kappa = 4)"

kappa = 4



measured = stats.vonmises.rvs(kappa, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(-3, 3, 1000)

pdf = stats.vonmises.pdf(x, kappa, loc=0, scale=1)

cdf = stats.vonmises.cdf(x, kappa, loc=0, scale=1)

vm = histogram(hist, x, pdf, cdf, label)

vm.opts(width = 800, height = 700 , show_grid=True)
def hist(kappa):

    data = stats.vonmises.rvs(kappa, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['kappa'])

hmap.redim.range(kappa= (2, 10)).opts(width = 700, height = 600 , show_grid=True).relabel("von Mises Distribution Histogram")
def pdf(kappa):

    xs = np.linspace(-4, 4, 1000)

    ys = [stats.vonmises.pdf(x, kappa, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['kappa'])

hmap1.redim.range(kappa= (2, 10)).opts(width = 700, height = 600 , show_grid=True).relabel("von Mises Distribution PDF")
def cdf(kappa):

    xs = np.linspace(-4, 4, 1000)

    ys = [stats.vonmises.cdf(x, kappa, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['kappa'])

hmap1.redim.range(kappa= (2, 10)).opts(width = 700, height = 600 , show_grid=True).relabel("von Mises Distribution CDF")
label = "Wald Distribution (a = 2, b = 3)"

a = 2

b = 3



measured = stats.wald.rvs(a, b, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 20, 1000)

pdf = stats.wald.pdf(x,loc=a, scale=b)

cdf = stats.wald.cdf(x,loc=a, scale=b)

wal = histogram(hist, x, pdf, cdf, label)

wal.opts(width = 800, height = 700 , show_grid=True)
def hist(a, b):

    data = stats.wald.rvs(a,b, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['a', 'b'])

hmap.redim.range(a= (1, 5),b = (1,4)).opts(width = 700, height = 600 , show_grid=True).relabel("Wald Distribution Histogram")
def pdf(a, b):

    xs = np.linspace(0, 2, 1000)

    ys = [stats.wald.pdf(x, loc=a, scale=b) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['a', 'b'])

hmap1.redim.range(a= (0.1, 2),b = (0.1,2)).opts(width = 700, height = 600 , show_grid=True).relabel("Wald Distribution PDF")
def cdf(a, b):

    xs = np.linspace(0, 3, 1000)

    ys = [stats.wald.cdf(x, loc=a, scale=b) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['a', 'b'])

hmap1.redim.range(a= (0.1, 2),b = (0.1,2)).opts(width = 700, height = 600 , show_grid=True).relabel("Wald Distribution CDF")
label = "Weibull Maximum Distribution (c = 2)"

c = 2



measured = stats.weibull_max.rvs(c, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(-3, 0, 1000)

pdf = stats.weibull_max.pdf(x, c, loc=0, scale=1)

cdf = stats.weibull_max.cdf(x, c, loc=0, scale=1)

wm = histogram(hist, x, pdf, cdf, label)

wm.opts(width = 800, height = 700 , show_grid=True)
def hist(c):

    data = stats.weibull_max.rvs(c, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['c'])

hmap.redim.range(c= (1, 10)).opts(width = 700, height = 600 , show_grid=True).relabel("Weibull Maximum Distribution Histogram")
def pdf(c):

    xs = np.linspace(-3, 0, 1000)

    ys = [stats.weibull_max.pdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['c'])

hmap1.redim.range(c= (1, 10)).opts(width = 700, height = 600 , show_grid=True).relabel("Weibull Maximum Distribution PDF")
def cdf(c):

    xs = np.linspace(-3, 0, 1000)

    ys = [stats.weibull_max.cdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['c'])

hmap1.redim.range(c= (1, 10)).opts(width = 700, height = 600 , show_grid=True).relabel("Weibull Maximum Distribution CDF")
label = "Weibull Minimum Distribution (c = 2)"

c = 2



measured = stats.weibull_min.rvs(c, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 3, 1000)

pdf = stats.weibull_min.pdf(x, c, loc=0, scale=1)

cdf = stats.weibull_min.cdf(x, c, loc=0, scale=1)

wm = histogram(hist, x, pdf, cdf, label)

wm.opts(width = 800, height = 700 , show_grid=True)
def hist(c):

    data = stats.weibull_min.rvs(c, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['c'])

hmap.redim.range(c= (1, 10)).opts(width = 700, height = 600 , show_grid=True).relabel("Weibull Minimum Distribution Histogram")
def pdf(c):

    xs = np.linspace(0, 3, 1000)

    ys = [stats.weibull_min.pdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['c'])

hmap1.redim.range(c= (1, 10)).opts(width = 700, height = 600 , show_grid=True).relabel("Weibull Minimum Distribution PDF")
def cdf(c):

    xs = np.linspace(0, 3, 1000)

    ys = [stats.weibull_min.cdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['c'])

hmap1.redim.range(c= (1, 10)).opts(width = 700, height = 600 , show_grid=True).relabel("Weibull Minimum Distribution CDF")
label = "Wrapped Cauchy Distribution (c = 0.03)"

c = 0.03



measured = stats.wrapcauchy.rvs(c, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 6, 1000)

pdf = stats.wrapcauchy.pdf(x, c, loc=0, scale=1)

cdf = stats.wrapcauchy.cdf(x, c, loc=0, scale=1)

wc = histogram(hist, x, pdf, cdf, label)

wc.opts(width = 800, height = 700 , show_grid=True)
def hist(c):

    data = stats.wrapcauchy.rvs(c, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['c'])

hmap.redim.range(c= (0.01, 0.5)).opts(width = 700, height = 600 , show_grid=True).relabel("Wrapped Cauchy Distribution Histogram")
def pdf(c):

    xs = np.linspace(0, 6, 1000)

    ys = [stats.wrapcauchy.pdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['c'])

hmap1.redim.range(c= (0.01, 0.5)).opts(width = 700, height = 600 , show_grid=True).relabel("Reciprocal Inverse Gaussian Distribution PDF")
def cdf(c):

    xs = np.linspace(0, 6, 1000)

    ys = [stats.wrapcauchy.cdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['c'])

hmap1.redim.range(c= (0.01, 0.5)).opts(width = 700, height = 600 , show_grid=True).relabel("Reciprocal Inverse Gaussian Distribution CDF")