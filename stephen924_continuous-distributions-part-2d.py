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
label = "Log Gamma Distribution (c = 0.5)"

c = 0.5



measured = stats.loggamma.rvs(c, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(-15, 4, 1000)

pdf = stats.loggamma.pdf(x, c, loc=0, scale=1)

cdf = stats.loggamma.cdf(x, c, loc=0, scale=1)

lg = histogram(hist, x, pdf, cdf, label)

lg.opts(width = 800, height = 700 , show_grid=True)
def hist(c):

    data = stats.loggamma.rvs(c, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['c'])

hmap.redim.range(c = (0.5,2)).opts(width = 700, height = 600 , show_grid=True).relabel('Log Gamma Distribution Histogram')
def pdf(c):

    xs = np.linspace(-15, 4, 1000)

    ys = [stats.loggamma.pdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['c'])

hmap1.redim.range(c = (0.5,2)).opts(width = 700, height = 600 , show_grid=True).relabel('Log Gamma Distribution PDF')
def cdf(c):

    xs = np.linspace(-15, 4, 1000)

    ys = [stats.loggamma.cdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['c'])

hmap1.redim.range(c = (0.5,2)).opts(width = 700, height = 600 , show_grid=True).relabel('Log Gamma Distribution CDF')
label = "Log-Laplace distribution (c = 3)"

c = 3



measured = stats.loglaplace.rvs(c, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 14, 1000)

pdf = stats.loglaplace.pdf(x, c, loc=0, scale=1)

cdf = stats.loglaplace.cdf(x, c, loc=0, scale=1)

ll = histogram(hist, x, pdf, cdf, label)

ll.opts(width = 800, height = 700 , show_grid=True)
def hist(c):

    data = stats.loglaplace.rvs(c, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['c'])

hmap.redim.range(c = (3,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Log-Laplace distribution Histogram')
def pdf(c):

    xs = np.linspace(0, 4, 1000)

    ys = [stats.loglaplace.pdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['c'])

hmap1.redim.range(c = (3,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Log-Laplace distribution PDF')
def cdf(c):

    xs = np.linspace(0, 4, 1000)

    ys = [stats.loglaplace.cdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['c'])

hmap1.redim.range(c = (3,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Log-Laplace distribution CDF')
label = "Lomax (Pareto of the second kind) Distribution (c = 3)"

c = 3



measured = stats.lomax.rvs(c, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 10, 1000)

pdf = stats.lomax.pdf(x, c, loc=0, scale=1)

cdf = stats.lomax.cdf(x, c, loc=0, scale=1)

lo = histogram(hist, x, pdf, cdf, label)

lo.opts(width = 800, height = 700 , show_grid=True)
def hist(c):

    data = stats.lomax.rvs(c, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['c'])

hmap.redim.range(c = (3,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Lomax (Pareto of the second kind) Distribution Histogram')
def pdf(c):

    xs = np.linspace(0, 6, 1000)

    ys = [stats.lomax.pdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['c'])

hmap1.redim.range(c = (3,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Lomax (Pareto of the second kind) Distribution PDF')
def cdf(c):

    xs = np.linspace(0, 6, 1000)

    ys = [stats.lomax.cdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['c'])

hmap1.redim.range(c = (3,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Lomax (Pareto of the second kind) Distribution CDF')
label = "Maxwell Distribution (a = 2)"

a = 2



measured = stats.maxwell.rvs(scale = a, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 10, 1000)

pdf = stats.maxwell.pdf(x,loc=0, scale=a)

cdf = stats.maxwell.cdf(x,loc=0, scale=a)

ma = histogram(hist, x, pdf, cdf, label)

ma.opts(width = 800, height = 700 , show_grid=True)
def hist(a):

    data = stats.maxwell.rvs(scale =a, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['a'])

hmap.redim.range(a = (2,5)).opts(width = 700, height = 600 , show_grid=True).relabel('Maxwell Distribution Histogram')
def pdf(a):

    xs = np.linspace(0, 10, 1000)

    ys = [stats.maxwell.pdf(x,loc=0, scale=a) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['a'])

hmap1.redim.range(a = (2,5)).opts(width = 700, height = 600 , show_grid=True).relabel('Maxwell Distribution PDF')
def cdf(a):

    xs = np.linspace(0, 10, 1000)

    ys = [stats.maxwell.cdf(x,loc=0, scale=a) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['a'])

hmap1.redim.range(a = (2,5)).opts(width = 700, height = 600 , show_grid=True).relabel('Maxwell Distribution CDF')
label = "Mielke Beta-Kappa Distribution (k=10, s=4)"

k, s = 10, 4



measured = stats.mielke.rvs(k, s, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 6, 1000)

pdf = stats.mielke.pdf(x, k, s, loc=0, scale=1)

cdf = stats.mielke.cdf(x, k, s, loc=0, scale=1)

mbk = histogram(hist, x, pdf, cdf, label)

mbk.opts(width = 800, height = 700 , show_grid=True)
def hist(k,s):

    data = stats.mielke.rvs(k, s, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['k', 's'])

hmap.redim.range(k = (3,20), s=(2,8)).opts(width = 700, height = 600 , show_grid=True).relabel('Mielke Beta-Kappa Distribution Histogram')
def pdf(k,s):

    xs = np.linspace(0, 6, 1000)

    ys = [stats.mielke.pdf(x, k, s, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['k', 's'])

hmap1.redim.range(k = (3,20), s=(2,8)).opts(width = 700, height = 600 , show_grid=True).relabel('Mielke Beta-Kappa Distribution PDF')
def cdf(k,s):

    xs = np.linspace(0, 6, 1000)

    ys = [stats.mielke.cdf(x, k, s, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['k', 's'])

hmap1.redim.range(k = (3,20), s=(2,8)).opts(width = 700, height = 600 , show_grid=True).relabel('Mielke Beta-Kappa Distribution CDF')
label = "Moyal Distribution"



measured = stats.moyal.rvs(size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(-4, 10, 1000)

pdf = stats.moyal.pdf(x, loc=0, scale=1)

cdf = stats.moyal.cdf(x, loc=0, scale=1)

moy = histogram(hist, x, pdf, cdf, label)

moy.opts(width = 800, height = 700 , show_grid=True)
label = "Nakagami Distribution (nu = 5)"

nu = 5



measured = stats.nakagami.rvs(nu, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 2, 1000)

pdf = stats.nakagami.pdf(x, nu, loc=0, scale=1)

cdf = stats.nakagami.cdf(x, nu, loc=0, scale=1)

na = histogram(hist, x, pdf, cdf, label)

na.opts(width = 800, height = 700 , show_grid=True)
def hist(nu):

    data = stats.nakagami.rvs(nu, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['nu'])

hmap.redim.range(nu = (2,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Nakagami Distribution Histogram')
def pdf(nu):

    xs = np.linspace(0, 2, 1000)

    ys = [stats.nakagami.pdf(x, nu, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['nu'])

hmap1.redim.range(nu = (2,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Nakagami Distribution PDF')
def cdf(nu):

    xs = np.linspace(0, 2, 1000)

    ys = [stats.nakagami.cdf(x, nu, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['nu'])

hmap1.redim.range(nu = (2,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Nakagami Distribution CDF')
label = "Non-central Chi-Squared Distribution (df = 21, nc = 1)"

df, nc = 21, 1



measured = stats.ncx2.rvs(df, nc, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 50, 1000)

pdf = stats.ncx2.pdf(x, df, nc, loc=0, scale=1)

cdf = stats.ncx2.cdf(x, df, nc, loc=0, scale=1)

nx = histogram(hist, x, pdf, cdf, label)

nx.opts(width = 800, height = 700 , show_grid=True)
def hist(df, nc):

    data = stats.ncx2.rvs(df, nc, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['df','nc'])

hmap.redim.range(df=(2, 50), nc = (2,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Non-central Chi-Squared Distribution Histogram')
def pdf(df, nc):

    xs = np.linspace(0, 18, 1000)

    ys = [stats.ncx2.pdf(x, df, nc, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['df','nc'])

hmap1.redim.range(df=(2, 50), nc = (2,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Non-central Chi-Squared Distribution PDF')
def cdf(df, nc):

    xs = np.linspace(0, 18, 1000)

    ys = [stats.ncx2.cdf(x, df, nc, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['df','nc'])

hmap1.redim.range(df=(2, 50), nc = (2,10)).opts(width = 700, height = 600 , show_grid=True).relabel('Non-central Chi-Squared Distribution CDF')
label = "Noncentral F-Distribution (dfn = 27, dfd = 27, nc = 0.5)"

dfn, dfd, nc = 27, 27, 0.5



measured = stats.ncf.rvs(dfn, dfd, nc, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 4, 1000)

pdf = stats.ncf.pdf(x, dfn, dfd, nc, loc=0, scale=1)

cdf = stats.ncf.cdf(x, dfn, dfd, nc, loc=0, scale=1)

nf = histogram(hist, x, pdf, cdf, label)

nf.opts(width = 800, height = 700 , show_grid=True)
def hist(dfn, dfd, nc):

    data = stats.ncf.rvs(dfn, dfd, nc, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['dfn', 'dfd', 'nc'])

hmap.redim.range(dfn=(5,50), dfd=(4,50), nc=(0.1, 3)).opts(width = 700, height = 600 , show_grid=True).relabel('Noncentral F-Distribution Histogram')
def pdf(dfn, dfd, nc):

    xs = np.linspace(0, 4, 1000)

    ys = [stats.ncf.pdf(x, dfn, dfd, nc, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['dfn', 'dfd', 'nc'])

hmap1.redim.range(dfn=(5,50), dfd=(4,50), nc=(0.1, 3)).opts(width = 700, height = 600 , show_grid=True).relabel('Noncentral F-Distribution PDF')
def cdf(dfn, dfd, nc):

    xs = np.linspace(0, 4, 1000)

    ys = [stats.ncf.cdf(x, dfn, dfd, nc, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['dfn', 'dfd', 'nc'])

hmap1.redim.range(dfn=(5,50), dfd=(4,50), nc=(0.1, 3)).opts(width = 700, height = 600 , show_grid=True).relabel('Noncentral F-Distribution CDF')
label = "Noncentral Student's t-Distribution (df = 15, nc = 0.4)"

df, nc = 15, 0.4



measured = stats.nct.rvs(df, nc, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(-8, 10, 1000)

pdf = stats.nct.pdf(x, df, nc, loc=0, scale=1)

cdf = stats.nct.cdf(x, df, nc, loc=0, scale=1)

nt = histogram(hist, x, pdf, cdf, label)

nt.opts(width = 800, height = 700 , show_grid=True)
def hist(df, nc):

    data = stats.nct.rvs(df, nc, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['df', 'nc'])

hmap.redim.range(df= (5, 50),nc = (0.1,3)).opts(width = 700, height = 600 , show_grid=True).relabel("Noncentral Student's t-Distribution Histogram")
def pdf(df, nc):

    xs = np.linspace(-6, 6, 1000)

    ys = [stats.nct.pdf(x, df, nc, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['df', 'nc'])

hmap1.redim.range(df= (5, 50), nc = (0.1,3)).opts(width = 700, height = 600 , show_grid=True).relabel("Noncentral Student's t-Distribution PDF")
def cdf(df, nc):

    xs = np.linspace(-6, 6, 1000)

    ys = [stats.nct.cdf(x, df, nc, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['df', 'nc'])

hmap1.redim.range(df= (5, 50), nc = (0.1,3)).opts(width = 700, height = 600 , show_grid=True).relabel("Noncentral Student's t-Distribution CDF")
label = "Normal-inverse Gaussian Distribution (a = 1, b = 0.5)"

a, b = 1, 0.5



measured = stats.norminvgauss.rvs(a, b, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(-2, 6, 1000)

pdf = stats.norminvgauss.pdf(x, a, b, loc=0, scale=1)

cdf = stats.norminvgauss.cdf(x, a, b, loc=0, scale=1)

nig = histogram(hist, x, pdf, cdf, label)

nig.opts(width = 800, height = 700 , show_grid=True)
def hist(a, b):

    data = stats.norminvgauss.rvs(a, b, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['a', 'b'])

hmap.redim.range(a= (0.5, 4),b = (0.1,4)).opts(width = 700, height = 600 , show_grid=True).relabel("Normal-inverse Gaussian Distribution Histogram")
def pdf(a, b):

    xs = np.linspace(-6, 6, 1000)

    ys = [stats.norminvgauss.pdf(x, a, b, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['a', 'b'])

hmap1.redim.range(a= (0.5, 4),b = (0.1,4)).opts(width = 700, height = 600 , show_grid=True).relabel("Normal-inverse Gaussian Distribution PDF")
def cdf(a, b):

    xs = np.linspace(-6, 6, 1000)

    ys = [stats.norminvgauss.cdf(x, a, b, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['a', 'b'])

hmap1.redim.range(a= (0.5, 4),b = (0.1,4)).opts(width = 700, height = 600 , show_grid=True).relabel("Normal-inverse Gaussian Distribution CDF")
label = "Pearson type III Distribution (skew = 0.5)"

skew = 0.5



measured = stats.pearson3.rvs(skew, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(-6, 6, 1000)

pdf = stats.pearson3.pdf(x, skew, loc=0, scale=1)

cdf = stats.pearson3.cdf(x, skew, loc=0, scale=1)

p3 = histogram(hist, x, pdf, cdf, label)

p3.opts(width = 800, height = 700 , show_grid=True)
def hist(skew):

    data = stats.pearson3.rvs(skew, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['skew'])

hmap.redim.range(skew= (0.5, 4)).opts(width = 700, height = 600 , show_grid=True).relabel("Pearson type III Distribution Histogram")
def pdf(skew):

    xs = np.linspace(-6, 6, 1000)

    ys = [stats.pearson3.pdf(x, skew, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['skew'])

hmap1.redim.range(skew= (0.5, 4)).opts(width = 700, height = 600 , show_grid=True).relabel("Pearson type III Distribution PDF")
def cdf(skew):

    xs = np.linspace(-6, 6, 1000)

    ys = [stats.pearson3.cdf(x, skew, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['skew'])

hmap1.redim.range(skew= (0.5, 4)).opts(width = 700, height = 600 , show_grid=True).relabel("Pearson type III Distribution CDF")
label = "Power Function Distribution (a = 2)"

a = 2



measured = stats.powerlaw.rvs(a, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 2, 1000)

pdf = stats.powerlaw.pdf(x, a, loc=0, scale=1)

cdf = stats.powerlaw.cdf(x, a, loc=0, scale=1)

pf = histogram(hist, x, pdf, cdf, label)

pf.opts(width = 800, height = 700 , show_grid=True)
def hist(a):

    data = stats.powerlaw.rvs(a, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['a'])

hmap.redim.range(a= (0.5, 4)).opts(width = 700, height = 600 , show_grid=True).relabel("Power Function Distribution Histogram")
def pdf(a):

    xs = np.linspace(0, 2, 1000)

    ys = [stats.powerlaw.pdf(x, a, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['a'])

hmap1.redim.range(a= (0.5, 4)).opts(width = 700, height = 600 , show_grid=True).relabel("Power Function Distribution PDF")
def cdf(a):

    xs = np.linspace(0, 2, 1000)

    ys = [stats.powerlaw.cdf(x, a, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['a'])

hmap1.redim.range(a= (0.5, 4)).opts(width = 700, height = 600 , show_grid=True).relabel("Power Function Distribution CDF")
label = "Power Lognormal Distribution (c = 2, s =0.5)"

c, s = 2, 0.5



measured = stats.powerlognorm.rvs(c, s, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 2, 1000)

pdf = stats.powerlognorm.pdf(x, c, s, loc=0, scale=1)

cdf = stats.powerlognorm.cdf(x, c, s, loc=0, scale=1)

pl = histogram(hist, x, pdf, cdf, label)

pl.opts(width = 800, height = 700 , show_grid=True)
def hist(c, s):

    data = stats.powerlognorm.rvs(c, s, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['c','s'])

hmap.redim.range(c= (0.5, 4), s=(0.2,5)).opts(width = 700, height = 600 , show_grid=True).relabel("Power Lognormal Distribution Histogram")
def pdf(c,s):

    xs = np.linspace(0, 2, 1000)

    ys = [stats.powerlognorm.pdf(x, c, s, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['c','s'])

hmap1.redim.range(c= (0.5, 4), s=(0.1,5)).opts(width = 700, height = 600 , show_grid=True).relabel("Power Lognormal Distribution PDF")
def cdf(c,s):

    xs = np.linspace(0, 2, 1000)

    ys = [stats.powerlognorm.cdf(x, c, s, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['c','s'])

hmap1.redim.range(c= (0.5, 4), s=(0.1,5)).opts(width = 700, height = 600 , show_grid=True).relabel("Power Lognormal Distribution CDF")
label = "Power Normal Distribution (c = 4)"

c = 4



measured = stats.powernorm.rvs(c, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(-4, 2, 1000)

pdf = stats.powernorm.pdf(x, c, loc=0, scale=1)

cdf = stats.powernorm.cdf(x, c, loc=0, scale=1)

pn = histogram(hist, x, pdf, cdf, label)

pn.opts(width = 800, height = 700 , show_grid=True)
def hist(c):

    data = stats.powernorm.rvs(c, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['c'])

hmap.redim.range(c= (2, 10)).opts(width = 700, height = 600 , show_grid=True).relabel("Power Normal Distribution Histogram")
def pdf(c):

    xs = np.linspace(-3, 2, 1000)

    ys = [stats.powernorm.pdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['c'])

hmap1.redim.range(c= (2, 10)).opts(width = 700, height = 600 , show_grid=True).relabel("Power Normal Distribution PDF")
def cdf(c):

    xs = np.linspace(-3, 2, 1000)

    ys = [stats.powernorm.cdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['c'])

hmap1.redim.range(c= (2, 10)).opts(width = 700, height = 600 , show_grid=True).relabel("Power Normal Distribution CDF")
label = "R Distribution (c = 4)"

c = 4



measured = stats.rdist.rvs(c, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(-2, 2, 1000)

pdf = stats.rdist.pdf(x, c, loc=0, scale=1)

cdf = stats.rdist.cdf(x, c, loc=0, scale=1)

r = histogram(hist, x, pdf, cdf, label)

r.opts(width = 800, height = 700 , show_grid=True)
def hist(c):

    data = stats.rdist.rvs(c, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['c'])

hmap.redim.range(c= (1, 10)).opts(width = 700, height = 600 , show_grid=True).relabel("R Distribution Histogram")
def pdf(c):

    xs = np.linspace(-2, 2, 1000)

    ys = [stats.rdist.pdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['c'])

hmap1.redim.range(c= (2, 10)).opts(width = 700, height = 600 , show_grid=True).relabel("R Distribution PDF")
def cdf(c):

    xs = np.linspace(-2, 2, 1000)

    ys = [stats.rdist.cdf(x, c, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['c'])

hmap1.redim.range(c= (2, 10)).opts(width = 700, height = 600 , show_grid=True).relabel("R Distribution CDF")
label = "Reciprocal Distribution (a = 0.1, b = 0.5)"

a, b = 0.1, 0.5



measured = stats.reciprocal.rvs(a, b, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 1, 1000)

pdf = stats.reciprocal.pdf(x, a, b, loc=0, scale=1)

cdf = stats.reciprocal.cdf(x, a, b, loc=0, scale=1)

nig = histogram(hist, x, pdf, cdf, label)

nig.opts(width = 800, height = 700 , show_grid=True)
def hist(a, b):

    data = stats.reciprocal.rvs(a, b, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['a', 'b'])

hmap.redim.range(a= (0.01, 0.1),b = (0.1,0.5)).opts(width = 700, height = 600 , show_grid=True).relabel("Reciprocal Distribution Histogram")
def pdf(a, b):

    xs = np.linspace(0, 0.5, 1000)

    ys = [stats.reciprocal.pdf(x, a, b, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['a', 'b'])

hmap1.redim.range(a= (0.01, 0.1),b = (0.1,0.5)).opts(width = 700, height = 600 , show_grid=True).relabel("Reciprocal Distribution PDF")
def cdf(a, b):

    xs = np.linspace(0, 0.5, 1000)

    ys = [stats.reciprocal.cdf(x, a, b, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['a', 'b'])

hmap1.redim.range(a= (0.01, 0.1),b = (0.1,0.5)).opts(width = 700, height = 600 , show_grid=True).relabel("Reciprocal Distribution CDF")
label = "Rayleigh Distribution (sigma = 1)"

sigma = 1



measured = stats.rayleigh.rvs(scale=sigma, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 4, 1000)

pdf = stats.rayleigh.pdf(x, loc=0, scale=sigma)

cdf = stats.rayleigh.cdf(x, loc=0, scale=sigma)

ray = histogram(hist, x, pdf, cdf, label)

ray.opts(width = 800, height = 700 , show_grid=True)
def hist(sigma):

    data = stats.rayleigh.rvs(scale=sigma, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['sigma'])

hmap.redim.range(sigma= (0.5, 5)).opts(width = 700, height = 600 , show_grid=True).relabel("Rayleigh Distribution Histogram")
def pdf(sigma):

    xs = np.linspace(0, 4, 1000)

    ys = [stats.rayleigh.pdf(x, loc=0, scale=sigma) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['sigma'])

hmap1.redim.range(sigma= (0.5, 5)).opts(width = 700, height = 600 , show_grid=True).relabel("Rayleigh Distribution PDF")
def cdf(sigma):

    xs = np.linspace(0, 4, 1000)

    ys = [stats.rayleigh.cdf(x, loc=0, scale=sigma) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['sigma'])

hmap1.redim.range(sigma= (0.5, 5)).opts(width = 700, height = 600 , show_grid=True).relabel("Rayleigh Distribution CDF")
label = "Rice Distribution (b = 1)"

b = 1



measured = stats.rice.rvs(b, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 6, 1000)

pdf = stats.rice.pdf(x, b, loc=0, scale=1)

cdf = stats.rice.cdf(x, b, loc=0, scale=1)

ri = histogram(hist, x, pdf, cdf, label)

ri.opts(width = 800, height = 700 , show_grid=True)
def hist(b):

    data = stats.rice.rvs(b, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['b'])

hmap.redim.range(b= (0.5, 5)).opts(width = 700, height = 600 , show_grid=True).relabel("Rice Distribution Histogram")
def pdf(b):

    xs = np.linspace(0, 6, 1000)

    ys = [stats.rice.pdf(x, b, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['b'])

hmap1.redim.range(b= (0.5, 5)).opts(width = 700, height = 600 , show_grid=True).relabel("Rice Distribution PDF")
def cdf(b):

    xs = np.linspace(0, 6, 1000)

    ys = [stats.rice.cdf(x, b, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['b'])

hmap1.redim.range(b= (0.5, 5)).opts(width = 700, height = 600 , show_grid=True).relabel("Rice Distribution CDF")
label = "Reciprocal Inverse Gaussian Distribution (mu = 1)"

mu = 1



measured = stats.recipinvgauss.rvs(mu, size=1000)

hist = np.histogram(measured,density=True, bins=40)



x = np.linspace(0, 15, 1000)

pdf = stats.recipinvgauss.pdf(x, mu, loc=0, scale=1)

cdf = stats.recipinvgauss.cdf(x, mu, loc=0, scale=1)

rig = histogram(hist, x, pdf, cdf, label)

rig.opts(width = 800, height = 700 , show_grid=True)
def hist(mu):

    data = stats.recipinvgauss.rvs(mu, size=1000)

    frequencies, edges = np.histogram(data, 40)

    return hv.Histogram((edges, frequencies))



hmap = hv.DynamicMap(hist, kdims=['mu'])

hmap.redim.range(mu= (0.5, 2)).opts(width = 700, height = 600 , show_grid=True).relabel("Reciprocal Inverse Gaussian Distribution Histogram")
def pdf(mu):

    xs = np.linspace(0, 15, 1000)

    ys = [stats.recipinvgauss.pdf(x, mu, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(pdf, kdims=['mu'])

hmap1.redim.range(mu= (0.5, 2)).opts(width = 700, height = 600 , show_grid=True).relabel("Reciprocal Inverse Gaussian Distribution PDF")
def cdf(mu):

    xs = np.linspace(0, 15, 1000)

    ys = [stats.recipinvgauss.cdf(x, mu, loc=0, scale=1) for x in xs]

    return hv.Curve((xs, ys))



hmap1 = hv.DynamicMap(cdf, kdims=['mu'])

hmap1.redim.range(mu= (0.5, 2)).opts(width = 700, height = 600 , show_grid=True).relabel("Reciprocal Inverse Gaussian Distribution CDF")