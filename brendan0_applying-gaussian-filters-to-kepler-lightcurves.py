import numpy as np

import pandas as pd

import matplotlib as plt

import seaborn as sns

from  scipy import ndimage
data = pd.read_csv('../input/exoTrain.csv', index_col=0)

test_data = data = pd.read_csv('../input/exoTest.csv', index_col=0)
data.head()
data.tail()
data = data.reset_index(drop=True)
data.head()
data.shape
time = np.arange(3197)

flux = data.iloc[1,:]
sns.set()
ax = sns.regplot(time, flux, fit_reg=False, scatter_kws={"s": 10})

sns.set(font_scale=1.5)

ax.set(xlabel='Time', ylabel='Flux')

ax.set_title("Lightcurve from Star 1")

ax.set_xlim(0, 3197)
smoothedFlux=ndimage.filters.gaussian_filter(flux, sigma=1)
print ("Without Gaussian Smoothing")

print ("Min: ", min(flux)) 

print ("Max: ", max(flux))

print ("Range: ", max(flux)-min(flux)) 
print ("With Gaussian Smoothing")

print ("Min: ", min(smoothedFlux)) 

print ("Max: ", max(smoothedFlux))

print ("Range: ", max(smoothedFlux)-min(smoothedFlux))
ax = sns.regplot(time, smoothedFlux, fit_reg=False, scatter_kws={"s": 10})

sns.set(font_scale=1.5)

ax.set(xlabel='Time', ylabel='Flux')

ax.set_title("Smoothed Lightcurve from Star 1")

ax.set_xlim(0, 3197)
verySmoothedFlux=ndimage.filters.gaussian_filter(flux, sigma=7)
ax = sns.regplot(time, verySmoothedFlux, fit_reg=False, scatter_kws={"s": 10})

sns.set(font_scale=1.5)

ax.set(xlabel='Time', ylabel='Flux')

ax.set_title("Very Smoothed Lightcurve from Star 1")

ax.set_xlim(0, 3197)
flux2 = data.iloc[2,:]



smoothedFlux2 = ndimage.filters.gaussian_filter(flux2, sigma=1)



verySmoothedFlux2 = ndimage.filters.gaussian_filter(flux2, sigma=7)
ax = sns.regplot(time, flux2, fit_reg=False, scatter_kws={"s": 10})

sns.set(font_scale=1.5)

ax.set(xlabel='Time', ylabel='Flux')

ax.set_title("Lightcurve from Star 2")

ax.set_xlim(0, 3197)
ax = sns.regplot(time, smoothedFlux2, fit_reg=False, scatter_kws={"s": 10})

sns.set(font_scale=1.5)

ax.set(xlabel='Time', ylabel='Flux')

ax.set_title("Smoothed Lightcurve from Star 2")

ax.set_xlim(0, 3197)
ax = sns.regplot(time, verySmoothedFlux2, fit_reg=False, scatter_kws={"s": 10})

sns.set(font_scale=1.5)

ax.set(xlabel='Time', ylabel='Flux')

ax.set_title("Very Smoothed Lightcurve from Star 2")

ax.set_xlim(0, 3197)
flux200 = data.iloc[200,:]

flux201 = data.iloc[201,:]



smoothedFlux200=ndimage.filters.gaussian_filter(flux200, sigma=1)

smoothedFlux201=ndimage.filters.gaussian_filter(flux201, sigma=1)



verySmoothedFlux200=ndimage.filters.gaussian_filter(flux200, sigma=7)

verySmoothedFlux201=ndimage.filters.gaussian_filter(flux201, sigma=7)
ax = sns.regplot(time, flux200, fit_reg=False, scatter_kws={"s": 10})

sns.set(font_scale=1.5)

ax.set(xlabel='Time', ylabel='Flux')

ax.set_title("Lightcurve from Star 200")

ax.set_xlim(0, 3197)
ax = sns.regplot(time, smoothedFlux200, fit_reg=False, scatter_kws={"s": 10})

sns.set(font_scale=1.5)

ax.set(xlabel='Time', ylabel='Flux')

ax.set_title("Smoothed Lightcurve from Star 200")

ax.set_xlim(0, 3197)
ax = sns.regplot(time, verySmoothedFlux200, fit_reg=False, scatter_kws={"s": 10})

sns.set(font_scale=1.5)

ax.set(xlabel='Time', ylabel='Flux')

ax.set_title("Very Smoothed Lightcurve from Star 200")

ax.set_xlim(0, 3197)
ax = sns.regplot(time, flux201, fit_reg=False, scatter_kws={"s": 10})

sns.set(font_scale=1.5)

ax.set(xlabel='Time', ylabel='Flux')

ax.set_title("Lightcurve from Star 201")

ax.set_xlim(0, 3197)
ax = sns.regplot(time, smoothedFlux201, fit_reg=False, scatter_kws={"s": 10})

sns.set(font_scale=1.5)

ax.set(xlabel='Time', ylabel='Flux')

ax.set_title("Smoothed Lightcurve from Star 201")

ax.set_xlim(0, 3197)
ax = sns.regplot(time, verySmoothedFlux201, fit_reg=False, scatter_kws={"s": 10})

sns.set(font_scale=1.5)

ax.set(xlabel='Time', ylabel='Flux')

ax.set_title("Very Smoothed Lightcurve from Star 201")

ax.set_xlim(0, 3197)
print ("Without Gaussian Smoothing")

print ("Min: ", min(flux201)) 

print ("Max: ", max(flux201))

print ("Range: ", max(flux201)-min(flux201))
print ("With Some Gaussian Smoothing")

print ("Min: ", min(smoothedFlux201)) 

print ("Max: ", max(smoothedFlux201))

print ("Range: ", max(smoothedFlux201)-min(smoothedFlux201))
print ("With Heavy Gaussian Smoothing")

print ("Min: ", min(verySmoothedFlux201)) 

print ("Max: ", max(verySmoothedFlux201))

print ("Range: ", max(verySmoothedFlux201)-min(verySmoothedFlux201))