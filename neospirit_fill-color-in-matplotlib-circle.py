import matplotlib.pyplot as plt

import matplotlib.cm     as cm

from matplotlib.colors import Normalize



# fill color in Circle

cmap = cm.jet   # Select colormap U want

# Declare for set range of value for normalization

vmin = 0        

vmax = 1

# Normalize value for cmap

norm = Normalize(vmin, vmax)



figure, axes = plt.subplots()



# There you are

draw_circle = plt.Circle((0.5,0.5),0.3,color=cmap(norm(0.7)))

draw_circle2 = plt.Circle((0.5,0.8),0.3,color=cmap(norm(0.5)))



axes.set_aspect(1)

axes.add_artist(draw_circle)

axes.add_artist(draw_circle2)

plt.title('Circle')

plt.show()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import numpy as np



x = list(range(1,6))

y = list(range(10, 20, 2))



print(x, y)



for i, data in enumerate(zip(x,y)):

    j, k = data

    plt.scatter(j,k, marker = "o", s = ((i+1)**4)*50, alpha = 0.3)
import numpy as np

import pylab as plt



def rect(x,y,w,h,c):

    ax = plt.gca()

    polygon = plt.Rectangle((x,y),w,h,color=c)

    ax.add_patch(polygon)



def rainbow_fill(X,Y, cmap=plt.get_cmap("jet")):

    plt.plot(X,Y,lw=0)  # Plot so the axes scale correctly



    dx = X[1]-X[0]

    N  = float(X.size)



    for n, (x,y) in enumerate(zip(X,Y)):

        color = cmap(n/N)

        rect(x,0,dx,y,color)



# Test data    

X = np.linspace(0,10,100)

Y = .25*X**2 - X

rainbow_fill(X,Y)

plt.show()
print(dir(plt.get_cmap('rainbow')))
import numpy as np

import matplotlib.pyplot as plt

import matplotlib.colors





def wavelength_to_rgb(wavelength, gamma=0.8):

    ''' taken from http://www.noah.org/wiki/Wavelength_to_RGB_in_Python

    This converts a given wavelength of light to an 

    approximate RGB color value. The wavelength must be given

    in nanometers in the range from 380 nm through 750 nm

    (789 THz through 400 THz).



    Based on code by Dan Bruton

    http://www.physics.sfasu.edu/astro/color/spectra.html

    Additionally alpha value set to 0.5 outside range

    '''

    wavelength = float(wavelength)

    if wavelength >= 380 and wavelength <= 750:

        A = 1.

    else:

        A=0.5

    if wavelength < 380:

        wavelength = 380.

    if wavelength >750:

        wavelength = 750.

    if wavelength >= 380 and wavelength <= 440:

        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)

        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma

        G = 0.0

        B = (1.0 * attenuation) ** gamma

    elif wavelength >= 440 and wavelength <= 490:

        R = 0.0

        G = ((wavelength - 440) / (490 - 440)) ** gamma

        B = 1.0

    elif wavelength >= 490 and wavelength <= 510:

        R = 0.0

        G = 1.0

        B = (-(wavelength - 510) / (510 - 490)) ** gamma

    elif wavelength >= 510 and wavelength <= 580:

        R = ((wavelength - 510) / (580 - 510)) ** gamma

        G = 1.0

        B = 0.0

    elif wavelength >= 580 and wavelength <= 645:

        R = 1.0

        G = (-(wavelength - 645) / (645 - 580)) ** gamma

        B = 0.0

    elif wavelength >= 645 and wavelength <= 750:

        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)

        R = (1.0 * attenuation) ** gamma

        G = 0.0

        B = 0.0

    else:

        R = 0.0

        G = 0.0

        B = 0.0

    return (R,G,B,A)



clim=(350,780)

norm = plt.Normalize(*clim)

wl = np.arange(clim[0],clim[1]+1,2)

colorlist = list(zip(norm(wl),[wavelength_to_rgb(w) for w in wl]))

spectralmap = matplotlib.colors.LinearSegmentedColormap.from_list("spectrum", colorlist)



fig, axs = plt.subplots(1, 1, figsize=(8,4), tight_layout=True)



wavelengths = np.linspace(200, 1000, 1000)

spectrum = (5 + np.sin(wavelengths*0.1)**2) * np.exp(-0.00002*(wavelengths-600)**2)

plt.plot(wavelengths, spectrum, color='darkred')



y = np.linspace(0, 6, 100)

X,Y = np.meshgrid(wavelengths, y)



extent=(np.min(wavelengths), np.max(wavelengths), np.min(y), np.max(y))



plt.imshow(X, clim=clim,  extent=extent, cmap=spectralmap, aspect='auto')

plt.xlabel('Wavelength (nm)')

plt.ylabel('Intensity')



plt.fill_between(wavelengths, spectrum, 8, color='w')

plt.savefig('WavelengthColors.png', dpi=200)



plt.show()
spectralmap = matplotlib.colors.LinearSegmentedColormap.from_list("spectrum", colorlist)

plt.imshow(X, clim=clim,  extent=extent, cmap=spectralmap, aspect='auto')

plt.xlabel('Wavelength (nm)')

plt.ylabel('Intensity')



plt.fill_between(wavelengths, spectrum, 8, color='w')

plt.savefig('WavelengthColors.png', dpi=200)



plt.show()