# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

#matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

# print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
wave_length = 5.55 * 0.01 # in meters

elevation_width = 0.84

azimuth_width = 12.3

elevation_elements = 20

azimuth_elements = 14

theta_el = wave_length / elevation_width

theta_az = wave_length / azimuth_width

d = 4 * np.pi / (theta_az * theta_el)

from IPython.display import display, HTML, Math

display(Math(r'\theta_{el} = '+'{:.3}'.format(theta_el / np.pi * 180)))

display(Math(r'\theta_{az} = '+'{:.3}'.format(theta_az / np.pi * 180)))

display(Math('D_{dbi} = 10 * log_{10}(D) = ' + '{:.4}'.format(10 * np.log10(d))))

display(Math('G = 0.87 \cdot D \implies G_{dbi} = 10 * log_{10}(G) = ' + '{:.4}'.format(10 * np.log10(0.87 * d))))
theta = np.linspace(-20, 20, 100)

theta_r = theta * np.pi / 180

delta_x = elevation_width / elevation_elements

rho = np.sinc(theta_r) * np.sum([np.exp(-1j * 2 * np.pi/wave_length * n * delta_x * np.sin(theta_r)) for n in range(elevation_elements)])

plt.plot(theta, rho)

plt.show()

theta = np.linspace(-2.5, 2.5, 101)

theta_r = theta * np.pi / 180

delta_x = azimuth_width / azimuth_elements

rho = np.sinc(theta_r) * np.sum([np.exp(-1j * 2 * np.pi/wave_length * n * delta_x * np.sin(theta_r)) for n in range(azimuth_elements)])

plt.plot(theta, rho)

plt.show()

rho2 = np.sinc(theta_r) * np.sum([np.exp(-1j * 2 * np.pi/wave_length * n * delta_x * (np.sin(theta_r) - np.sin(1.2 * np.pi/180))) for n in range(azimuth_elements)])

plt.plot(theta, rho2)

plt.show()

rho_3db = rho2[50].real - 3

print(np.real(rho2))
