# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import matplotlib.pyplot as plt
N = 121 #Want a lot of points. I made a point for every half angle between 30 and 90 degrees

theta = np.linspace(30,90,N)



def v(theta,x,g,h,H):

    radians = theta * np.pi / 180 #np.cos/tan take radian values

    return (x/np.cos(radians))*np.sqrt((g/2)/(x*np.tan(radians)+h-H)) #maybe write this equation on scratch paper to see what it looks like



x = 15 #ft Distance to hoop

g = 32.174 #ft/s^2

h = 6 #ft Height of person

H = 10 #ft Height of hoop

rim_radius = 0.738 #ft

ball_radius = 0.398 #ft



plt.plot(theta,v(theta,x,g,h,H),label = 'perfect')

plt.plot(theta,v(theta,x-rim_radius+ball_radius,g,h,H),label = 'min')

plt.plot(theta,v(theta,x+rim_radius-ball_radius,g,h,H),label = 'max')



plt.ylim(24.3,27)

plt.xlim(38,68)



#Label axes with units

plt.xlabel('Angle')

plt.ylabel('Velocity')



plt.legend()

plt.grid() #to better see the plot

plt.show()