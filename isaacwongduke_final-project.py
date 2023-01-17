# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

N = 121 #Want a lot of points. I made a point for every half angle between 30 and 90 degrees

x = np.linspace(140*0.9144,150*0.9144,N)

def v(theta,x,g):

    return (x/np.cos(theta))*np.sqrt((g/2)/(x*np.tan(theta))) #maybe write this equation on scratch paper to see what it looks like

theta = 49*np.pi*(1/180) 

g = 9.81 

a=((36*np.cos(theta))**2)/(2*10*0.9144) #negative acceleration due to the green

aa=0.1*a

tmax=np.sqrt(0.021/(0.5*9.81)) #maximum time the ball can be over the hole for it to drop half the diameter of the ball (0.021 m) and thus not bounce out

vmax=0.108/tmax #max velocity of the ball before it goes over the hole (0.108m in diameter) for this time to be met

xx = np.linspace(0,10*0.9144,N) #distane the ball lands before the hole

xxx= np.linspace(0,5*0.9144,N) #distane the ball lands behind the hole

vobackspin=np.sqrt(2*aa*3*0.9144) #initial velocity of ball spinning backwards (assumes negative acceleration due to green is half of scenario 2)

def vhole(xx,a):

    return np.sqrt(((36*np.cos(theta))**2)-(2*a*(xx)))



def vholebackspin(xxx,aa):

    return np.sqrt((vobackspin**2)-(2*aa*(xxx)))





plt.plot(x,v(theta,x,g),label = 'perfect')





plt.xlim(120,140)

plt.ylim(30,40)

#Label your own axes with units

plt.xlabel('distance covered in air (m)')

plt.ylabel('velocity(m/s)')

plt.legend()

plt.grid() #to better see the plot

plt.show()



plt.plot(xx,vhole(xx,a),label = 'perfect')





plt.xlim(0,10)

plt.ylim(0,36)

#Label your own axes with units

plt.xlabel('distance before the hole of ball landing (m)')

plt.ylabel('velocity at hole (m/s)')

plt.legend()

plt.grid() #to better see the plot

plt.show()



plt.plot(xxx,vholebackspin(xxx,aa),label = 'perfect')





plt.xlim(0,3)

plt.ylim(0,36)

#Label your own axes with units

plt.xlabel('distance behind the hole of ball landing (m)')

plt.ylabel('velocity at hole (m/s)')

plt.legend()

plt.grid() #to better see the plot

plt.show()