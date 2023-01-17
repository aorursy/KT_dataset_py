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
import matplotlib.pyplot as plt

# Cristiano Ronaldo goals from 2008-2009 season to 2019-2020

goals = [18,26,40,46,34,31,48,35,25,26,21,14]

print( "the number of seasons: ", len(goals))

print( "range: ", np.min(goals), "-", np.max(goals) )      #np->numpy linear algebra

print( "goals per season: ", np.around(np.mean(goals),1))  #used around to keep 1 decimal place

print( "median: ", np.median(goals))

from statistics import mode #import mode method

print( "mode: ", mode(goals))



plt.hist(goals,6)

plt.xlabel("Goals") #label x-axis

plt.ylabel("Frequency") #label y-axis

plt.title("Cristiano Ronaldo Goal Distribution") #mark title

plt.show()



season_number = np.arange(len(goals))

print(season_number + 2008)
plt.plot(season_number + 2008,goals)

plt.xlabel("Season Year")

plt.ylabel("Goals")

plt.title("Cristiano Ronaldo's goals over year")

plt.show()
#add 4 other players

Hazard_goals = [4,5,7,20,9,14,14,4,16,np.nan,12,16]

Bale_goals = [np.nan,3,7,10,21,15,13,19,7,16,8,2]

Benzema_goals = [17,8,15,21,11,17,15,24,11,5,21,12]

Modric_goals = [3,3,3,4,3,1,1,2,1,1,3,3]



plt.plot(season_number + 2008, goals, label = "Ronaldo" )

plt.plot(season_number + 2008, Hazard_goals, label = "Hazard")

plt.plot(season_number + 2008, Bale_goals, label = "Bale")

plt.plot(season_number + 2008, Benzema_goals, label = "Benzema")

plt.plot(season_number + 2008, Modric_goals, label = "Modric")

plt.legend()



plt.xlabel("Season Year")

plt.ylabel("Goals")

plt.title("Real Madrid's goals over time")

plt.show()

goals_std = np.std(goals)

print("Standard Deviation:", goals_std)
plt.errorbar(season_number + 2008 ,goals,yerr=goals_std) 

#The error bar is added to each season to show uncertainty, yerr means error bar in y-axis, in this case

#the standard deviation of goals

plt.xlabel("Season Year")

plt.ylabel("Goals")

plt.title("Cristiano Ronaldo's goals over the years")

plt.ylim(0,60)

plt.show()
from scipy.optimize import curve_fit

def f(x,A,B): return A*x + B

popt,pcov = curve_fit(f,season_number, goals)

print("Slope:", popt[0])

#popt returns the array of A and B, so popt[0] means A

print("Intercept:", popt[1])

#This returns the value B in popt array
y_fit = popt[0]*season_number + popt[1]

#y= A*x + B

plt.errorbar(season_number + 2008, goals, yerr = goals_std)

plt.plot(season_number + 2008, y_fit, '--') #'type of line used','-'would be a continuous line

plt.xlabel("Season Year")

plt.ylabel("Goals")

plt.title("Cristiano Ronaldo's goals over the years")

plt.ylim(0,60)

plt.show()
goals_per_90min = [0.59,0.95,1.24,1.24,1.13,1.10,1.39,0.99,0.89,1.02,0.90,1.02]

plt.plot(season_number + 2008,goals_per_90min)

plt.xlabel("Season Year")

plt.ylabel("Goals per 90 minutes")

plt.title("Ronaldo's Goals/90min over time")

plt.show()
