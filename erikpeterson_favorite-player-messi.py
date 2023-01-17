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
# Lionel Messi goals in La Liga

goals = [1,6,14,10,23,34,31,50,46,28,43,26,37,34,36,13]
print("number of seasons: ", len(goals)) # len() is the length of the list
print("range: ", np.min(goals), "-", np.max(goals))
print("goals per season: ", np.mean(goals))

print("median: ", np.median(goals))
from statistics import mode #import the function 'mode'

print("mode: ", mode(goals))
plt.hist(goals,10) # Even though the homework says 6 bins, I've put 10 to see the histogram better

plt.xlabel("Goals")

plt.ylabel("N")

plt.title("Lionel Messi Goal Distribution")

plt.show()
# goals over time



season_number = np.arange(len(goals))

print(season_number) # keep in mind, computers start counting at 0!
plt.plot(season_number,goals)

plt.xlabel("Season Number")

plt.ylabel("Goals")

plt.title("Lionel Messi's goals over the years")

plt.show()
# Adding 4 other players

# Making sure these are the same seasons that they played in and filling in nothing if they didn't play

suarez_goals = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,16,40,29,25,21,11]

neymar_goals = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,9,22,24,13,np.nan,np.nan,np.nan]

pedro_goals = [np.nan,np.nan,np.nan,0,0,12,13,5,7,15,6,np.nan,np.nan,np.nan,np.nan,np.nan]

iniesta_goals = [2,0,6,3,4,1,8,2,3,3,0,1,0,1,np.nan,np.nan]
plt.plot(season_number,goals,label = "Messi")

plt.plot(season_number,suarez_goals,label = "Suarez")

plt.plot(season_number,neymar_goals,label = "Neymar")

plt.plot(season_number,pedro_goals,label = "Pedro")

plt.plot(season_number,iniesta_goals,label = "Iniesta")

plt.legend()



plt.xlabel("Season Number")

plt.ylabel("Goals")

plt.title("Barcelona's goals over time")

plt.show()
goals_std = np.std(goals)

print("Standard Deviation:", goals_std)
# What I've commented off below is what we plotted in Homework 2. Look how similar the code is!

# plt.plot(season_number,goals)

plt.errorbar(season_number,goals,

             yerr=goals_std)

plt.xlabel("Season Number")

plt.ylabel("Goals")

plt.title("Lionel Messi's goals over the years")

# I also want to restrict the width of my y axis because negative goals are not possible (well, maybe own goals)

# Use your discression here, but think about what limits might make sense in your case!

plt.ylim(0,65)

plt.show()
from scipy.optimize import curve_fit
def f(x, A, B): # this is your 'straight line' y=f(x)

    return A*x + B
popt, pcov = curve_fit(f, season_number, goals) # your data x, y to fit

print("Slope:",popt[0]) 

print("Intercept:",popt[1])
# y = m*x + b

y_fit = popt[0]*season_number + popt[1]
plt.errorbar(season_number,goals,

             yerr=goals_std)

# the fit!

plt.plot(season_number, y_fit,'--')

plt.xlabel("Season Number")

plt.ylabel("Goals")

plt.title("Lionel Messi's goals over the years")

plt.ylim(0,65)

plt.show()