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
# Mohamed Salah all career league goals

goals = [0,4,7,5,6,6,14,15,32,22,15]

points = [0,1,10,8,8,9,20,26,42,30,21]
print("number of seasons: ", len(goals)) 
print("range: ", np.min(goals), "-", np.max(goals))
print("goals per season: ", np.mean(goals))

print("median: ", np.median(goals))
plt.hist(goals,6)

plt.xlabel("Goals")

plt.ylabel("Frequency")

plt.title("Mohamed Salah Leauge Goal Distribution")

plt.show()
# goals over time



season_number = np.arange(len(goals))

print(season_number)
plt.plot(season_number,goals)

plt.xlabel("Season Number")

plt.ylabel("Goals")

plt.title("Mohamed Salah's goals over the years")

plt.show()
Mane_goals = [np.nan,np.nan,1,17,13,12,11,13,10,22,11]

Mane_std=np.std(Mane_goals)

Firmino_goals = [np.nan, 3,7,5,16,7,10,11,15,12,8]

Firmino_std = np.std(Firmino_goals)

Shaqiri_goals = [4,5,9,4,6,2,3,4,8,6,1]

Shaqiri_std = np.std(Shaqiri_goals)

Origi_goals = [np.nan,np.nan,2,1,5,8,5,7,6,3,3]

Origi_std=np.std(Origi_goals)

goals_std=np.std(goals)

points_std=np.std(points)


plt.plot(season_number,goals,label = "Salah")

plt.errorbar(season_number,goals,yerr=goals_std)

plt.legend()

plt.ylim(0,42)

plt.xlabel("Season Number")

plt.ylabel("Goals")

plt.title("Salah's goals over time")

plt.show()
from scipy.optimize import curve_fit

def f(x, A, B): # this is your 'straight line' y=f(x)

    return A*x + B

popt, pcov = curve_fit(f, season_number, points)

y_fit = popt[0]*season_number + popt[1]

plt.errorbar(season_number,points,

             yerr=points_std)

plt.plot(season_number,points,label = "Salah")

plt.errorbar(season_number,points,yerr=points_std)

plt.plot(season_number, y_fit,'--')

plt.legend()

plt.ylim(0,60)

plt.xlabel("Season Number")

plt.ylabel("Points")

plt.title("Salah's points over time")

plt.show()
from scipy.optimize import curve_fit

def f(x, A, B): # this is your 'straight line' y=f(x)

    return A*x + B

popt, pcov = curve_fit(f, season_number, goals)

y_fit = popt[0]*season_number + popt[1]

plt.errorbar(season_number,goals,

             yerr=goals_std)

# the fit!

plt.plot(season_number, y_fit,'--')

plt.xlabel("Season Number")

plt.ylabel("Goals")

plt.title("Mohamed Salah's goals over the years")

plt.ylim(0,42)

plt.show()
