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
# Zion Williamson

points = [28,27,21,13,13,22,25,17,18,20,17,17,25,30,11,35,27,25,22,26,29,16,18,27,32,0,29,31,21,25,32,23,24]
print("number of games: ", len(points)) # len() is the length of the list
print("range: ", np.min(points), "-", np.max(points))
print("points per game: ", np.mean(points))

print("median: ", np.median(points))
from statistics import mode #import the function 'mode'

print("mode: ", mode(points))
plt.hist(points,6) # 6 bins

plt.xlabel("Points")

plt.ylabel("N")

plt.title("Zion Williamson Point Distribution")

plt.show()
# points over time



game_number = np.arange(len(points))

print(game_number) # keep in mind, computers start counting at 0!
plt.plot(game_number,points)

plt.xlabel("Game Number")

plt.ylabel("Points")

plt.title("Zion Williamson's points over time")

plt.show()
# Adding 4 other players

# Making sure these are the same games that they played in and filling in nothing if they didn't play

rj_points = [33,23,20,20,18,23,22,26,27,30,27,16,13,21,32,23,30,26,24,17,15,19,26,13,23,33,23,15,17,26,16,18,21]

tre_points = [6,8,2,14,10,17,15,0,10,6,3,13,10,6,8,2,np.nan,np.nan,6,9,13,11,13,6,13,3,15,11,18,5,11,22,4] #np.nan means "not a number"

cam_points = [22,25,3,16,18,10,13,23,5,10,9,8,4,10,23,np.nan,9,15,7,13,16,24,17,22,9,27,7,6,11,12,13,np.nan,8]

bold_points = [7,0,8,4,11,6,0,4,4,2,7,0,11,12,3,12,2,7,2,8,10,6,5,2,9,0,np.nan,np.nan,np.nan,2,0,4,0]
plt.plot(game_number,points,label = "Williamson")

plt.plot(game_number,rj_points,label = "Barrett")

plt.plot(game_number,tre_points,label = "Jones")

plt.plot(game_number,cam_points,label = "Reddish")

plt.plot(game_number,bold_points,label = "Bolden")

plt.legend()



plt.xlabel("Game Number")

plt.ylabel("Points")

plt.title("Duke's points over time")

plt.show()
points_std = np.std(points)

print("Standard Deviation:", points_std)
# What I've commented off below is what we plotted in Homework 2. Look how similar the code is!

# plt.plot(game_number,points)

plt.errorbar(game_number,points,

             yerr=points_std)

plt.xlabel("Game Number")

plt.ylabel("Points")

plt.title("Zion Williamson's points over time")

# I also want to restrict the width of my y axis because negative points are not possible (well, maybe...)

# Use your discression here, but think about what limits might make sense in your case!

plt.ylim(0,45)

plt.show()
from scipy.optimize import curve_fit
def f(x, A, B): # this is your 'straight line' y=f(x)

    return A*x + B
popt, pcov = curve_fit(f, game_number, points) # your data x, y to fit

print("Slope:",popt[0]) 

print("Intercept:",popt[1])
# y = m*x + b

y_fit = popt[0]*game_number + popt[1]
plt.errorbar(game_number,points,

             yerr=points_std)

# the fit!

plt.plot(game_number, y_fit,'--')

plt.xlabel("Game Number")

plt.ylabel("Points")

plt.title("Zion Williamson's points over time")

plt.ylim(0,45)

plt.show()