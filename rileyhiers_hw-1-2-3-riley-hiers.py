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
# Lebron James

points = [33, 29, 23, 25, 25, 20, 31, 32, 25, 28, 32, 20, 21, 23, 21, 13, 31, 17, 21, 31, 35, 31, 19, 31, 15]
print("number of games: ", len(points))
import numpy as np # linear algebra
print("range: ", np.min(points), "-", np.max(points))
print("points per game: ", np.mean(points))

print("median: ", np.median(points))
from statistics import mode

print("mode: ", mode(points))
plt.hist(points,6)

plt.xlabel("Points")

plt.ylabel("# of Games")

plt.title("Lebron James Point Distribution")

plt.show()
game_number = np.arange(len(points))

print(game_number)
plt.plot(game_number,points)

plt.xlabel("Game Number")

plt.ylabel("Points")

plt.title("Lebron James's points over time")

plt.show()
anthonydavis_points = [22, 19, 41, 26, 27, 25, 26, 39, 50, 16, 33, 27, 36, 24, 20, 23, 26, 46, 24, 5, np.nan, np.nan, np.nan, np.nan, 9] #np.nan means "not a number"

dannygreen_points = [6, 4, 11, 8, 5, 6, 3, 12, 8, 9, 10, 6, 21, 6, 3, 10, 9, 25, 7, 11, 3, 10, 11, 20, 7]

kylekuzma_points = [10, 16, 14, 4, 6, 13, 15, 7, np.nan, np.nan, np.nan, np.nan, np.nan, 25, 24, 0, 19, 10, 4, 16, 26, 11, 4, 23, 13]

alexcaruso_points = [7, 0, 8, 4, 10, 6, 5, 8, 16, 7, 2, 2, 11, 0, 2, 9, 6, 0, np.nan, 13, 2, 9, 10, 0, 4]
plt.plot(game_number,points,label = "Lebron")

plt.plot(game_number,anthonydavis_points,label = "Davis")

plt.plot(game_number,dannygreen_points,label = "Green")

plt.plot(game_number,kylekuzma_points,label = "Kuzma")

plt.plot(game_number,alexcaruso_points,label = "Caruso")

plt.legend()



plt.xlabel("Game Number")

plt.ylabel("Points")

plt.title("Lakers points over time")

plt.show()
points_std = np.std(points)

print("Standard Deviation:", points_std)
plt.errorbar(game_number,points,

             yerr=points_std)

plt.xlabel("Game Number")

plt.ylabel("Points")

plt.title("Lebron James's points over time")

plt.ylim(0,50)

plt.show()
from scipy.optimize import curve_fit
def f(x, A, B):

    return A*x + B
popt, pcov = curve_fit(f, game_number, points)

print("Slope:",popt[0]) 

print("Intercept:",popt[1])
y_fit = popt[0]*game_number + popt[1]
plt.errorbar(game_number,points,

             yerr=points_std)

plt.plot(game_number, y_fit,'--')

plt.xlabel("Game Number")

plt.ylabel("Points")

plt.title("Lebron James's points over time")

plt.ylim(0,50)

plt.show()