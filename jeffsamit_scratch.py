print('Hello World')
# Homework 1



import numpy as np 

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

points = [18,32,20,23,39,21,30,25,13,19,23,29,33,25,23,30,33,29,23,25,25,20,31,32,25,28,32,20,21,23,21,13,31,17,21,31,35,31,19,31,31]

print("number of games: ", len(points)) 

print("range: ", np.min(points), "-", np.max(points))

print("points per game: ", np.mean(points))

print("median: ", np.median(points))

from statistics import mode 

print("mode: ", mode(points))

plt.hist(points,6)

plt.xlabel("Points")

plt.ylabel("N")

plt.title("Lebron James Point Distribution")

plt.show()







#Homework 2



game_number = np.arange(len(points))

print(game_number)

plt.plot(game_number,points)

plt.xlabel("Game Number")

plt.ylabel("Points")

plt.title("Lebron James's points over time")

plt.show()



bi_points = [22,25,35,27,25,7,40,27,25,21,28,33,24,23,26,20,24,21,np.nan,31,25,32,21,22,34,25,19,31,24,27,22,16,35,29,28,16,49,21,25,22,13]

ad_points = [25,21,29,40,31,25,15,26,27,24,np.nan,17,14,34,33,22,19,41,26,27,25,26,39,50,16,33,27,36,32,24,20,23,26,46,24,5,9,28,16,31,26]

giannis_points = [30,29,14,22,29,36,34,38,30,35,38,26,33,33,24,28,50,30,33,26,29,35,27,32,np.nan,np.nan,37,29,48,34,22,18,18,23,32,32,24,30,13,32,37]

beal_points = [19,17,25,46,30,22,30,20,44,44,34,33,30,20,14,35,18,23,42,26,23,20,16,29,35,22,37,36,30,15,np.nan,np.nan,27,np.nan,np.nan,25,23,29,38,36,40]



plt.plot(game_number,points,label = "James")

plt.plot(game_number,bi_points,label = "Ingram")

plt.plot(game_number,ad_points,label = "Davis")

plt.plot(game_number,giannis_points,label = "Antetokounmpo")

plt.plot(game_number,beal_points,label = "Beal")

plt.legend()



plt.xlabel("Game Number")

plt.ylabel("Points")

plt.title("NBA All Stars' points over time")

plt.show()
#Homework 3



import numpy as np 

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

points = [18,32,20,23,39,21,30,25,13,19,23,29,33,25,23,30,33,29,23,25,25,20,31,32,25,28,32,20,21,23,21,13,31,17,21,31,35,31,19,31,31]

print("number of games: ", len(points)) 

print("range: ", np.min(points), "-", np.max(points))

print("points per game: ", np.mean(points))

print("median: ", np.median(points))

from statistics import mode 

print("mode: ", mode(points))

plt.hist(points,6)

plt.xlabel("Points")

plt.ylabel("N")

plt.title("Lebron James Point Distribution")

plt.show()





game_number = np.arange(len(points))

print(game_number)

plt.plot(game_number,points)

plt.xlabel("Game Number")

plt.ylabel("Points")

plt.title("Lebron James's points over time")

plt.show()



points_std = np.std(points)

print("Standard Deviation:", points_std)



plt.errorbar(game_number,points,yerr=points_std)

plt.xlabel("Game Number")

plt.ylabel("Points")

plt.title("Lebron James's points over time")



plt.ylim(0,50)

plt.show()



from scipy.optimize import curve_fit



def f(x, A, B): return A*x + B



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
#Final Component: Favorite Player Project

import numpy as np 

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

shots = [19,22,14,15,23,23,19,19,15,18,21,20,21,21,20,27,24,18,16,20,18,21,23,20,24,22,21,20,19,24,15,10,21,17,21,19,25,16,19,25,16]

print("number of games: ", len(shots)) 

print("range: ", np.min(shots), "-", np.max(shots))

print("points per game: ", np.mean(shots))

print("median: ", np.median(shots))

from statistics import mode 

print("mode: ", mode(shots))



game_number = np.arange(len(shots))

print(game_number)





from scipy.optimize import curve_fit



def f(x, A, B): return A*x + B



popt, pcov = curve_fit(f, game_number, shots) 

print("Slope:",popt[0]) 

print("Intercept:",popt[1])



y_fit = popt[0]*game_number + popt[1]



plt.plot(game_number, shots, y_fit,'--')

plt.xlabel("Game Number")

plt.ylabel("Field Goals Attempted")

plt.title("Lebron James's FGA Over Time")

plt.ylim(5,32)

plt.show()