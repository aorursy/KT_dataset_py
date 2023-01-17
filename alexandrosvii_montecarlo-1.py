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
S=100000# S as a multiple of 1000 



def rounder( n ):   

    # Smaller multiple 

    a = (n // 1000) * 1000      

    # Larger multiple 

    b = a + 1000      

    # Return of closest of two 

    return (b if n - a > b - n else a)  



# driver code 

print(rounder(S)) 

S=rounder(S)

Q=20000

R=4



for R_loop in range(R):

    triples_user=int(S//Q)

    triples = triples_user

    shots = int(S//R)

    tenperc=int(0.1*shots)

    

    if R_loop == 0:

        shots= max(tenperc,Q)+shots

    

    elif R_loop == R-1:

        shots= shots -max(tenperc,Q)

    print('TEN_PERCENT   TRIPLES     SHOTS     S     Q     Rloop    ')

    print(tenperc,'            ',triples,'', shots,' ',S,' ',Q,' ',R_loop,' ',)

    coordinates = np.random.rand(triples, shots, 2) * 2 - 1  #Coordinates between [-1,1]            

    dist_x2y2 = np.sqrt((coordinates**2).sum(axis=2)) # This is the distance from Sqrt X2Y2    

    incircle = dist_x2y2 <= 1                       

    num_thrown = np.arange(1, shots+1)  #How many shots we applied ?

    num_incircle = np.cumsum(incircle, axis=1) #How many landed inside the dartboard?

    

    #triples trials

    total_estimate = 4 * num_incircle / num_thrown

    mean_incircle = total_estimate.mean(axis=0)  

    std_incircle = total_estimate.std(axis=0)     



import matplotlib.pyplot as plt

%matplotlib notebook



fig, ax = plt.subplots()

ax.plot(num_thrown, mean_incircle, label="mean");

ax.fill_between(num_thrown, y1=mean_incircle-std_incircle, y2=mean_incircle+std_incircle,

                alpha=0.2, label="standard deviation")

ax.hlines(y=np.pi, xmin=1, xmax=shots+1, linestyles="--")

ax.set_xscale("log")

ax.grid(True)

ax.set_ylabel("Estimated value of pi")

ax.set_xlabel("Number of darts thrown")

ax.legend();



max(tenperc,Q)