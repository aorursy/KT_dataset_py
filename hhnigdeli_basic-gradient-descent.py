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
data = pd.read_csv("../input/genderheightweightcsv/weight-height.csv")
data.head() # Height and Weight column names are missmatched
data = data[data["Gender"]=="Male"] # I use just males' data
data = data[["Height","Weight"]] # I won't use Gender column 
data.columns = ["Weight","Height"] # Height and Weight column names are missmatched
data.head(5)
import matplotlib.pyplot as plt
plt.scatter(data.Height,data.Weight)
import numpy as np
import random
newIndex = random.sample(list(data.index), 25)

dataSampled = data.reindex(newIndex)
plt.scatter(dataSampled.Height,dataSampled.Weight)
x = list(dataSampled.Height)

y = list(dataSampled.Weight)

N = len(dataSampled)
x = [(i-np.mean(x))/np.std(x)  for i in x]

y = [(i-np.mean(y))/np.std(y)  for i in y]
plt.scatter(x,y)
# Parameters 



LearningRate = 0.01

b = 0

m = 1

iter = 300
c=0

liste = []

mselist = [] # mean square errors list for each iteration 



while c <iter:

        slopem = 0

        slopeb = 0

        for i in range(len(dataSampled)):



                slopem  +=  -2*x[i]*(y[i]-(m*x[i]+b)) # sum of lost function derivatives with respect to m

                

                slopeb  +=  -2*(y[i]-(m*x[i]+b))      # sum of lost function derivatives with respect to b

                

    

        b -= ((slopeb/N)  * LearningRate) # new b

        m -= ((slopem/N) * LearningRate)  # new m

        

        mse = 0

       

        for i in range(N):

            mse += (y[i] - (m*x[i]+b))**2 # mean square error calculation

            

        mselist.append(mse/N)  

        

        liste.append((b,m))

        

        c+=1

print("intercept = {} , coefficent = {} , MeanSuqareError = {}".format(b,m,mse))
plt.plot(mselist,color="r")

plt.scatter(range(iter),mselist)

plt.xlabel("Iteration")

plt.ylabel("Mean Suqare Error")

plt.title("Gradient Descent")
regressionLine = [ (b + m*i)    for i in range(-3,5)]





plt.scatter(x,y, color="k")

plt.plot(range(-3,5),regressionLine, color="r",linewidth=4)

plt.title("Fited Regression Line ")

plt.xlabel("X")

plt.ylabel("Y")