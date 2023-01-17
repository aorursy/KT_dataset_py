# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

%matplotlib inline



import matplotlib.pyplot as plt

#Create 100 random numbers between 0 and 0.1

count = 10

noise = np.random.rand(count)



#Make the input variable imperfect, by adding a random

#number between 0 and 0.1 to it

noisy_x = x 



#Calculate the imperfect values of "y", if "x" were imperfect

y = 50*(noisy_x)*(noisy_x) + 2*(noisy_x) 



#Plot imperfect y against the perfect x

plt.plot(x, y, 'ro')
'''Play with the number of random "events" to get 

visual proof of central limit theorem. 



Lets take some random unrelated "events" that produce values 

between 0 and 100. 



Lets have 1000 trials ( each try has a set of "events" )



Find the average of the events in each trial and plot them. 



If number of event = 1, then avarage across trials is 

uniformly distributed between 0 and 100 



As you increase the number of events, the distribution of 

the average, starts converging to the center (~50) '''



events = 5   #Number of random unrelated events

trials =1000 #Number of tries for each set of random events above

z= np.random.randint(0,100,[events,trials]) #integers between 0 and 100, randomly chose for each variable, and each try

total = sum(z[:])/events #For each trial, Average the set of events 



#Plot a histogram with 11 bins for values of "averages" stored in array total

plt.hist(total, bins=11, normed=0)

plt.axis([0,100,0,300])

plt.xlabel('average')

plt.ylabel('frequency')

plt.show()
import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
# A matrix is an array of numbers arranged in rows and columns

rows = 10

cols = 10

# Lets define an array with random integer values between 0 and 99

# In numpy you can first define a series of numbers and then 

# reshape them into rows and columns

a = np.random.randint(0,100,(rows*cols)) # a series of rows*cols numbers. 

a = a.reshape(rows, cols)

print("A numpy matrix with %d and %d columns \n \n" % (rows,cols), a)
# A matrix with only one row or one columns is also called a vector

# A row vector has only 1 row

row_vector = np.random.randint(0,100,cols).reshape(1,cols)

print("A 1x%d Row Vector : \n" % (cols), row_vector)
# A column vector has only 1 coulmn

column_vector = np.random.randint(0, 100, rows).reshape(rows,1)

print("A %dx1 Column Vector : \n" % (rows), column_vector)
# You can use a matrix to store points in space

points = np.array([[-3,-2],[3,-2],[3,2],[-3,2],[-3,-2]])

x = points[:,0] # pulls out all the "x values"

y = points[:,1] # pulls out all the "y values"



# Lets plot the points and draw a line between them

import matplotlib.pyplot as plt

plt.axis([-10,10,-10,10])

plt.plot(x,y, marker = "o")

plt.show()
# You can also use a Matrix to store pixels.



# Lets create a 8x10 matrix of pixel brightness values between 0 and 255

my_pixels = np.linspace(255,0,8*10) #Ascending series of 80 numbers between 0 and 255

my_pixels = my_pixels.reshape(8,10) # Reshape into 8 rows and 10 columns

plt.imshow(my_pixels, cmap = plt.cm.Blues) # Using a grayscale colormap(cmap)

plt.show()
# A 2x2 matrix named a

a = np.array([[1,2],

             [3,4]])

print(a)



# Another 2x2 matrix named b

b = np.array([[5,6], 

              [7,8]])

print(a + b)

print(a * 3)



x = np.array([1,2,3,4,5,6])

x.reshape([6,1])  # A column vector
theta = np.array([2,4,6,8,10,12])

theta.reshape([6,1]) # Another column vector
theta_t = np.transpose(theta)

theta_t
np.dot(theta_t, theta)
a = ([3,5], [4 , 3])

b = ([12,16])





a_t = np.transpose(a)

print(a_t)

print("The dot product is : ", np.dot(a_t,b))



print("The inner product is : ", np.inner(a_t,b))