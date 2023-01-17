import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
#Display two line graphs
#import necessary libraries
import numpy as np #linear algebra
import pandas as pd #data processing,CSV file I/O (e.g.pd.read_csv) 
import matplotlib.pyplot as plt
#Create the data table 
time_series = np.matrix([[1,10,2],[2,13,4],[3,12,8],[4,9,14],[5,10.5,20]])
print(time_series)
#Produce the figure
plt.figure(figsize=(12,8)) #figure size
#"""Draw the Earning of Person 1:
X1 = time_series[:,0] #Column 0 of our time_series matrix 
Y1 = time_series[:,1] #Earning of persion 1,Column 1 of our time_serirs matrix #"""
X2 = time_series[:,0]
Y2 = time_series[:,2]
plt.plot(X1,Y1,'b-',label='Person 1') 
plt.plot(X2,Y2,'r-',label='person 2') 
#plt.plot(Y1,Y2)
plt.xlabel('Date')
plt.ylabel('Earning')
plt.title('Earning of Person 1 & 2') 
plt.legend()