# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

#I will first import data from an external open library  called "solar-power-generation-data
#using pandas
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data_project_industrial = pd.read_csv("../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv")
print(data_project_industrial)


#Then, I want to check the objects using dtypes, I should be able to 
#look for any data.
print(data_project_industrial['DATE_TIME'].dtypes)
print(data_project_industrial['DATE_TIME'][1])

#However, this is not enough, if I just put one argument like [1] the code just pick the first 
#value in the first column out. What I want to do is to have the hole row.
A=max(data_project_industrial['DATE_TIME'][1])
print(A)

#To start solving this issue, it is necessary to take a look at the entire 
#data in a organized plot

data_project_industrial.shape
data_project_industrial.head()

#As we saw previously, there is a 0 column in python, but the purpose is just to get the
#irradiation data, wich is in this case the numer 5, from wich I will need the maximum radiation number
data_project_industrial.iloc[:,5]
A=data_project_industrial.iloc[:,5]
B=max(A)
print(B)



#Now I create an object to drop the irradiation column and then I will figure out how to plot it
#using the same algorithm I normaly use in matlab

DataFrame=data_project_industrial['IRRADIATION']
df=DataFrame
df

import matplotlib.pyplot as plt
import numpy as np
l=len(df)
x=np.arange(0,l, 1)
y=df
plt.plot(x,y)





#I tried for all means to adapt this code to my data, but for some reason the y.index is not callable. 
#I just want to put this:
# ymax = max(y)
#xpos = y.index(ymax)
#xmax = x[xpos]
#but it keeps saying, is not callable. Although this is a 2D array, that I ploted correctly, I don't find the
#mistake. I would really apprecciate if you lnow how. Thanks. I would like to make more interesting things but I 
#struggle with little details in python.

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

x=[1,2,3,4,5,6,7,8,9,10]
y=[1,1,1,2,10,2,1,1,1,1]
line, = ax.plot(x, y)

ymax = max(y)
xpos = y.index(ymax)
xmax = x[xpos]

ax.annotate('local max', xy=(xmax, ymax), xytext=(xmax, ymax+5),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )

ax.set_ylim(0,20)
plt.show()





import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

x=[1,2,3,4,5,6,7,8,9,10]
y=[1,1,1,2,10,2,1,1,1,1]
line, = ax.plot(x, y)

ymax = max(y)
xpos = y.index(ymax)
xmax = x[xpos]

ax.annotate('local max', xy=(xmax, ymax), xytext=(xmax, ymax+5),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )

ax.set_ylim(0,20)
plt.show()
x
y