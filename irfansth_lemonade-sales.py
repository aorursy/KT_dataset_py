print("Irfan Lasman 5826801")
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# -------------------------------- Question 5.1 ----------------------------------------- #
df = pd.read_csv("../input/Lemonade.csv")
avg = np.mean(df)
avg1 = avg.Sales
print("Averages sales of lemonade is: "+str(avg1))
# -------------------------------- Question 5.4.1 ----------------------------------------- #
averagest = df[(df.Day == "Saturday") & df.Sales]
t = np.mean(averagest)
print("The average sales of Saturdays is = "+str(t.Sales))
# -------------------------------- Question 5.4.2 ----------------------------------------- #
averagesu = df[(df.Day == "Sunday") & df.Sales]
t1 = np.mean(averagesu)
print("The average sale of Sundays is = "+str(t1.Sales))
# -------------------------------- Question 5.2 ----------------------------------------- #
df[df.Sales < avg1]
# -------------------------------- Question 5.3 ----------------------------------------- #
temperature = df['Temperature']
sales = df['Sales']

ser =  pd.Series(index = sales,data=temperature)
df =ser.to_frame()

df.reset_index(inplace=True)
df.columns = ['sales','temperature']
df.plot(kind='scatter',x='sales',y='temperature')
plt.show()
