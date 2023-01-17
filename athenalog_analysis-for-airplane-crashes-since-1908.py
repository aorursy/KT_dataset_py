



import numpy as np

import pandas as pd 

from matplotlib import pyplot as plt

import matplotlib

from sklearn.linear_model import LinearRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

frame=pd.read_csv('../input/3-Airplane_Crashes_Since_1908.txt',sep=',')
frame.head()
frame['Date'] = pd.to_datetime(frame['Date'])
frame['year'] = frame['Date'].dt.year
frame['year'].head()
from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns
plt.figure(figsize=(45,10))

sns.barplot('year','Aboard',data=frame)
plt.figure(figsize=(45,10))

sns.barplot('year','Fatalities',data=frame)
plt.figure(figsize=(45,10))

sns.stripplot('year','Fatalities',data=frame)
# Total Aboard and Fatalities plot - for each year

plt.figure(figsize=(100,10))



#Plot 1 - background - "total" (top) series

sns.barplot('year', 'Aboard',data=frame, color = "blue")



#Plot 2 - overlay - "bottom" series

bottom_plot = sns.barplot('year', 'Fatalities',data=frame, color = "red")





bottom_plot.set_ylabel("mean(Fatalities) and mean(Aboard)")

bottom_plot.set_xlabel("year")
