# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
# Importing data set
Iris = pd.read_csv("../input/Iris_data.csv",na_values="n/a")
# A sneak into the dataset
Iris.head()
# The task to draw a scatter plot of petal length and petal width data of versicolor species.
# let us extract the Iris-versicolor data into a saperate dataset

versicolor_data = Iris[Iris.Species == "Iris-versicolor"]
versicolor_data.head()
#Extract the petal length and petal width into two saperate numpy arrays

versicolor_petallength = versicolor_data.iloc[:,2]
versicolor_petalwidth = versicolor_data.iloc[:,3]
# Converting these two into numpu arrays
versicolor_petallength  = np.array(versicolor_petallength)
versicolor_petalwidth = np.array(versicolor_petalwidth)
# plotting a scatter plot
plt.plot(versicolor_petallength,versicolor_petalwidth,marker = ".",linestyle = "none")
plt.style.use('fivethirtyeight')
