# deleted the stuff that automatically gets added when you 

# start a new notebook (you don't actually need it :p)

import pandas as pd

import matplotlib.pyplot as plt



# removed the 'r' argument from read_csv

# For pandas read_csv function specifically, you don't need it

# b/c it will only go a read a copy of the file you send it. (It's

# different from Python's default file reading syntax, which is

# confusing)

data=pd.read_csv('../input/data.csv')

print(data['radius_mean'].describe())



plt.figure()

data['radius_mean'].hist()

plt.title('hist_tu')

#plt.show() # one nice thing about Kaggle is that is shows

# plots by default, so you don't need plt.show()