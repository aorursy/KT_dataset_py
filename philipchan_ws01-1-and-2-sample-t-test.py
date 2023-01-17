import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import stats #for t-tests

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
dat=pd.read_csv("../input/heightmenwomen.csv",names=['men','women'],delim_whitespace=True) #space delimited csv
print(dat.head())
print("Sample size is {0}".format(len(dat)))

#Question 2
#mean height of women
round(dat.women.mean(),3)
#Question 3
print("Variance of height of women is {0:.3f}".format(dat.women.var())+"cm")
print("Sd of height of women is {0:.3f}".format(dat.women.std()) +"cm")
print("Sd of height of men is {0:.3f}".format(dat.men.std())+"cm")
#Question 4
# Computing sd of height of women manually
var=sum((dat.women-dat.women.mean())**2)/(len(dat)-1)
sd=var**0.5
print("Sd of height of women is {0:.3f}".format(sd) +"cm")

stats.ttest_1samp(dat.women,163)
stats.ttest_ind(dat.women,dat.men,equal_var=True)