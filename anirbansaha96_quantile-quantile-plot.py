dataset1=rnorm(1000) #This generates a dataset with 1000 points from the standard normal distribution
qqnorm(dataset1)     #This plots the Normal QQ Plot for the dataset1

qqline(dataset1)
dataset2=rcauchy(500, location=0, scale=1)    #This generates a dataset2 with 500 points from the Cauchy Distribution
qqnorm(dataset2)

qqline(dataset2)
qqplot(x=dataset1, y=dataset2)
import numpy as np

dataset1=np.random.normal(0,1,1000) #Generates a dataset1 with 1000 points drawn  from a Normal(0,1) distribution

dataset2 = np.random.standard_cauchy(500) #Generates dataset2 with 1000 points drawn  from a Cauchy distribution with mode=0
import statsmodels.api as sm

import pylab as py
sm.qqplot(dataset1,line='45')

py.show()
sm.qqplot(dataset2,line='45')

py.show()