import numpy as np

import pandas as pd

licensees = pd.read_csv("../input/federal-firearm-licensees.csv")[1:]

licensees.head(10) # Exploring the d

licensees.info()
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

licensees["Premise Zip Code"].value_counts().plot.hist(bins=30) #value counts displays unique values arranged in descending order of frequency.
licensees["Premise Zip Code"].value_counts().mean()
X= licensees["Premise Zip Code"].value_counts()
import numpy as np

import scipy.stats as stats



def t_value(X,h_0):

    se = np.sqrt(np.var(X) / len(X))

    return (np.mean(X) - h_0) / se



def p_value(t):

    return stats.norm.sf(abs(t))*2 #2-sided p-value so we multiple the test statistic by 2

      

t = t_value(X,2.75)    

p = p_value(t)

t,p

#A simpler method of conducting the above t-test

import scipy.stats as stats

stats.ttest_1samp(a=X,popmean=2.75)