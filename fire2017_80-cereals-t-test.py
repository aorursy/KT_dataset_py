# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Import the libraries

from scipy.stats import ttest_ind #t-test package

from scipy.stats import probplot #qqplot package

import matplotlib.pyplot as plt #for qqplot

import pylab



#Load data

cereals = pd.read_csv("../input/cereal.csv")



#check the first fews lines of the data

cereals.head()
#plot a qqplot to check normality

probplot(cereals["sodium"], dist = "norm", plot = pylab)

#get the sodium for hot cereals

hotCereals = cereals["sodium"][cereals["type"] == "H"]

#get the sodium for cold cereals

coldCereals = cereals["sodium"][cereals["type"] == "C"]



#compare the two types of cereals

ttest_ind(hotCereals, coldCereals, equal_var=False)