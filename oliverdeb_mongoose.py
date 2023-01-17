# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))

df = pd.read_csv('../input/insurance3r2.csv')
# print(df.keys())

# df.groupby('sex').count()

# test
print ("Gender counts, 0=female, 1=male")
female = int (df.sex.value_counts()[0])
print (df.sex.value_counts())
print ("Prob Male: %.3f,\tProb Female: %.3f" % (1-(female/1338), (female/1338)))

print ("\n\nSmoker stats, 0=non-smoker, 1=smoker")
non_smoker = df.smoker.value_counts()[0]
print (df.smoker.value_counts())
print ("Prob Non-smoker: %.3f,\tProb smoker: %.3f" % ((non_smoker/1338), 1-(non_smoker/1338)))

print ("\n\nBMI data stats")
underweight = df.bmi.between(0,18.499999, inclusive=True).value_counts()[True]
normal = df.bmi.between(18.5, 24.999, inclusive=True).value_counts()[True]
overweight = df.bmi.between(25, 29.99999, inclusive=True).value_counts()[True]
obese = df.bmi.between(30, 10000, inclusive=True).value_counts()[True]
# print (underweight, normal, overweight, obese)
print ("underweight: \t%d,\t%.3f" % (underweight, underweight/1338))
print ("normal: \t%d,\t%.3f" % (normal, normal/1338))
print ("overweight: \t%d,\t%.3f" % (overweight, overweight/1338))
print ("obese: \t\t%d,\t%.3f" % (obese, obese/1338))

print ("\n\nAge stats")
underage = df.age.between(0,17.999, inclusive=True).value_counts().get(True, 0)
adult = df.age.between(18, 39.999, inclusive=True).value_counts().get(True, 0)
overadult = df.age.between(40, 59.999, inclusive=True).value_counts().get(True, 0)
old = df.age.between(60, 200, inclusive=True).value_counts().get(True, 0)
print ("0-18: \t%d,\t%.3f" % (underage, underage/1338))
print ("18-40: \t%d,\t%.3f" % (adult, adult/1338))
print ("40-60: \t%d,\t%.3f" % (overadult, overadult/1338))
print ("60-: \t%d,\t%.3f" % (old, old/1338))




# Any results you write to the current directory are saved as output.
