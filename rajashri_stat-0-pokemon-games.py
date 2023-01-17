# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats import kurtosis

from sklearn import preprocessing 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
game = pd.read_csv("../input/Pokemon.csv")
game.head()
game.describe()
val = pd.value_counts(game['Type 1'].values, sort=True)
val
val.plot(kind = 'bar')
#We could see that resistance to water tops,followed by Normal,Bug,Grass and so on.
val2 = pd.value_counts(game['Type 2'].values,sort = True)
val2
val2.plot(kind = "bar")
#We have flying toping the chart here followed by close competition between Grounf,Poison & Psychic.
#Examining distribution of total
#The idea here is to know whether there are influential observations in the variable like ones which are abnormal and how do they affect the model/analysis.
#https://www.khanacademy.org/math/cc-eighth-grade-math/cc-8th-data/cc-8th-interpreting-scatter-plots/a/outliers-in-scatter-plots
    

plt.scatter(game.Total,game.Name)
plt.subplots_adjust(bottom=0.5, right=1.5, top=40)
##Looks like we have few influential observations,

fig=plt.figure()
ax=fig.add_subplot(111)
# plot points inside distribution's width
plt.scatter(game.Total,game.Name)
plt.scatter(game.Name, game.Total<700, marker="s", color="#2e91be")
# plot points outside distribution's width

plt.subplots_adjust(bottom=0.5, right=1.8, top=10)
plt.show()
#examining health points or HP
game.drop(['#'], axis=1)
game['Id']=game.index
game.head()

HP = pd.value_counts(game['HP'].values,sort = "False")
HP.plot(kind = 'bar')
plt.subplots_adjust(bottom=0.5, right=2, top=5)
##The value of HP is skewed towards right

print("mean:",np.mean(HP))
print("var:",np.var(HP))
print("skewness:",skew(HP))
print("kurtosis:",kurtosis(HP))
#Examining Attack column

Attk = pd.value_counts(game['Attack'].values,sort = True)
Attk.plot(kind = 'bar')
plt.subplots_adjust(bottom = 0.5,right = 2,top = 5)
#This one again is skewed
print("mean:",np.mean(Attk))
print("var:",np.var(Attk))
print("skewness:",skew(Attk))
print("kurtosis:",kurtosis(Attk))
Defnc = pd.value_counts(game['Defense'].values,sort = True)
Defnc.plot(kind = 'bar')
plt.subplots_adjust(bottom = 0.5,right = 2,top = 5)

print("mean:",np.mean(Defnc))
print("var:",np.var(Defnc))
print("skewness:",skew(Defnc))
print("kurtosis:",kurtosis(Defnc))
Splatk = pd.value_counts(game['Sp. Atk'].values,sort = True)
Splatk.plot(kind = 'bar')
plt.subplots_adjust(bottom = 0.5,right = 2,top = 5)

print("mean:",np.mean(Splatk))
print("var:",np.var(Splatk))
print("skewness:",skew(Splatk))
print("kurtosis:",kurtosis(Splatk))
Spldef = pd.value_counts(game['Sp. Def'].values,sort = True)
Spldef.plot(kind = 'bar')
plt.subplots_adjust(bottom = 0.5,right = 2,top = 5)
print("mean:",np.mean(Spldef))
print("var:",np.var(Spldef))
print("skewness:",skew(Spldef))
print("kurtosis:",kurtosis(Spldef))
spd = pd.value_counts(game['Speed'].values,sort = True)
spd.plot(kind = 'bar')
plt.subplots_adjust(bottom = 0.5,right = 2,top = 5)

print("mean:",np.mean(spd))
print("var:",np.var(spd))
print("skewness:",skew(spd))
print("kurtosis:",kurtosis(spd))
gen = pd.value_counts(game['Generation'].values,sort = True)
gen.plot(kind = 'bar')
plt.subplots_adjust(bottom = 0.5,right = 2,top = 2)

legend = pd.value_counts(game['Legendary'].values,sort = True)
legend.plot(kind = 'bar')
plt.subplots_adjust(bottom = 0.5,right = 2,top = 2)