# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

from scipy.stats import ttest_ind



print(check_output(["ls", "../input"]).decode("utf8"))

cereal = pd.read_csv("../input/cereal.csv")

#print(cereal)





print("calories vs. sodium")

print(ttest_ind(cereal["calories"], cereal["sodium"]))

print("weight vs. cups")

print(ttest_ind(cereal["weight"], cereal["cups"]))

print("carbo vs. sugars")

print(ttest_ind(cereal["carbo"], cereal["sugars"]))

print("sugars vs. fat")

print(ttest_ind(cereal["sugars"], cereal["fat"]))

print("sugars vs. weight")

print(ttest_ind(cereal["sugars"], cereal["protein"]))



plt.hist(cereal["calories"])

plt.hist(cereal["sodium"])

plt.title("Calories vs. Sodium")





#name

#mfr

#type

#calories

#protein

#fat

#sodium

#fiber

#carbo

#sugars

#potass

#vitamins

#shelf

#weight

#cups

#rating
