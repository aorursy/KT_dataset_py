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
# Reading the data from the csv

det = pd.read_csv('../input/cereal.csv')

det.head()
#Summary of the data

det.describe()
#importing Seaborn for visualisations

import seaborn as sns
# I want to plot a histogram of cthe fat contents of different cereals ;)

# I place it in a variable x

x = det['rating']
#Plotting the rating on a histogram

sns.distplot(x, kde = False, bins = 25).set_title("Cereal Rating")
# Getting the records with ratings higher than 90 

det[det['rating'] > 90]
# We're going to do a t-test, but before that, we need to import the t-test function!

from scipy.stats import ttest_ind
pr = det['protein']

fat = det['fat']
import matplotlib.pyplot as plt

#sns.distplot(pr, kde = False, bins = 25).set_title("Cereal Protein Content")

plt.hist([pr,fat], color=['brown','g'], alpha=0.5, label = ["Protien",'Fat'])

plt.ylabel("No. of Cereals",)

plt.xlabel("Nutrient Amount")

plt.legend()
ttest_ind(fat,pr, equal_var=False)