import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns







from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



cereal =pd.read_csv("../input/cereal.csv")

cereal.head()
cereal.describe()
cereal.shape
num_bins = 10





fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,15))

ax0, ax1, ax2, ax3 = axes.flatten()



ax0.hist(cereal['sugars'], num_bins, alpha=0.7, label=["sugars"], edgecolor = 'black')

ax0.set_title("Grams of sugars")



ax1.hist(cereal['sodium'], num_bins, alpha=0.7, label=["sodium"], edgecolor = 'black')

ax1.set_title("Milligrams of sodium")



ax2.hist(cereal['protein'], num_bins, alpha=0.7, label=["protein"], edgecolor = 'black')

ax2.set_title("Grams of protein")



ax3.hist(cereal['fat'], num_bins, alpha=0.7, label=["fat"], edgecolor = 'black')

ax3.set_title("Grams of fat")





plt.show()
from scipy import stats

from scipy.stats import ttest_ind, norm, skew

cereal_ttest = cereal[['type', 'sugars', 'sodium']]

cereal_ttest.head(10)
Suger_cold = cereal_ttest[cereal_ttest['type'] == "C" ]['sugars']

Suger_hot = cereal_ttest[cereal_ttest['type'] == "H" ]['sugars']

print("Standard deviation for the suger content of hot cereal is {:.3f} and std for the suger content of cold cereal is {:.3f}".

     format(Suger_hot .std(), Suger_cold.std()))
print("Mean of the suger content of hot cereal is {:.3f} and mean for the suger content of cold cereal is {:.3f}".

     format(Suger_hot .mean(), Suger_cold.mean()))
print("Sample size of hot cereal is {:.3f} and cold cereal is {:.3f}".

     format(len(Suger_hot), len(Suger_cold)))
ttest_ind(Suger_cold, Suger_hot, axis=0, equal_var=False)
num_bins = 20

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,8))

ax0, ax1 = axes.flatten()



ax0.hist(cereal_ttest[cereal_ttest['type'] == "C" ]['sugars'], num_bins, alpha=0.7, 

         label=[" Cold Cereal group"], edgecolor = 'black')

ax0.set_title("Cold Cereal group sugar content")



ax1.hist(cereal_ttest[cereal_ttest['type'] == "H" ]['sugars'], num_bins, alpha=0.7, 

         label=["Hot Cereal group"], edgecolor = 'black')

ax1.set_title("Hot Cereal group sugar content")

plt.show()
sodium_cold = cereal_ttest[cereal_ttest['type'] == "C" ]['sodium']

sodium_hot = cereal_ttest[cereal_ttest['type'] == "H" ]['sodium']

print("Standard deviation for the sodium content of hot cereal is {:.3f} and std for the sodium content of cold cereal is {:.3f}".

     format(sodium_cold .std(), sodium_hot.std()))
ttest_ind(sodium_cold, sodium_hot, axis=0, equal_var=False)
num_bins = 10

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,8))

ax0, ax1 = axes.flatten()



ax0.hist(cereal_ttest[cereal_ttest['type'] == "C" ]['sodium'], num_bins, alpha=0.7, 

         label=[" Cold Cereal group"], edgecolor = 'black')

ax0.set_title("Cold Cereal group sodium content")



ax1.hist(cereal_ttest[cereal_ttest['type'] == "H" ]['sodium'], num_bins, alpha=0.7, 

         label=["Hot Cereal group"], edgecolor = 'black')

ax1.set_title("Hot Cereal group sodium content")

plt.show()