import numpy as np 

import pandas as pd 

import seaborn as sns

import pylab as plt

import scipy.stats 



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
cereal = pd.read_csv("../input/cereal.csv")

cereal.head().T
cereal['calories'].dtype
set(cereal['mfr'].values)
len(cereal['calories'][cereal['mfr']=='P'])
len(cereal['calories'][cereal['mfr']=='Q'])
scipy.stats.ttest_ind(cereal['calories'][cereal['mfr']=='P'], 

                      cereal['calories'][cereal['mfr']=='Q'], 

                      axis=0, equal_var=False)

# As it could be expected we do not reject the null hypothesis 

# "two independent samples (from two manufacturers) have identical average values (calories).

# It's true for any level of confidence (0.1, 0.05, etc.)
plt.style.use('seaborn-whitegrid')

plt.style.use('seaborn-pastel')

plt.figure(1, figsize=(15,5))



plt.subplot(121)

sns.distplot(cereal.calories[cereal['mfr']=='P'], bins=10, kde = False)

plt.xlabel('Calories', fontsize=15)

plt.ylabel('Frequency', fontsize=15)

plt.title('Calories, Manufacturer of cereal is Post', fontsize=20);

plt.subplot(122)

sns.distplot(cereal.calories[cereal['mfr']=='Q'], bins=10, kde = False)

plt.xlabel('Calories', fontsize=15)

plt.ylabel('Frequency', fontsize=15)

plt.title('Calories, Manufacturer of cereal is Quaker Oats', fontsize=20);