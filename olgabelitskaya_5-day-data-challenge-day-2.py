import numpy as np 

import pandas as pd 

import seaborn as sns

import pylab as plt



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
cereal = pd.read_csv("../input/cereal.csv")

cereal.head().T
cereal.describe().T
plt.style.use('seaborn-whitegrid')

plt.style.use('seaborn-pastel')

plt.figure(figsize=(15,5))

sns.distplot(cereal.calories, bins=50, kde = False)

plt.xlabel('Calories', fontsize=15)

plt.ylabel('Frequency', fontsize=15)

plt.title('Distribution of the variable "calories"', fontsize=20);