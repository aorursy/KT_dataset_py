import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
import os
import seaborn as sns
print(os.listdir())
df = pd.read_csv("../input/kepler.csv", error_bad_lines=False)
pd.set_option('display.max_columns', None)
df.head(20)
df.columns
plt.figure(figsize=(20,13))
sns.countplot(df["detection_type"])
plt.figure(figsize=(20,13))
sns.countplot(df["discovered"])
plt.figure(figsize=(20,13))
sns.countplot(df["star_mass"])
plt.show()
print("standart deviation: ",np.std(df["star_mass"]))
plt.figure(figsize=(20,13))
sns.countplot(df["star_radius"])
plt.figure(figsize=(40,13))
plt.scatter(df["star_mass"],df["star_radius"])