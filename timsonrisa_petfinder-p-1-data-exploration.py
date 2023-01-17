import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))
breeds = pd.read_csv("../input/breed_labels.csv")
print("{}\n\n{} ".format(breeds.head(2), breeds.tail(2)))
print("\nIndex {} is BreedId={}: {}".format(240, breeds.BreedID[240], breeds.BreedName[240]))
colors = pd.read_csv("../input/color_labels.csv")
print("{}".format(colors.head(7)))
states = pd.read_csv("../input/state_labels.csv")
print("{}\n\n{} ".format(states.head(2), states.tail(2)))
train = pd.read_csv('../input/train/train.csv')
test = pd.read_csv('../input/test/test.csv')
print("Train data has {} entries with {} columns".format(train.shape[0], train.shape[1]))
print("Test data has {} entries with {} columns".format(test.shape[0], test.shape[1]))
train.head(1)
plt.figure(figsize=(8,6))
sns.countplot(x='Type', hue='Gender', data=train)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Type', fontsize=12)
plt.grid()
plt.figure(figsize=(10,5))
sns.violinplot(x='Type', y='AdoptionSpeed', hue="Gender", data=train)
plt.xlabel('Type: Cat or Dog', fontsize=12)
plt.ylabel('Adoption Speed', fontsize=12)
plt.grid()
plt.show()