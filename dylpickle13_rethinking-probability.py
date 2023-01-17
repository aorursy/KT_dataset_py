import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

import os

import seaborn as sns

import matplotlib.pyplot as plt



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



seven_balls = pd.read_csv("../input/independent-probability-dice-coins/7balls.csv").T

coins_data = pd.read_csv("../input/independent-probability-dice-coins/coinsdata.csv").T

dices_data = pd.read_csv("../input/independent-probability-dice-coins/dicesdata.csv").T

print("Setup ready to go!")
seven_balls
for i in range(10):

    seven_balls[i] = LabelEncoder().fit_transform(seven_balls[i])

seven_balls
plt.figure(figsize=(12,6))

plt.title("Distribution of balls by color")

plt.ylabel("Ball color (follow legend)")

sns.scatterplot(x=seven_balls.index, y=seven_balls[0])
plt.figure(figsize=(12,6))

plt.title("Distribution of balls by color")

plt.ylabel("Ball color (follow legend)")

sns.scatterplot(x=seven_balls.index, y=seven_balls[1])
coins_data
for i in range(10):

    coins_data[i] = LabelEncoder().fit_transform(coins_data[i])

coins_data
plt.figure(figsize=(12,6))

plt.title("Distribution of coin flips")

plt.ylabel("Head or Tail (1 or 0)")

sns.lineplot(data=coins_data[0])
plt.figure(figsize=(12,6))

plt.title("Distribution of coin flips")

plt.ylabel("Head or Tail (1 or 0)")

sns.lineplot(data=coins_data[1])
dices_data
plt.figure(figsize=(12,6))

plt.title("Distribution of die rolls")

plt.ylabel("Die outcome")

sns.scatterplot(x=dices_data.index, y=dices_data[0])
plt.figure(figsize=(12,6))

plt.title("Distribution of die rolls")

plt.ylabel("Die outcome")

sns.scatterplot(x=dices_data.index, y=dices_data[1])