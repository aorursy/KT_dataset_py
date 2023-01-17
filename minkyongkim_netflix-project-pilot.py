# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv("../input/results_cleaned - Sheet1.csv")
df

overall = df['Star rating (overall)']

ST = df['ST rating']

LT = df['LT rating']

time = df['Time']

entertainment = df['Entertainment']

relaxing = df['Relaxing']

meaning = df['Meaningful']

social = df['Social']

thought = df['Thought-provoking']

learning = df['Learning']
plt.scatter(ST, LT)

plt.xlabel('Short-Term Value Ratings')

plt.ylabel('Long-Term Value Ratings')
plt.scatter(ST,overall)

plt.xlabel("Short-term Value Ratings")

plt.ylabel("Overall Star Ratings")

plt.scatter(LT, overall)

plt.xlabel("Long-term Value Ratings")

plt.ylabel("Overall Star Ratings")

plt.scatter(LT, meaning)

plt.xlabel("Long-term value ratings")

plt.ylabel("Ratings of Movies on Meaningfulness")