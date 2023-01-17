import pandas as pd

import os 



file = "../input/responses.csv"

df = pd.read_csv(r'../input/young-people-survey/responses.csv')



df.describe()
# First let's to check if there are nan values, histograms don't allow nan values.

print(len(df.Dance))

print(len(df.Dance.dropna()))
import matplotlib.pyplot as plt

import numpy as np



# Remove nan with dropna()

x = df.Dance.dropna()



fig = plt.figure()

ax = fig.add_subplot(111)

bins = 9

ax.hist(x, bins, color='blue')

plt.title('Dance', fontsize = 22)

print()
import pandas as pd

columns = pd.read_csv("../input/young-people-survey/columns.csv")

responses = pd.read_csv("../input/young-people-survey/responses.csv")