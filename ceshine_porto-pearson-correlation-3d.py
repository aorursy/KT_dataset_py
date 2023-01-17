import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

train.head()
sns.set(style="white")



# Compute the correlation matrix

corr = train.corr()



# Set self-correlation to zero to avoid distraction

for i in range(corr.shape[0]):

    corr.iloc[i, i] = 0



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

# cmap = sns.diverging_palette(220, 10, as_cmap=True)

cmap = sns.color_palette("RdBu_r", 20)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, cmap=cmap, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr.iloc[1:40, 1:40], cmap=cmap, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()