# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd

import plotly.plotly as py

import plotly.figure_factory as fs

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Get current size

fig_size = plt.rcParams["figure.figsize"]

df = pd.read_csv("../input/Suicide Statistics in Indian States.csv")

# Prints: [8.0, 6.0]

print ("Current size:", fig_size)

 

# Set figure width to 12 and height to 9

fig_size[0] = 24

fig_size[1] = 9

plt.rcParams["figure.figsize"] = fig_size





# Any results you write to the current directory are saved as output.
import numpy as np

plt.xlabel("States")

plt.ylabel("No. of Death")

plt.title("Male data")

male = df['Males Involved'].values

States = df['State/UT/City'].values

female = df['Female Involved'].values

ax = np.arange(len(States))

bar_width = 0.5

plt.bar(ax,male,width=bar_width, color='g', align = 'edge', label='Male')

plt.bar(ax+bar_width,female,width=bar_width, color='b', align = 'edge', label='Female')

plt.xticks(ax+0.25,df['State/UT/City'].values )

plt.xticks(rotation=45, ha = "right")

plt.legend()

plt.rcParams.update({'font.size': 24})

plt.show()