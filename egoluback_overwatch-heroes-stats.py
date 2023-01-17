# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

path = ""

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        path = os.path.join(dirname, filename)

        

data = pd.read_csv(path, encoding = "ISO-8859-1")



print(data)



# Any results you write to the current directory are saved as output.
samples = data.sample(n = 50)



max_val = 50



pick_rate = (np.array(samples["Pick_rate"]) * max_val) / max(np.array(samples["Pick_rate"]))

win_rate = (np.array(samples["Win_rate"]) * max_val) / max(np.array(samples["Win_rate"]))

fire_rate = (np.array(samples["On_fire"]) * max_val) / max(np.array(samples["On_fire"]))

tie_rate = (np.array(samples["Tie_Rate"]) * max_val) / max(np.array(samples["Tie_Rate"]))
figure = plt.figure()

ax = figure.add_subplot()



figure.set_facecolor('white')



ax.plot(sorted(samples["Hero"]), pick_rate, color = "red", label = "Pick_rate")

ax.plot(sorted(samples["Hero"]), win_rate, color = "blue", label = "Win_rate")

ax.plot(sorted(samples["Hero"]), fire_rate, color = "green", label = "On_fire")

ax.plot(sorted(samples["Hero"]), tie_rate, color = "purple", label = "Tie_Rate")



ax.set_facecolor('white')



figure.set_figwidth(40)

figure.set_figheight(7)



plt.legend()

plt.show()



plt.savefig("image.png", dpi=figure.dpi)