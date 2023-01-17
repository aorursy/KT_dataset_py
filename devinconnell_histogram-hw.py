# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
passing = [6,2843,3764,3620,3692,4110,3529,4806,76,4398,3900,5235,4827,4343,4109,4770,3554,4577,4355,4057]
print ("number of seasons: ", len(passing))
print("range: ", np.min(passing), "-", np.max(passing))
print("passing yards per season: ", np.mean(passing))

print("median: ", np.median(passing))
from statistics import mode

print("mode: Does not exist")
plt.hist(passing,6)

plt.xlabel("Yards Passing")

plt.ylabel("N")

plt.title("Tom Brady Passing Yards Distribution")

plt.show()
import numpy as np
# Passing Yards Over Time



passing = [6,2843,3764,3620,3692,4110,3529,4806,76,4398,3900,5235,4827,4343,4109,4770,3554,4577,4355,4057]



season_number = np.arange(len(passing))

print(season_number)
import matplotlib.pyplot as plt

plt.plot(season_number,passing)

plt.xlabel("Season Number")

plt.ylabel("Passing Yards")

plt.title("Tom Brady's Passing Yards over the years")

plt.show()
brees_passing = [np.nan,221,3284,2108,3159,3576,4418,4423,5069,4388,4620,5476,5177,5162,4952,4870,5208,4334,3992,2979]

pmanning_passing = [4413,4131,4200,4267,4557,3747,4397,4040,4002,4500,4700,np.nan,4659,5477,4727,2249,np.nan,np.nan,np.nan,np.nan]

rothlisberger_passing = [np.nan,np.nan,np.nan,np.nan,2621,85,3513,3154,3301,4328,3200,4077,3265,4261,4952,3938,3819,4251,5129,351]

rivers_passing = [np.nan,np.nan,np.nan,np.nan,33,115,3388,3152,4009,4254,4710,4624,3606,4478,4286,4792,4386,4515,4308,4615]
brees_passing = [np.nan,221,3284,2108,3159,3576,4418,4423,5069,4388,4620,5476,5177,5162,4952,4870,5208,4334,3992,2979]

pmanning_passing = [4413,4131,4200,4267,4557,3747,4397,4040,4002,4500,4700,np.nan,4659,5477,4727,2249,np.nan,np.nan,np.nan,np.nan]

rothlisberger_passing = [np.nan,np.nan,np.nan,np.nan,2621,85,3513,3154,3301,4328,3200,4077,3265,4261,4952,3938,3819,4251,5129,351]

rivers_passing = [np.nan,np.nan,np.nan,np.nan,33,115,3388,3152,4009,4254,4710,4624,3606,4478,4286,4792,4386,4515,4308,4615]



plt.plot(season_number,passing,label = "Brady")

plt.plot(season_number,brees_passing,label = "Brees")

plt.plot(season_number,pmanning_passing,label = "P. Manning")

plt.plot(season_number,rothlisberger_passing,label = "Rothlisberger")

plt.plot(season_number,rivers_passing,label = "Rivers")

plt.legend()



plt.xlabel("Season Number")

plt.ylabel("Passing Yards")

plt.title("Top QB's Passing Yards Over Time")

plt.show()