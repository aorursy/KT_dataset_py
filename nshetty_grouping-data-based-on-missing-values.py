# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import kagglegym



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
env = kagglegym.make()

observation = env.reset()

train = observation.train

print("Train has {} rows".format(len(train)))
train.describe()
unique_id = train["id"].unique()

print("There are {} unique ids".format(len(unique_id)))
train.groupby("id").count()
count = 0

missing_value = np.zeros((len(unique_id), train.shape[1]))

for item in unique_id:

    tmp_id = train[train["id"]==item]

    tmp_id = tmp_id.fillna(0)

    tmp_id = tmp_id.values

    missing_value[count] = np.sum(tmp_id, 0) == 0

    count = count + 1
missing_value # binary matrix indicating (1s) completely missing features for each id
missing_value.shape
unique_missing_value = np.vstack(set(map(tuple, missing_value)))
print("So the ids can be clustered into {} groups, based on completely missing features".format(unique_missing_value.shape[0]))
# create a frequency table of ids in each group

freq_missing_value = []

for row in missing_value:

    for index, row_unique in enumerate(unique_missing_value):

        if np.array_equal(row, row_unique):

            freq_missing_value.append(index)

            

group, id_count = np.unique(freq_missing_value, return_counts=True)            



plt.plot(id_count)

plt.xlabel("group")

plt.ylabel("frequency")

plt.show()
supergroup, group_count = np.unique(id_count, return_counts=True)



print(np.asarray((supergroup, group_count)).T)



plt.plot(supergroup, group_count, '-o', label="Original noisy data")

plt.xlabel("supergroup")

plt.ylabel("frequency")

plt.legend()

plt.show()
group_count = group_count*10 # rescale data to zoom into the long tail



from scipy.optimize import curve_fit

def func(x, a, b, c):

    return a * np.power(x,-b) + c



popt, pcov = curve_fit(func, supergroup, group_count, maxfev=2000)



print("Fit parameters are", popt) 

print("Error on parameters are", np.sqrt(np.diag(pcov)))



plt.plot(supergroup, group_count, 'ko', label="Original noisy data")

plt.plot(supergroup, func(supergroup, *popt), 'r-', label="Exponential fit")

plt.xlabel("supergroup")

plt.ylabel("frequency")

plt.legend()

plt.show()
plt.loglog(supergroup, group_count, 'ko', label="Original noisy data")

plt.loglog(supergroup, func(supergroup, *popt), 'r-', label="Fitted Curve")

plt.xlabel("supergroup")

plt.ylabel("frequency")

plt.legend()

plt.show()