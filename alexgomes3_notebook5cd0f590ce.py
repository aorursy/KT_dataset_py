import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from scipy import stats

from matplotlib import pyplot as plt

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/cereal.csv')

print(df)

print(df.describe())



ldf_mf = [df[df.shelf == i] for i in df.shelf.unique()]
plt.figure(1)

plt.subplot(121)

plt.hist(df.sugars)

plt.subplot(122)

plt.hist(df.rating)

plt.show()



# Correlated features?

plt.figure(2)

plt.scatter(*df[["sugars","rating"]].values.transpose())

plt.xlabel('Sugars')

plt.ylabel('Rating')

plt.show()
plt.figure(3)

plt.subplot(131)

plt.hist(ldf_mf[0].sugars)

plt.title('Shelf = ' + str(ldf_mf[0].shelf.iloc[0]))

plt.subplot(132)

plt.hist(ldf_mf[1].sugars)

plt.title('Shelf = ' + str(ldf_mf[1].shelf.iloc[0]))

plt.subplot(133)

plt.hist(ldf_mf[2].sugars)

plt.title('Shelf = ' + str(ldf_mf[2].shelf.iloc[0]))

plt.show()
print(stats.ttest_ind(ldf_mf[0].sugars, ldf_mf[1].sugars, equal_var = False))

print(stats.ttest_ind(ldf_mf[1].sugars, ldf_mf[2].sugars, equal_var = False))