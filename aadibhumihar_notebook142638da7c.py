# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd

test = pd.read_csv("../input/arrests.csv")

test.head()
test
test[test.Border=='Coast']
from matplotlib import pylab as plt

zone_grouping = test.groupby('Border').mean()

zone_grouping['2010 (All Illegal Immigrants)'].plot.bar()

plt.show()
zone_grouping = test.groupby('Border').mean()

zone_grouping['2016 (Mexicans Only)'].plot.bar()

plt.show()
test
zone_grouping = test.groupby('Border').mean()

zone_grouping
test[(test.Sector=='All') | (test.Sector=='NaN')].values
test['Sector'].fillna('All', inplace=True)

test_m = test[(test.Sector=='All')].values
test_m
test_m[0]
test_m1 = np.delete(test_m, [1,2], axis=1)
test_m1
year_c = np.array(np.arange(2000,2017))
year_c