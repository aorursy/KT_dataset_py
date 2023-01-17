# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import pylab

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import warnings

warnings.filterwarnings("ignore")



# Any results you write to the current directory are saved as output.
# Loaded the dataset and did some cleaning.

h1b = pd.read_csv('../input/h1b_kaggle.csv', index_col = 0)

h2b = h1b[~h1b.YEAR.isnull()]

h3b = h2b[~h2b.EMPLOYER_NAME.isnull()]

h4b = h3b[~h3b.JOB_TITLE.isnull()]

h5b = h4b[~h4b.PREVAILING_WAGE.isnull()]

h5b['year'] = h5b.YEAR.astype('int')

del h5b['YEAR']

h5b['YEAR'] = h5b['year']

del h5b['year']

h6b = h5b[~h5b.SOC_NAME.isnull()]

np.sum(h5b.isnull())
summary1 = h5b['EMPLOYER_NAME'].groupby(h5b['EMPLOYER_NAME']).count()

summary2 = h5b['PREVAILING_WAGE'].groupby(h5b['EMPLOYER_NAME']).mean()

CompaniesWithMaximumH1B = summary1.sort_values(ascending = False).head(10).index
print(summary1.describe())

print("------------------Listing 10 companies with maximum H1B----------------------")

print(summary1[CompaniesWithMaximumH1B])

print("------------------Listing the average salary for those companies. ------------------")

print(summary2[CompaniesWithMaximumH1B])
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (15,6)
summary1[CompaniesWithMaximumH1B].plot()
# Same analysis with SOC

summary3 = h6b['PREVAILING_WAGE'].groupby(h6b['SOC_NAME']).count()

summary4 = h6b['PREVAILING_WAGE'].groupby(h6b['SOC_NAME']).mean()

maximum_SOC = summary3.sort_values(ascending = False).head(10).index



print(summary3.describe())

print();print()

print(summary3[maximum_SOC])

print();print()

print(summary4[maximum_SOC])
# Same analysis with worksite

summary5 = h6b['PREVAILING_WAGE'].groupby(h6b['WORKSITE']).count()

summary6 = h6b['PREVAILING_WAGE'].groupby(h6b['WORKSITE']).mean()

maximum_h1b_location = summary5.sort_values(ascending = False).head(10).index





print(summary5.describe())

print();print()

print(summary5[maximum_h1b_location])

print();print()

print(summary6[maximum_h1b_location])
summary7 = h1b['PREVAILING_WAGE'].groupby(h6b['YEAR']).count()

summary8 = h1b['PREVAILING_WAGE'].groupby(h6b['YEAR']).mean()

pylab.plot(summary7.index, summary7, 'co')

pylab.plot(summary8.index, summary8, 'x')