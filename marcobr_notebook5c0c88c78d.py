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
ice = pd.read_csv("../input/seaice.csv")
ice
nice = ice[(ice.Month==9) & (ice.Day==1) & (ice.hemisphere=='north')]

#nice.plot(x='Year', y='Extent', kind='scatter')

nice.Year.values
from sklearn import linear_model

print(nice.Year.values)

print(nice.Extent.values)

regr = linear_model.LinearRegression(fit_intercept=True)

regr.fit(nice.Year.values.reshape(-1,1), nice.Extent.values)

print(regr.coef_, regr.intercept_)
