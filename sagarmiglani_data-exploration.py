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
##import (library) as (give the library a nickname/alias)

import matplotlib.pyplot as plt

import pandas as pd #this is how I usually import pandas



from numpy.random import randn

from matplotlib import rcParams

import seaborn as sb



# Enable inline plotting

%matplotlib inline
rawdf= f = pd.read_csv("../input/SolarPrediction.csv")
rawdf.head(5)
rawdf.shape
#no missing values

rawdf.isnull().sum()
# check for duplicate values

rawdf.duplicated().sum()
#dependent variable 



rawdf['Radiation'].describe()


rawdf['Radiation'].hist()

plt.show()