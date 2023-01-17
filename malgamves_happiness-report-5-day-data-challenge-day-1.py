# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from pandas import DataFrame,Series

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Reading the data from the two CSV's

d2017 = pd.read_csv('../input/2017.csv')

d2016 = pd.read_csv('../input/2016.csv')
# Viewing the first few rows of the data set

d2017.head()
#Summersing the contents of the data set 

d2017.describe()