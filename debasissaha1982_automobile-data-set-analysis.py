# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%pylab inline

import pandas as pd

from pandas import Series, DataFrame 
# Load the dataset

cars=pd.read_csv('../input/Automobile_data.csv')

# List the available columns

#cars.columns

list(cars)
#Get the dataTypes

cars.dtypes
#Checking the null value in the data set

cars.isnull().sum()
#Describe the data set

cars.describe().round(2)