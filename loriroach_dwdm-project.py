# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inlinedata3.columns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_path='/kaggle/input/consumer/cacSurveys3May2016-withColumnNames.csv'

data = pd.read_csv(data_path)

data
from pandasql import sqldf
data.columns
data3= data.copy()

print(data3)
data3.columns
data.columns
data.shape
data = data.drop(columns="Column 13")

#this drops column 13 that was not needed
data.columns
data3 = data3.drop(columns= "Column 13")
data3.rename(columns = {'Column 1':'ID'},inplace=True)

print(data3.columns)