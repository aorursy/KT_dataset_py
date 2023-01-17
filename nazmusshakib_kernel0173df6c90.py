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

import numpy as np 
import pandas as pd
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import plotly.plotly as plty
import plotly.graph_objs as go
from sklearn import datasets, linear_model
import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols

dataframe1 = pd.read_csv("../input/Untitled Spreadsheet_Sheet1.csv")
print(dataframe1)


print(dataframe1['Year 1 Scores'].describe())
print(dataframe1['Year 2 Scores'].describe())
print(dataframe1['Year 3 Scores'].describe())

from pandas.tools import plotting
plotting.scatter_matrix(dataframe1[['Year 1 Scores', 'Year 2 Scores', 'Year 3 Scores']])   

dataframe1.boxplot()
dataframe1.hist()

model = ols("'Year_1_score' ~ 'Year_3_score'", data=dataframe1).fit()
print(model.summary()) 
