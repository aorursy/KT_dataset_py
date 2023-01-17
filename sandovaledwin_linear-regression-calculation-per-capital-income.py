# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import linear_model
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Youtube Tutorial:
# https://www.youtube.com/watch?v=8jazNUpO3lQ
# Dataset:
# https://data.worldbank.org/indicator/NY.GNP.PCAP.CD?locations=CA&view=chart

df = pd.read_csv( '../input/canada_per_capita_income.csv' )

df
%matplotlib inline
plt.xlabel( 'per capita income (US$)' )
plt.ylabel( 'year' )
plt.scatter( df[ 'year' ],df[ 'per capita income (US$)' ], color='red', marker='+' )

# Create linear regression object
reg = linear_model.LinearRegression()
reg.fit( df[[ 'year' ]], df[ 'per capita income (US$)' ] )
print( "Predicting the Canada per capita income for the 2020 year:" )
reg.predict( [ [ 2020 ] ] )