# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# create data frame using read_csv

case_c = pd.read_csv('/kaggle/input/group-c-data/analysis_data_test.csv')

# display first 10 rows

case_c.head(10)
# reshape dataframe into pivot table

d = case_c.pivot(index = 'x_value', columns = 'n_value', values = 'avg_peak_bed_usage')

# set visualization features like title, axis names, and legend name

d.columns.name = 'Initial Infections'

plot1 = d.plot(title = 'Policy Strictness v. Hospital Bed Demand')

plot1.set(xlabel = 'Maximum Allowed Group Size', ylabel = 'Hospital Bed Demand')
# reshape dataframe into pivot table

f = case_c.pivot(index = 'x_value', columns = 'n_value', values = 'success_rate')

# set visualization features like title, axis names, and legend name

f.columns.name = 'Initial Infections'

plot2 = f.plot(title = 'Policy Strictness v. Simulation Success Rate')

plot2.set(xlabel = 'Maximum Allowed Group Size', ylabel = 'Success Rate')
# find subset of dataframe using conditional formatting 

s = case_c[(case_c.success_rate >= 0.9) & (case_c.n_value == 2)]

# display last row to show highest x_value that satisfies threshold

s.tail(1)
# find subset of dataframe using conditional formatting 

s = case_c[(case_c.success_rate >= 0.9) & (case_c.n_value == 20)]

# display last row to show highest x_value that satisfies threshold

s.tail(1)
# find subset of dataframe using conditional formatting 

s = case_c[(case_c.success_rate >= 0.9) & (case_c.n_value == 40)]

# display last row to show highest x_value that satisfies threshold

s.tail(1)