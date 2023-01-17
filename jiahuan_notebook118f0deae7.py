# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import pandas as pd

import statsmodels.formula.api as smf

import numpy as np

import matplotlib.pyplot as plt

from IPython.display import display



# Some appearance options.

%matplotlib inline

%pylab inline

pylab.rcParams['figure.figsize'] = (10, 6)

pd.set_option('display.max_rows', 21)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

data = pd.read_csv("../input/car_data.csv")

data = data.dropna()
for i in data.columns:

    data[i] = data[i].astype("category")

display(data.head())
category_col = ['make', 'fuel_type', 'aspiration', 'num_of_doors',

                'body_style', 'drive_wheels', 'engine_location', 

                'engine_type', 'num_of_cylinders','fuel_system']

numeric_col = ['wheel_base', 'length','width', 'height', 'curb_weight',

              'engine_size','compression_ratio', 'horsepower',

               'peak_rpm', 'city_mpg', 'highway_mpg', 'price']

for i in category_col:

    data[i] = data[i].astype("category")

for i in numeric_col:

    data[i] = pd.to_numeric(data[i],errors= "coerce")
for i in data.columns:

    print("%20s : %10s" %(i,data[i].dtype), ",", data.at[0,i])
formula = " price ~ fuel_type+aspiration+num_of_doors+body_style+drive_wheels+engine_location+wheel_base+length+width+height+curb_weight+engine_type+num_of_cylinders+engine_size+fuel_system+compression_ratio+horsepower+peak_rpm+city_mpg+highway_mpg"

model = smf.ols(formula=formula, data = data).fit()

model.summary()