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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



bad_food = pd.read_csv('../input/CAERS_ASCII_2004_2017Q2.csv')

bad_food.head()
bad_food['CI_Age at Adverse Event'][bad_food['RA_Report #'] == 65333]
bad_food.info()
bad_food.describe()
# Drop the rows with outrageous ages

bad_food.drop(bad_food[bad_food['CI_Age at Adverse Event']>120].index, inplace=True)
# Get the rows where CI_Age at Adverse Event does not have a NaN value

no_nan_age = bad_food[pd.notnull(bad_food['CI_Age at Adverse Event'])]

no_nan_age.head()
age_hist = sns.distplot(no_nan_age['CI_Age at Adverse Event'], bins=25, kde=False)

axes = age_hist.axes

age_hist.axes.set_title('Age at Adverse Event')

age_hist.set_xlabel('Age')