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
# load the dataset and understand the data

heart_df = pd.read_csv(r'../input/heart.csv')

heart_df.head()
# pandas describe function helps you to understand statistical properties of the dataset

heart_df.describe()



# look on mean, max, min and find out outliers etc. it gives a quick glance but

# but misses many properties - NAN, missing values etc
# Info function helps in knowing the total values, nan etc,

heart_df.info()
# you can use pandas profiling for robust and more details analyiss of the data

# you need to install pandas-profiling using pip 

# import the pandas_profiling 

# to make report you can use ProfileReport

from pandas_profiling import ProfileReport

ProfileReport(heart_df)



# it gives whole report of your dataset

# and very useful for first analysis
# you can save this work by as an image / html page 
