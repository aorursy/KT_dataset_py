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
%matplotlib inline
import matplotlib.pyplot as plt
foodFacts=pd.read_csv("../input/FoodFacts.csv",low_memory=False)
#print(foodFacts.head(3))
foodFacts.info()
foodFacts.describe()
#searching for countries with highest carbon footprint wrt food
df=foodFacts[['countries','carbon_footprint_100g']]
df=df.dropna()
print(df.shape)
print(df.head(20))
data=df['countries'].value_counts(sort=True,dropna=False)