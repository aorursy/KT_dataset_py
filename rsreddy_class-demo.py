# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Load Data

cars=pd.read_csv('../input/Automobile_data.csv')
cars.head()

# Check Data Quality
cars.isnull().sum()
print(cars['make'].unique())
cars.describe()

cars[['engine-size','peak-rpm','curb-weight','horsepower']].hist(color='Y')
plt.show()
sns.countplot(cars['make'])
plt.xticks(rotation='vertical')
plt.show()
plt.figure(figsize=(11,4))
sns.heatmap(cars[['length','width','height','curb-weight','engine-size','horsepower','city-mpg','highway-mpg','price']].corr(),annot=True)
plt.show()
