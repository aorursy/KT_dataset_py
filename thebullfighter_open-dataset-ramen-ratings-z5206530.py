# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns
sns.set(style="white", color_codes=True)
import numpy as np
import math
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
plotly.__version__


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
ramen = pd.read_csv('../input/ramen-ratings.csv')
ramen.info()
#this is to view the columns in the dataset
ramen.columns
# this is to show the first 20 lines of the dataset
ramen.head(20)
# this shows the last 20 lines of the dataset
ramen.tail(20)
columns = ramen.columns.values
for col in columns:
    number = ramen[col].count()
    print(col, number/len(ramen), number)
# data types
ramen.dtypes
ramen['Stars'] = pd.to_numeric(ramen['Stars'], errors='coerce')
ramen.dtypes
gb_country = ramen.groupby('Country')
median_stars = gb_country['Stars'].median().sort_values(ascending = False)

number_per_country = gb_country.count()
print(number_per_country)
gb_country['Stars'].unique()
gb_country['Style'].unique()
gb_brand = ramen.groupby('Brand')
gb_brand['Stars'].describe()
ramen['Variety'].value_counts()
plt.figure(figsize=(20,10))
ramen['Style'].value_counts().plot(kind='bar')
plt.xlabel("Style")
plt.ylabel("Review")
plt.title('Ramen Ratings')
columns_of_interest = ['Variety', 'Stars']
two_columns_of_data = ramen[columns_of_interest]
two_columns_of_data.describe()
ramen['Stars'].value_counts().plot.area()
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(ramen.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)