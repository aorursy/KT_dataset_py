# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns #data visalization

%matplotlib inline 
# we load our tips dataset from seaborn

tips = sns.load_dataset('tips')

# we can check the first 5 values of our dataset

tips.head()
# we do the same for flights dataset

flights = sns.load_dataset('flights')

flights.head()
## first we can convert the tips dataset into matrix form with data correlation

tc = tips.corr()
# we can have our first matrix plot for tips correlation

sns.heatmap(tc)
## you can aslo add the annotations

## it annotates the actual numerical values that belong to the cell

sns.heatmap(tc, annot=True)
## you can also change the colormap

sns.heatmap(tc, annot=True, cmap='viridis')
# since the data is currently not in matrix form we can convert it into a matrix form by making a pivot table

fp = flights.pivot_table(index='month',columns='year',values='passengers')
fp.head()
# we can have our heatmap

sns.heatmap(fp)
# we can change the color map again

sns.heatmap(fp, cmap='viridis')
# we can add line colors and linewidth to make our separate our cells and make distinguishable

sns.heatmap(fp, linecolor='white', linewidths=3)
sns.clustermap(fp)
## We can also normalize the scale by passing the standard_scale argument

sns.clustermap(fp, cmap='viridis', standard_scale=1)