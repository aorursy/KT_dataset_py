# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

from sklearn.decomposition import PCA #principal component analysis module

from sklearn.cluster import KMeans #KMeans clustering

import matplotlib.pyplot as plt #Python defacto plotting library

import seaborn as sns # More snazzy plotting library

%matplotlib inline

# Any results you write to the current directory are saved as output.
hr=pd.read_csv('..//input/HR_comma_sep.csv')

hr.shape
hr.head(10)
hr.corr()

"""

1. satisfaction_level and left are negatively correlated so low satisfaction is equal to high turnover?

2. number_projects and average_monthly_hours are positively correlated so more projects leads to more hours



"""
correlation=hr.corr()

plt.figure(figsize=(10,10))

sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')



plt.title('Correlation between different fearures')