import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# set seed for reproducibility
np.random.seed(0) 
#import Computation Libraries
import pandas as pd
import numpy as np

from scipy.stats import ttest_ind # just the t-test from scipy.stats
from scipy.stats import probplot 

#import Visualization Libraries
import matplotlib.pyplot as plt
import pylab
import seaborn as sns
#read musesum csv
museums = pd.read_csv("/kaggle/input/museum-directory/museums.csv")
museums.shape
list(museums)
museums.sample(5)
museums['Museum Type'].unique()
plt.figure(figsize=(45,20))
sns.countplot(data=museums, x='Museum Type', palette='coolwarm')
plt.title('Counts of Muesum Type', fontsize=15)
plt.xlabel('Museum Type')
plt.ylabel("Count")
museums['Museum Type'].value_counts() *100 / museums.shape[0] 
museums = museums[['Museum Name','Museum Type', 'State (Administrative Location)', 'Income', 'Revenue']]
# get the number of missing data points per column
missing_values_count = museums.isnull().sum()

# look at the # of missing points
missing_values_count
# just how much data did we lose?
rows_with_na_dropped = museums.dropna()

print("Rows in original dataset: %d \n" % museums.shape[0])
print("Rows with na's dropped: %d" % rows_with_na_dropped.shape[0])
# remove all the rows that contain a missing value
museums = museums.dropna()
museums.shape
museums.isnull().sum()
plt.hist(museums['Revenue'], alpha=0.5, label='cold')
# plot a plot to check normality. If the varaible is normally distributed, most of the points 
# should be along the center diagonal.
probplot(museums["Revenue"], dist="norm", plot=pylab)
museumsreduced = museums[museums.Revenue < 500000]
museumsreduced = museums[museums.Revenue > 0]

entries_removed = museums.shape[0] - museumsreduced.shape[0]

print("Rows in dataset with nulls removed: %d \n" % museums.shape[0])

print("Removed museums with revenue that are under 500000: %d \n" % entries_removed)
museums = museums[museums.Revenue < 500000]
museums = museums[museums.Revenue > 0]

print("Current number of rows: %d \n" % museums.shape[0])
plt.hist(museums['Revenue'], alpha=0.5, label='cold')
# plot a plot to check normality. If the varaible is normally distributed, most of the points 
# should be along the center diagonal.
probplot(museums["Revenue"], dist="norm", plot=pylab)
museums.shape
museums.head()
zoos = museums[museums['Museum Type'] =="ZOO, AQUARIUM, OR WILDLIFE CONSERVATION"]
notzoos =museums[museums['Museum Type']!="ZOO, AQUARIUM, OR WILDLIFE CONSERVATION"]
ttest_ind(zoos.Revenue,notzoos.Revenue,equal_var=False)