# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
def show_summary(df):
    print("Train file rows and columns are : ", df.shape)
    print("---data header---")
    display(df.head())
    
    print("---data little info---")
    display(df.info())
    
    print("---data describe---")
    display(df.describe(include="all"))    
show_summary(train)
show_summary(test)
# histogram
plt.figure(figsize = (14, 8))
sns.distplot(train.target.values, axlabel='target')
plt.show() 
# histogram
plt.figure(figsize = (14, 8))
sns.distplot(np.log1p(train.target.values), axlabel='target',  bins=100, kde=False)
plt.show() 
train.columns
from sklearn.ensemble import ExtraTreesRegressor
# Build a forest and compute the feature importances
y = np.log1p(train.target.values)
X = train.drop(['ID', 'target'], axis=1)

forest = ExtraTreesRegressor(n_estimators=250,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
plot_num = 30
for f in range(0, plot_num):
    print("%d. %s (%f)" % (f + 1, X.columns[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(0, plot_num), importances[indices[0:plot_num]],
       color="r", yerr=std[indices[0:plot_num]], align="center")
plt.xticks(range(0, plot_num), indices)
plt.xlim([-1, plot_num])
plt.show()
