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
df=pd.read_csv('/kaggle/input/social-network-ads/Social_Network_Ads.csv')
df.sample(5)
df.shape
import seaborn as sns
sns.scatterplot(df['Age'],df['EstimatedSalary'],hue=df['Purchased'])
df.isnull().sum()
X=df.iloc[:,2:4].values
y=df.iloc[:,-1].values
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(max_depth=1)
clf.fit(X,y)
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X, y, clf=clf, legend=2)
#underfitting


#overfitting
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()
clf.fit(X,y)
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X, y, clf=clf, legend=2)

#well fitted
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(max_depth=4)
clf.fit(X,y)
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X, y, clf=clf, legend=2)
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(max_depth=2)
clf.fit(X,y)
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X, y, clf=clf, legend=2)

from sklearn import tree
tree.plot_tree(clf)
