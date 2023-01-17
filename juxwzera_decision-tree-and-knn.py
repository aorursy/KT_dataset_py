# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/glass/glass.csv')
data.head()
data.dtypes
# checking the size of our dataset

data.shape
# checking missing data. There's no missing values

data.isnull().sum().sum()
## Checking each column for outliers

fig = plt.figure(figsize = (12,10))



ax1 = fig.add_subplot(3,3,1)

ax2 = fig.add_subplot(3,3,2)

ax3 = fig.add_subplot(3,3,3)

ax4 = fig.add_subplot(3,3,4)

ax5 = fig.add_subplot(3,3,5)

ax6 = fig.add_subplot(3,3,6)

ax7 = fig.add_subplot(3,3,7)

ax8 = fig.add_subplot(3,3,8)

ax9 = fig.add_subplot(3,3,9)



# Boxplot for RI

ax1.boxplot(data['RI'])

ax1.set_title('Distribution of RI')



# Boxplot for Na

ax2.boxplot(data['Na'])

ax2.set_title('Distribution of Na')



# Boxplot for Mg

ax3.boxplot(data['Mg'])

ax3.set_title('Distribution of Mg')





# Boxplot for AL

ax4.boxplot(data['Al'])

ax4.set_title('Distribution of Al')







# Boxplot for Si

ax5.boxplot(data['Si'])

ax5.set_title('Distribution of Si')





# Boxplot for K

ax6.boxplot(data['K'])

ax6.set_title('Distribution of K')



# Boxplot for Ca

ax7.boxplot(data['Ca'])

ax7.set_title('Distribution of Ca')





# Boxplot for Ba

ax8.boxplot(data['Ba'])

ax8.set_title('Distribution of Ba')





# Boxplot for Fe

ax9.boxplot(data['Fe'])

ax9.set_title('Distribution of Fe');

sns.distplot(data['Type'], kde = False)
X = data.iloc[:,:-1]

X
y = data['Type']

y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 50)
from sklearn import tree
classifier_tree = tree.DecisionTreeClassifier()
classifier_tree.fit(X_train, y_train)
plt.figure(figsize = (20,10))

tree.plot_tree(classifier_tree);
predict_type = classifier_tree.predict(X_test)
predict_type
## Let's plot to see how it's performance

sns.distplot(y_test, kde = False)

sns.distplot(predict_type, kde = False)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predict_type)
from sklearn.neighbors import KNeighborsClassifier
classifier_n = KNeighborsClassifier()
# important to mention that n_neighbors = 5, is not a multiple value of our quantity of features

classifier_n.fit(X_train,y_train)
predict_type_n = classifier_n.predict(X_test)
## Let's plot to see how it's performance

sns.distplot(y_test, kde = False)

sns.distplot(predict_type_n, kde = False)
accuracy_score(y_test,predict_type_n)