# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
data.head()
data.info()
data[data.isnull()].count()
X = data.drop('quality', axis=1)

y = data.quality
# For each feature find the data points with extreme high or low values

for feature in X.keys():



    # Q1 (25th percentile of the data) for the given feature

    Q1 = np.percentile(X[feature], q=25)

 

    # Q3 (75th percentile of the data) for the given feature

    Q3 = np.percentile(X[feature], q=75)

 

    # We use the interquartile range to calculate an outlier step (1.5 times the interquartile range)

    interquartile_range = Q3 - Q1

    step = 1.5 * interquartile_range

 

    # Display the outliers

    print("Data points considered outliers for the feature '{}':".format(feature))

    display(X[~((X[feature] >= Q1 - step) & (X[feature] <= Q3 + step))])

 
X.describe()
sns.pairplot(X)
sns.distplot(y)
### we can plot heat map to examin the correlation

correlation = X.corr()

# display(correlation)

plt.figure(figsize=(14, 12))

heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")
fixedAcidity_citricAcid = X[['citric acid', 'fixed acidity']]

g = sns.JointGrid(x="fixed acidity", y="citric acid", data=fixedAcidity_citricAcid, size=10)

g = g.plot_joint(sns.regplot, scatter_kws={"s": 10})

g = g.plot_marginals(sns.distplot)


# A new dataframe containing only pH and fixed acidity columns to visualize their co-relations

fixedAcidity_pH = X[['pH', 'fixed acidity']]

gridA = sns.JointGrid(x="fixed acidity", y="pH", data=fixedAcidity_pH, size=10)

#Regression plot in the grid 

gridA = gridA.plot_joint(sns.regplot, scatter_kws={"s": 10})

#Distribution plot in the same grid

gridA = gridA.plot_marginals(sns.distplot)
volatileAcidity_quality=data[['volatile acidity','quality']]

fig, axs = plt.subplots(ncols=1,figsize=(15,10))

sns.barplot(x='quality', y='volatile acidity', data=volatileAcidity_quality)

plt.title('quality VS volatile acidity')

plt.show()
alcohol_quality=data[['alcohol','quality']]

fig, axs = plt.subplots(ncols=1,figsize=(15,10))

sns.barplot(x='quality', y='alcohol', data=alcohol_quality)

plt.title('quality VS alcohol content')

plt.show()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(X_train,y_train)

pred=lr.predict(X_test)
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, pred)*100)

print('MSE:', metrics.mean_squared_error(y_test, pred)*100)

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred))*100)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, max_depth=5,random_state=0)

clf.fit(X_train,y_train)
pred=clf.predict(X_test)
from sklearn.metrics import accuracy_score,f1_score
print("accuracy:",(accuracy_score(y_test,pred)*100))
