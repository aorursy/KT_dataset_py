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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn.preprocessing import RobustScaler

from scipy import stats

import numpy as np
dfs = pd.read_csv('../input/yeh-concret-data/Concrete_Data_Yeh.csv')
dfs.head()
dfs.columns = ['cement', 'blast_furnace_slag', 'fly_ash','water','superplast','course_aggregate','fine_aggregate','age','compressive_strength']

#Checking for null values

dfs.isnull().sum()
dfs.describe()
plt.figure(figsize = (20,10))

plt.title("Correlation Heatmap")

sns.heatmap(dfs.corr(), annot=True,cmap="YlGnBu")
fig, ax = plt.subplots() #initialization
import matplotlib.pyplot as plt

# Green markers indicating outliers in the feature



green_diamond = dict(markerfacecolor='g', marker='D')





ax1 = fig.add_subplot(111)

ax2 = fig.add_subplot(122)

ax3 = fig.add_subplot(133)







fig, (ax1, ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))



ax1.set_title('Cement')

ax1.boxplot(dfs['cement'], flierprops=green_diamond)



ax2.set_title('Blast Furnace Slag')

ax2.boxplot(dfs['blast_furnace_slag'], flierprops=green_diamond)



ax3.set_title('Fly Ash')

ax3.boxplot(dfs['fly_ash'], flierprops=green_diamond)
ax4 = fig.add_subplot(111)

ax5 = fig.add_subplot(122)

ax6 = fig.add_subplot(133)

fig, (ax4, ax5, ax6) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))



ax4.set_title('Water')

ax4.boxplot(dfs['water'], flierprops=green_diamond)



ax5.set_title('Super Plasticizer')

ax5.boxplot(dfs['superplast'], flierprops=green_diamond)



ax6.set_title('Coarse Aggregate')

ax6.boxplot(dfs['course_aggregate'], flierprops=green_diamond)



ax7 = fig.add_subplot(111)

ax8 = fig.add_subplot(122)

ax9 = fig.add_subplot(133)



fig, (ax7, ax8, ax9) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))



ax7.set_title('Fine Aggregate')

ax7.boxplot(dfs['fine_aggregate'], flierprops=green_diamond)



ax8.set_title('Age')

ax8.boxplot(dfs['age'], flierprops=green_diamond)



ax9.set_title('Compressive Strength')

ax9.boxplot(dfs['compressive_strength'], flierprops=green_diamond)
from sklearn.model_selection import train_test_split

X= dfs.iloc[:, dfs.columns != 'compressive_strength']

y = dfs.iloc[:, dfs.columns == 'compressive_strength']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=0)



from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)



#Predicted Variable

y_pred = regressor.predict(X_test)



# MAE

mae = mean_absolute_error(y_test, y_pred)



#RMSE 

rmse = mean_squared_error(y_test,y_pred, squared = False)



#R2_Score

r2 = r2_score(y_test,y_pred)

print("For a linear model the errors are as follows-")

print("Mean Absolute Error:", mae)

print("Root Mean Squared Error:", rmse)

print("R2 Score:",r2)
#Visualizing the correlation among the variables



g = sns.pairplot(dfs, diag_kind="kde")

g.map_lower(sns.kdeplot, levels=4, color=".2")
#Plotting a Compressive Strength vs Cement, Age, Water plot

fig_dims = (15, 8)

fig, ax = plt.subplots(figsize=fig_dims)

x = dfs['compressive_strength']

y= dfs['cement']

sns.scatterplot(y=y, x=x, hue="water",size="age", data=dfs, sizes=(50, 300))

ax.set_title('Compressive Strength vs Cement, Age, Water')
dfs = pd.read_csv('../input/yeh-concret-data/Concrete_Data_Yeh.csv')

from sklearn.preprocessing import RobustScaler

rs = RobustScaler()

dfs_scaled = pd.DataFrame(rs.fit_transform(dfs),columns = dfs.columns)



X= dfs_scaled.iloc[:, dfs.columns != 'compressive_strength']

y = dfs_scaled.iloc[:, dfs.columns == 'compressive_strength']





#Removing outliers using interquartile range

Q1 = dfs.quantile(0.25)

Q3 = dfs.quantile(0.75)

IQR = Q3 - Q1

print(IQR)

dfs_out = dfs[~((dfs < (Q1 - 1.5 * IQR)) |(dfs > (Q3 + 1.5 * IQR))).any(axis=1)]
# Pearson's correlation feature selection for numeric input and numeric output

from sklearn.datasets import make_regression

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_regression

# generate dataset

X= dfs_out.iloc[:, [0,1,2,3,4,5,6,7]]

y = dfs_out.iloc[:, 8]



# define feature selection SELECTING THE BEST FEATURES 7 OUT OF 8

fs = SelectKBest(score_func=f_regression, k=7)

# apply feature selection

X_selected = fs.fit_transform(X, y)

print(X_selected.shape)

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size= 0.2, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)

y_pre = regressor.predict(X_test)

from sklearn.metrics import mean_absolute_error

maep = mean_absolute_error(y_test, y_pre)



rmsep = rmse = mean_squared_error(y_test,y_pre, squared = False)

from sklearn.metrics import r2_score

r2p = r2_score(y_test,y_pre)



print("For a linear model the errors are as follows-")

print("Mean Absolute Error:", maep)

print("Root Mean Squared Error:", rmsep)

print("R2 Score:",r2p)
from sklearn.datasets import make_friedman1

from sklearn.feature_selection import RFE

from sklearn.svm import SVR

X= dfs.iloc[:, dfs.columns != 'compressive_strength']

y = dfs.iloc[:, 8]

estimator = SVR(kernel="linear")

selector = RFE(estimator, n_features_to_select=5, step=1)

selector = selector.fit(X, y)

selector.ranking_
#Selecting the number 1 ranking variables only

# The variables are: cement, blast_furnace_slag,  water, superplast and age

X= dfs_out.iloc[:, [0,1,3,4,7]]

y = dfs_out.iloc[:, 8]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)

y_pre = regressor.predict(X_test)

from sklearn.metrics import mean_absolute_error

maef = mean_absolute_error(y_test, y_pre)



rmsef = mean_squared_error(y_test,y_pre, squared = False)

from sklearn.metrics import r2_score

r2f = r2_score(y_test,y_pre)



print("For a linear model the errors are as follows-")

print("Mean Absolute Error:", maef)

print("Root Mean Squared Error:", rmsef)

print("R2 Score:",r2f)
X= dfs_scaled.iloc[:, dfs.columns != 'compressive_strength']

y = dfs_scaled.iloc[:, 8]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=0)

from sklearn.decomposition import PCA

pca = PCA(n_components = 4)

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_



from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)

y_pr = regressor.predict(X_test)



from sklearn.metrics import mean_absolute_error

maepca = mean_absolute_error(y_test, y_pr)



rmsepca = mean_squared_error(y_test,y_pr, squared = False)

from sklearn.metrics import r2_score

r2pca = r2_score(y_test,y_pr)



print("For a linear model the errors are as follows-")

print("Mean Absolute Error:", maepca)

print("Root Mean Squared Error:", rmsepca)

print("R2 Score:",r2pca)