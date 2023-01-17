# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn import linear_model

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
Powerlifting_base= pd.read_csv('../input/powerlifting-database/openpowerlifting.csv')
Powerlifting_base.drop_duplicates() #remove duplicates
Powerlifting_base.info() #feature output
#We look at the number of unique values for each function (This is how categories are defined)
unique_counts = pd.DataFrame.from_records([(col, Powerlifting_base[col].nunique()) for col in Powerlifting_base.columns],
                                         columns = ['Column_Name', 'Num_Unique']).sort_values(by=['Num_Unique'])
unique_counts
#How we can save memory:
#Because we don't need float64 accuracy for age, but we don't even need float32
Powerlifting_base['Age'] = Powerlifting_base['Age'].astype('float32')
#Categories take up less memory (and time too)!
Powerlifting_base['Equipment'] = Powerlifting_base['Equipment'].astype('category')
Powerlifting_base['Sex'] = Powerlifting_base['Sex'].astype('category')
Powerlifting_base['AgeClass'] = Powerlifting_base['AgeClass'].astype('category')
Powerlifting_base['Event'] = Powerlifting_base['Event'].astype('category')
Powerlifting_base['Country'] = Powerlifting_base['Country'].astype('category')
Powerlifting_base['MeetCountry'] = Powerlifting_base['MeetCountry'].astype('category')
Powerlifting_base.info() #feature output
cols=['Equipment','Age', 'BodyweightKg', 'WeightClassKg', 'Best3SquatKg', 'Best3DeadliftKg', 'Best3BenchKg', 'Place', 'TotalKg', 'Sex']
Powerlifting_base = Powerlifting_base[cols]
Powerlifting_base.head()
#Displaying a bar graph alphabetically and relative frequencies along the 0y axis
# It is a cool tool for displaying information by category
fig = plt.figure(figsize=(14,5))
(Powerlifting_base['Equipment'].value_counts()/len(Powerlifting_base)).sort_index().plot.bar()
#Pandas bar = seaborn countplot, but we do not need to use value_counts
fig = plt.figure(figsize=(14,5))
sns.countplot(Powerlifting_base['Equipment'])
#We calculate the average body weight of an athlete and the number of lifted kilograms for each category in Equipment
Powerlifting_base.groupby(['Equipment'])['BodyweightKg', 'TotalKg'].mean()
#We can visualize this
Powerlifting_base.groupby(['Equipment'])['BodyweightKg', 'TotalKg'].mean().plot.bar(figsize=(14,5),stacked=False)
#Age distribution visualisation
Powerlifting_base['Age'].hist(bins=50)
#Smoothing through kernel density estimation
fig=plt.figure(figsize=(10,5))
sns.kdeplot(Powerlifting_base.Age)
#dropna() - is it deleting rows and columns that have NaN
fig=plt.figure(figsize=(14,5))
sns.distplot(Powerlifting_base['Age'].dropna(), kde=True)
#This visualization is bad
Powerlifting_base[Powerlifting_base['Age']<50].plot.scatter(x='BodyweightKg', y='Age', figsize=(17,6))
#This visualization is good
Powerlifting_base[Powerlifting_base['Age']<50].plot.hexbin(x='BodyweightKg', y='Age', figsize=(17,6), gridsize=25)
#Analog scatterplot is jointplot in seaborn
sns.jointplot(x='BodyweightKg', y='Age', data=Powerlifting_base, kind='hex', gridsize=20)
#Useful thing: inter-quantile percentile plot
fig=plt.figure(figsize=(10,10))
sns.boxplot(y='Age', x='BodyweightKg', data=Powerlifting_base[Powerlifting_base.Age.isin(np.arange(20,40,1))], orient='h')
#Bimodal distribution
fig=plt.figure(figsize=(14,5))
sns.violinplot(y='TotalKg', x='Equipment', hue='Sex', split=True, data=Powerlifting_base)
#Pairwise dependencies
fig=plt.figure(figsize=(5,5))
cols=['Equipment','Age', 'BodyweightKg', 'WeightClassKg', 'Best3SquatKg', 'Best3DeadliftKg', 'Best3BenchKg', 'Place', 'TotalKg', 'Sex']
sns_plot=sns.pairplot(Powerlifting_base[cols].dropna())
#Feature correlation
Powerlifting_base[cols].dropna().corr()
#And  visualization
fig=plt.figure(figsize=(14,14))
sns.heatmap(Powerlifting_base.dropna().corr(), square = True, annot = True, linewidths = .5)
plt.title("Correlation matrix:")
plt.show()
#Missing data visualization
msno.bar(Powerlifting_base, color = 'b', labels = True)
#Linear Regression
#Collecting X and Y:
Powerlifting_base = Powerlifting_base.dropna()
X = Powerlifting_base['Best3SquatKg'].values
Y = Powerlifting_base['TotalKg'].values
#Mean X and Y
mean_X = np.mean(X)
mean_Y = np.mean(Y)
#Total number of values
n = len(X)
#Using the formula to calculate m and c in y=m*x+c
numer = 0
demon = 0
for i in range(n):
    numer += (X[i] - mean_X) * (Y[i] - mean_Y)
    demon += (X[i] - mean_X) ** 2
m = numer / demon
c = mean_Y - (m * mean_X)

print('m = ', m)
print('c = ', c)
#Let's visualize
max_x = np.max(X) + 100
min_x = np.min(X) - 100
#Calculating line values x and y
xx = np.linspace(min_x, max_x)
yy = m * xx + c
#Ploting line
plt.plot(xx, yy, color = 'r', label = 'Regression line')
#Ploting scatter points
plt.scatter(X, Y, color = 'm', label = 'Scatter plot',  alpha=0.6)

plt.xlabel('Best3SquatKg')
plt.ylabel('TotalKg')
plt.legend()
plt.show()
#Missing data visualization (everything alright)
msno.bar(Powerlifting_base, color = 'b', labels = True)
#Linear regression with multiple features by using the SCIKIT-LEARN library
X = Powerlifting_base.loc[:, ('Best3SquatKg', 'Best3DeadliftKg', 'Best3BenchKg')] #independent features
Y = Powerlifting_base.loc[:, 'TotalKg'] #dependent features
#Breakdown of data into 50% training and 50% test data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5)
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, Y_train)
Y_pred = reg.predict(X_test)
Y_pred
Y_test
print('Accuracy: ', reg.score(X_test, Y_test))
#Evaluating the accuracy of the model using special metrics (MSE and MAE)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

mse = mean_squared_error(Y_test, Y_pred) #1/N * sum(Y_test - Y_pred)^2
mae = mean_absolute_error(Y_test, Y_pred) #1/N * sum|Y_test - Y_pred|
print('mse: ', mse)
print('mae: ', mae)
#We can visualize this
fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(121)
plt.scatter(Y_test, Y_pred, alpha=0.1, color = 'b')
plt.xlabel('True TotalKg')
plt.ylabel('Predicted TotalKg')
ax2 = fig.add_subplot(122)
sns.distplot(Y_test-Y_pred)
#Ridge regression with multiple features by using the SCIKIT-LEARN library
from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train, Y_train)
Y_pred = ridge.predict(X_test)
Y_pred
Y_test
print('Accuracy: ', ridge.score(X_test, Y_test))
mse = mean_squared_error(Y_test, Y_pred) #1/N * sum(Y_test - Y_pred)^2
mae = mean_absolute_error(Y_test, Y_pred) #1/N * sum|Y_test - Y_pred|
print('mse: ', mse)
print('mae: ', mae)
#We can visualize this
fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(121)
plt.scatter(Y_test, Y_pred, alpha=0.1, color = 'b')
plt.xlabel('True TotalKg')
plt.ylabel('Predicted TotalKg')
ax2 = fig.add_subplot(122)
sns.distplot(Y_test-Y_pred)
#Lasso regression with multiple features by using the SCIKIT-LEARN library
from sklearn.linear_model import Lasso
lasso = Lasso().fit(X_train, Y_train)
Y_pred = lasso.predict(X_test)
Y_pred
Y_test
print('Accuracy: ', lasso.score(X_test, Y_test))
mse = mean_squared_error(Y_test, Y_pred) #1/N * sum(Y_test - Y_pred)^2
mae = mean_absolute_error(Y_test, Y_pred) #1/N * sum|Y_test - Y_pred|
print('mse: ', mse)
print('mae: ', mae)
#We can visualize this
fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(121)
plt.scatter(Y_test, Y_pred, alpha=0.1, color = 'b')
plt.xlabel('True TotalKg')
plt.ylabel('Predicted TotalKg')
ax2 = fig.add_subplot(122)
sns.distplot(Y_test-Y_pred)
#ElasticNet regression with multiple features by using the SCIKIT-LEARN library
from sklearn.linear_model import ElasticNet
elasticNet= ElasticNet().fit(X_train, Y_train)
Y_pred = elasticNet.predict(X_test)
Y_pred
Y_test
print('Accuracy: ', lasso.score(X_test, Y_test))
mse = mean_squared_error(Y_test, Y_pred) #1/N * sum(Y_test - Y_pred)^2
mae = mean_absolute_error(Y_test, Y_pred) #1/N * sum|Y_test - Y_pred|
print('mse: ', mse)
print('mae: ', mae)
#We can visualize this
fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(121)
plt.scatter(Y_test, Y_pred, alpha=0.1, color = 'b')
plt.xlabel('True TotalKg')
plt.ylabel('Predicted TotalKg')
ax2 = fig.add_subplot(122)
sns.distplot(Y_test-Y_pred)
from sklearn.ensemble import VotingRegressor
votingReg=VotingRegressor([('liReg',reg),('Lasso',lasso),('Ridge',ridge),('ElasticNet',elasticNet)]).fit(X_train, Y_train)
Y_pred = votingReg.predict(X_test)
Y_pred
Y_test
print('Accuracy: ', votingReg.score(X_test, Y_test))
mse = mean_squared_error(Y_test, Y_pred) #1/N * sum(Y_test - Y_pred)^2
mae = mean_absolute_error(Y_test, Y_pred) #1/N * sum|Y_test - Y_pred|
print('mse: ', mse)
print('mae: ', mae)
#We can visualize this
fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(121)
plt.scatter(Y_test, Y_pred, alpha=0.1, color = 'b')
plt.xlabel('True TotalKg')
plt.ylabel('Predicted TotalKg')
ax2 = fig.add_subplot(122)
sns.distplot(Y_test-Y_pred)