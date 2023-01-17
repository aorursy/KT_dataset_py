# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model

#keras libraries
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data_file = "../input/StudentsPerformance.csv"
data = pd.read_csv(data_file)
data.head()
#check if the columns have null value
data.isnull().values.any()
#let's visualise all scores (math, reading and writing)
f, axes = plt.subplots(1, 3, figsize=(20, 5), sharex=True)
sns.distplot(data['math score'], ax=axes[0])
sns.distplot(data['reading score'], ax=axes[1])
sns.distplot(data['writing score'], ax=axes[2])
#gender visualisation
f, axes = plt.subplots(1, 3, figsize=(20, 3), sharex=True)
sns.scatterplot(x="math score", y="gender", data=data, ax=axes[0])
sns.scatterplot(x="reading score", y="gender", data=data, ax=axes[1])
sns.scatterplot(x="writing score", y="gender", data=data, ax=axes[2])
#race visualisation
f, axes = plt.subplots(1, 3, figsize=(20, 5), sharex=True)
sns.scatterplot(x="math score", y="race/ethnicity", data=data, ax=axes[0])
sns.scatterplot(x="reading score", y="race/ethnicity", data=data, ax=axes[1])
sns.scatterplot(x="writing score", y="race/ethnicity", data=data, ax=axes[2])
#parental level of education visualisation
f, axes = plt.subplots(2, 2, figsize=(20, 10), sharex=True)
sns.scatterplot(x="math score", y="parental level of education", data=data, ax=axes[0, 0])
sns.scatterplot(x="reading score", y="parental level of education", data=data, ax=axes[0, 1])
sns.scatterplot(x="writing score", y="parental level of education", data=data, ax=axes[1, 0])
#lunch visualisation
f, axes = plt.subplots(1, 3, figsize=(20, 3), sharex=True)
sns.scatterplot(x="math score", y="lunch", data=data, ax=axes[0])
sns.scatterplot(x="reading score", y="lunch", data=data, ax=axes[1])
sns.scatterplot(x="writing score", y="lunch", data=data, ax=axes[2])
#test preparation course visulaisation
f, axes = plt.subplots(1, 3, figsize=(20, 3), sharex=True)
sns.scatterplot(x="math score", y="test preparation course", data=data, ax=axes[0])
sns.scatterplot(x="reading score", y="test preparation course", data=data, ax=axes[1])
sns.scatterplot(x="writing score", y="test preparation course", data=data, ax=axes[2])
#prepare X and y
target_columns = ['math score', 'reading score', 'writing score']
X = data.drop(target_columns, axis=1)
y = data[['math score', 'reading score', 'writing score']]

#one hot encoding
encoded_X = pd.get_dummies(X)

# splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(encoded_X, y, test_size=0.3, random_state=1)
# create linear regression object 
reg = linear_model.LinearRegression() 
  
# train the model using the training sets 
reg.fit(X_train, y_train) 
  
# regression coefficients 
print('Coefficients: \n', reg.coef_) 
  
# variance score: 1 means perfect prediction 
print('Variance score: {}'.format(reg.score(X_test, y_test)))
#relationship between scores
f, axes = plt.subplots(1, 3, figsize=(20, 5), sharex=True)
sns.scatterplot(x="math score", y="reading score", data=data, ax=axes[0])
sns.scatterplot(x="reading score", y="writing score", data=data, ax=axes[1])
sns.scatterplot(x="writing score", y="math score", data=data, ax=axes[2])
#relationship between scores (gender analysis)
f, axes = plt.subplots(1, 3, figsize=(20, 5), sharex=True)
sns.scatterplot(x="math score", y="reading score", data=data, ax=axes[0], hue="gender")
sns.scatterplot(x="reading score", y="writing score", data=data, ax=axes[1], hue="gender")
sns.scatterplot(x="writing score", y="math score", data=data, ax=axes[2], hue="gender")
#relationship between scores (race/ethnicity analysis)
f, axes = plt.subplots(1, 3, figsize=(20, 5), sharex=True)
sns.scatterplot(x="math score", y="reading score", data=data, ax=axes[0], hue="race/ethnicity")
sns.scatterplot(x="reading score", y="writing score", data=data, ax=axes[1], hue="race/ethnicity")
sns.scatterplot(x="writing score", y="math score", data=data, ax=axes[2], hue="race/ethnicity")
#relationship between scores (parental level of education analysis)
f, axes = plt.subplots(1, 3, figsize=(20, 5), sharex=True)
sns.scatterplot(x="math score", y="reading score", data=data, ax=axes[0], hue="parental level of education")
sns.scatterplot(x="reading score", y="writing score", data=data, ax=axes[1], hue="parental level of education")
sns.scatterplot(x="writing score", y="math score", data=data, ax=axes[2], hue="parental level of education")
#relationship between scores (lunch analysis)
f, axes = plt.subplots(1, 3, figsize=(20, 5), sharex=True)
sns.scatterplot(x="math score", y="reading score", data=data, ax=axes[0], hue="lunch")
sns.scatterplot(x="reading score", y="writing score", data=data, ax=axes[1], hue="lunch")
sns.scatterplot(x="writing score", y="math score", data=data, ax=axes[2], hue="lunch")
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(min_samples_split=20, n_jobs=2, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
regr.score(X_test, y_test)
#scatter plots for gender
f, axes = plt.subplots(1, 3, figsize=(20, 5), sharex=True)
sns.stripplot(x="gender", y="math score", data=data, ax=axes[0])
sns.stripplot(x="gender", y="reading score", data=data, ax=axes[1])
sns.stripplot(x="gender", y="writing score", data=data, ax=axes[2])
#average score and variance for subjects across gender
print("Average math score by gender")
print(data.groupby(['gender'])['math score','reading score','writing score'].mean())
print("Variance in score by gender")
print(data.groupby(['gender'])['math score','reading score','writing score'].var())
print("Standard deviation by gender")
print(data.groupby(['gender'])['math score','reading score','writing score'].std())
#scatter plots for test preparation course
f, axes = plt.subplots(1, 3, figsize=(20, 5), sharex=True)
sns.stripplot(x="test preparation course", y="math score", data=data, ax=axes[0])
sns.stripplot(x="test preparation course", y="reading score", data=data, ax=axes[1])
sns.stripplot(x="test preparation course", y="writing score", data=data, ax=axes[2])
data.head()
#let's visualise scores with gender and see if being male or female affects the scores
f, axes = plt.subplots(1, 3, figsize=(20, 5), sharex=True)
sns.boxplot(x="gender", y="math score", data=data, ax=axes[0])
sns.boxplot(x="gender", y="reading score", data=data, ax=axes[1])
sns.boxplot(x="gender", y="writing score", data=data, ax=axes[2])
#let's visualise race/ethinicity
f, axes = plt.subplots(1, 3, figsize=(20, 5), sharex=True)
sns.boxplot(x="race/ethnicity", y="math score", data=data, ax=axes[0])
sns.boxplot(x="race/ethnicity", y="reading score", data=data, ax=axes[1])
sns.boxplot(x="race/ethnicity", y="writing score", data=data, ax=axes[2])
#let's do one hot encoding on all variables and feed it to the model
#split data into X and Y
target_columns = ['math score', 'reading score', 'writing score']
X = data.drop(target_columns, axis=1)
Y = data[target_columns]
encoded_X = pd.get_dummies(X)
def base_model(input_dim, output_dim):
    #create model
    model = Sequential()
    model.add(Dense(input_dim, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(output_dim, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
#fix random seed for reproducebility
seed = 7
np.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=base_model, epochs=100, batch_size=5, verbose=0, input_dim=encoded_X.shape[1], output_dim=Y.shape[1])
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, encoded_X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
#let's scale Y to 0 and 1
Y_norm = Y/100.0
Y_norm.head()
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=base_model, epochs=100, batch_size=5, verbose=1, input_dim=encoded_X.shape[1], output_dim=Y_norm.shape[1])
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, encoded_X, Y_norm, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))