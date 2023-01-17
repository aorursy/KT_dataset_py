# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # import seaborn library for correlation matrix heatmap and other plots



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# create a variable for the wine file path

wine_file_path = '../input/red-wine-quality/winequality-red.csv'



# turn the csv file into a dataframe based on the semicolon delimiter

wine_data = pd.read_csv(wine_file_path, sep=';')



# examine the data

wine_data.head()
# import plotting functions

import matplotlib.pyplot as plt



# plot all of the values for quality along with their counts

sns.countplot(x=wine_data['quality'], data=wine_data)

plt.title('Count of All Wine Quality Values')
# get descriptive statistics about this dataset to better understand what is going on

wine_data.describe()
# create the prediction target: quality

y = wine_data.quality



# look at the structure of the prediction target

y.describe()
# import required libraries to conduct a linear regression

from sklearn.linear_model import LinearRegression



# define X and Y for the plot

X = wine_data.iloc[:,10].values.reshape(-1,1)

Y = wine_data.iloc[:,11].values.reshape(-1,1)



# build and train the model

model = LinearRegression()

model.fit(X,Y)

Y_pred = model.predict(X)



# plot the scatter plot

plt.scatter(X,Y,s=10)



# plot the prediction line

plt.plot(X, Y_pred, color='red')



# plot the title and axes

plt.title('The Higher the Alcohol Content, the Higher the Red Wine Quality')

plt.xlabel('Alcohol Content')

plt.ylabel('Red Wine Quality')
# define X and Y for the plot

X = wine_data.iloc[:,3].values.reshape(-1,1)

Y = wine_data.iloc[:,11].values.reshape(-1,1)



# build and train the model

model = LinearRegression()

model.fit(X,Y)

Y_pred = model.predict(X)



# plot the scatter plot

plt.scatter(X,Y,s=10)



# plot the prediction line

plt.plot(X, Y_pred, color='red')



# plot the title and axes

plt.title('There is No Relationship Between Residual Sugar and Red Wine Quality')

plt.xlabel('Residual Sugar')

plt.ylabel('Red Wine Quality')
# define X and Y for the plot

X = wine_data.iloc[:,0].values.reshape(-1,1)

Y = wine_data.iloc[:,11].values.reshape(-1,1)



# build and train the model

model = LinearRegression()

model.fit(X,Y)

Y_pred = model.predict(X)



# plot the scatter plot

plt.scatter(X,Y,s=10)



# plot the prediction line

plt.plot(X, Y_pred, color='red')



# plot the title and axes

plt.title('There is a Weak Positive Relationship between Fixed Acidity and Red Wine Quality')

plt.xlabel('Fixed Acidity')

plt.ylabel('Red Wine Quality')
# plot a correlation matrix of all of the variables

f = plt.figure(figsize=(12,10))

wine_datacorr = wine_data.corr()

sns.heatmap(wine_datacorr)

plt.xticks(rotation=45)

plt.title("Correlation Matrix Between All Red Wine Variables")
# finding specific correlations of each feature with our target variable of quality

correlations = wine_data.corr()['quality'].drop('quality')

print(correlations)
# choose the features that I hypothesize may have the biggest impact on quality based on correlation matrix

wine_features = ['alcohol', 'sulphates', 'citric acid', "volatile acidity"]



# training features

X = wine_data[wine_features]



# target

y = wine_data.quality



# review the data that we'll use to make the quality prediction

X.describe()
# review what the rows look like for these columns

X.head()
# set up train and test to see how well the model does

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score, accuracy_score



# set aside 20% of the data to test the model on later

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=3)



# define model

linregressor = LinearRegression()



# fit linear regression to training data

linregressor.fit(X_train, y_train)



# create the quality predictions

y_pred = linregressor.predict(X_test)



# see how well the model did based on r squared score and mean squared error

print("The r-squared value for the linear regression model with 4 features is", r2_score(y_test, y_pred))

print("The mean squared error for the linear regression model with 4 features is", mean_squared_error(y_test, y_pred))
# include all variables as training features this time

X = wine_data.drop('quality', axis=1)
# set aside 20% of the data to test the model on later

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=3)



# define model

linregressor = LinearRegression()



# fit linear regression to training data

linregressor.fit(X_train, y_train)



# create the quality predictions

linreg_pred = linregressor.predict(X_test)



# see how well the model did based on r squared score and mean squared error

print("The r-squared value for the linear regression model with all features is", r2_score(y_test, linreg_pred))

print("The mean squared error for the linear regression model with all features is", mean_squared_error(y_test, linreg_pred))
# make some predictions of quality based on linear regression model

print("Making linear regression model predictions for the following 5 wine quality values:")



# we choose rows 15-19 because they have a wide range of quality values

print(wine_data['quality'].loc[15:19])

print("The linear regression model predictions of quality are", linregressor.predict(X.loc[15:19]))
# we choose to build a decision tree based on scikit learn's diagram

from sklearn.tree import DecisionTreeRegressor



# set aside 20% of the data to test the model on later

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=3)



# define model

treeregressor = DecisionTreeRegressor()



# fit decision tree to training data

treeregressor.fit(X_train,y_train)



# predict

tree_pred = treeregressor.predict(X_test)



# print accuracy score

tree_accscore = accuracy_score(y_test, tree_pred)



print("The decision tree has an accuracy score of", tree_accscore*100,"%.")
# the decision tree makes some predictions of quality

print("Making decision tree predictions for the following 5 wine quality values:")



# we choose rows 15-19 because they have a wide range of quality values

print(wine_data['quality'].loc[15:19])

print("The decision tree predictions of quality are", treeregressor.predict(X.loc[15:19]))
# we choose to build a logistic regression model

from sklearn.linear_model import LogisticRegression



# set aside 20% of the data to test the model on later

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=3)



# define model

logisticregression = LogisticRegression()



# fit decision tree to training data

logisticregression.fit(X_train,y_train)



# predict

logr_pred = logisticregression.predict(X_test)



# print accuracy score

logisticregression_accscore = accuracy_score(y_test, logr_pred)



print("The logistic regression model has an accuracy score of ", logisticregression_accscore*100,"%.")
# the logistic regression model makes some predictions of quality

print("Making logistic regression predictions for the following 5 wine quality values:")



# we choose rows 15-19 because they have a wide range of quality values

print(wine_data['quality'].loc[15:19])

print("The logistic regression predictions of quality are", logisticregression.predict(X.loc[15:19]))
# import random forest classifier

from sklearn.ensemble import RandomForestClassifier



# set aside 20% of the data to test the model on later

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=3)



# define model

rf = RandomForestClassifier()



# fit decision tree to training data

rf.fit(X_train,y_train)



# predict

rf_pred = rf.predict(X_test)



# print accuracy score

rf_accscore = accuracy_score(y_test, rf_pred)



print("The random forest has an accuracy score of ", rf_accscore*100,"%.")
# the random forest model makes some predictions of quality

print("Making random forest predictions for the following 5 wine quality values:")



# we choose rows 15-19 because they have a wide range of quality values

print(wine_data['quality'].loc[15:19])

print("The random forest predictions of quality are", rf.predict(X.loc[15:19]))
# group numerical scores into human-understandable quality categories

# if the wine scores a 3 or a 4, the wine is LowQuality QualityCategory

# if the wine scores a 5 or a 6, the wine is MediumQuality QualityCategory

# if the wine scores a 7 or an 8, the wine is HighQuality QualityCategory



# create a list called QualityCategory

QualityCategory = []



# iterate through all rows in the dataset and assign a category for the quality score and add to the list

for wine in wine_data['quality']:

    if wine == 3 or wine == 4:

        QualityCategory.append('LowQuality')

    if wine == 5 or wine == 6:

        QualityCategory.append('MediumQuality')

    if wine == 7 or wine == 8:

        QualityCategory.append('HighQuality')



# add the category list as a column to the wine_data df     

wine_data['QualityCategory'] = QualityCategory



# print out 5 rows of the dataframe we know have values in each of the categories with the new category label

wine_data.loc[15:19]
# plot all of the values for qualitycategory along with their counts

sns.countplot(x=wine_data['QualityCategory'], data=wine_data)

plt.title('Count of All Wine Quality Category Values')
# define our new target as the QualityCategory

y = wine_data.QualityCategory



# set aside 20% of the data to test the model on later

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=3)



# define model

rf_category = RandomForestClassifier()



# fit decision tree to training data

rf_category.fit(X_train,y_train)



# predict

rf_category_pred = rf_category.predict(X_test)



# print accuracy score

rf_category_accscore = accuracy_score(y_test, rf_category_pred)



print("The random forest has an accuracy score of ", rf_category_accscore*100,"%.")
# the random forest model makes some predictions of quality

print("Making random forest predictions for the following 5 wine category values:")



# chose rows 15-19 because they have an example of each category

print(wine_data['QualityCategory'].loc[15:19])

print("The random forest predictions of quality category are", rf.predict(X.loc[15:19]))
from sklearn.naive_bayes import GaussianNB



# target

y = wine_data.QualityCategory



# set aside 20% of the data to test the model on later

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=3)



# define naive bayes model

nb = GaussianNB()



# fit decision tree to training data

nb.fit(X_train,y_train)



# predict

nb_pred = nb.predict(X_test)



# print accuracy score

nb_accscore = accuracy_score(y_test, nb_pred)



print("When predicting the wine quality category, the naive bayes model has an accuracy score of ", nb_accscore*100,"%.")
# the naive bayes model makes some predictions of quality

print("Making naive bayes predictions for the following 5 wine category values:")



# chose rows 15-19 because they have an example of each category

print(wine_data['QualityCategory'].loc[15:19])

print("The naive bayes predictions of quality category are", nb.predict(X.loc[15:19]))
# create a variable for the wine file path

wine_file_path = '../input/white-wine-quality/winequality-white.csv'



# turn the csv file into a dataframe based on the semicolon delimiter

white_wine_data = pd.read_csv(wine_file_path, sep=';')



# examine the data

white_wine_data.head()
# check the distributions of quality values

sns.countplot(x=white_wine_data['quality'], data=white_wine_data)
# group numerical scores into human-understandable quality categories

# if the wine scores a 3 or a 4, the wine is LowQuality QualityCategory

# if the wine scores a 5 or a 6, the wine is MediumQuality QualityCategory

# if the wine scores a 7 or an 8, the wine is HighQuality QualityCategory



# create a list called QualityCategory

WhiteQualityCategory = []



# iterate through all rows in the dataset and assign a category for the quality score and add to the list

for wine in white_wine_data['quality']:

    if wine == 3 or wine == 4:

        WhiteQualityCategory.append('LowQuality')

    if wine == 5 or wine == 6:

        WhiteQualityCategory.append('MediumQuality')

    if wine == 7 or wine == 8 or wine == 9:

        WhiteQualityCategory.append('HighQuality')



# add the category list as a column to the wine_data df     

white_wine_data['WhiteQualityCategory'] = WhiteQualityCategory



# print out 5 rows of the dataframe we know have values in each of the categories with the new category label

white_wine_data.loc[95:99]
# define our new target as the White wine QualityCategory

y = white_wine_data.WhiteQualityCategory[0:1599]



# test the model on 20% of the white wine data (just as we did for red wine data)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=3)



# predict using the already trained model from part 2 (note that we did not re-train the model)

rf_whitecategory_pred = rf_category.predict(X_test)



# print accuracy score

rf_whitecategory_accscore = accuracy_score(y_test, rf_whitecategory_pred)



print("The random forest trained on red white quality categories has an accuracy score of ", rf_whitecategory_accscore*100,"%", "when predicting white wine quality categories.")
# define our new target as the White wine QualityCategory

# look at the same number of rows as were in the red wine data set - this seems like a hack?



# define our white model prediction target

ywhite = white_wine_data.WhiteQualityCategory



# define our white model features

Xwhite = white_wine_data.drop(['quality','WhiteQualityCategory'], axis=1)



# import random forest classifier

from sklearn.ensemble import RandomForestClassifier



# set aside 20% of the data to test the model on later

Xwhite_train, Xwhite_test, ywhite_train, ywhite_test = train_test_split(Xwhite,ywhite,test_size=0.2, random_state=3)



# define model

rfwhite = RandomForestClassifier()



# fit decision tree to training data

rfwhite.fit(Xwhite_train,ywhite_train)



# predict

rfwhite_pred = rfwhite.predict(Xwhite_test)



# print accuracy score

rfwhite_accscore = accuracy_score(ywhite_test, rfwhite_pred)



print("The random forest has an accuracy score of ", rfwhite_accscore*100,"% when trained on the white wine data set.")