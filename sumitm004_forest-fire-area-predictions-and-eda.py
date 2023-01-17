# Importing the Required Libraries



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from scipy.stats import norm

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

from sklearn.preprocessing import PolynomialFeatures

from sklearn.svm import SVR

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

import warnings; warnings.simplefilter('ignore')
# Importing the dataset



df_forest = pd.read_csv("/kaggle/input/forest-fire-area/forestfires.csv")

df_forest.head()
# Shape of the dataset

print ("The shape of the dataset : ", df_forest.shape)
# Skewness of the Area in the dataset

plt.rcParams['figure.figsize'] = [8, 8]

sns.distplot(df_forest['area']);
# Reducing the Right Skewness of the Area using log(n) + 1

df_forest['u_area'] = np.log(df_forest['area'] + 1)
# setting parameters

plt.rcParams['figure.figsize'] = [20, 10]

sns.set(style = "darkgrid", font_scale = 1.3)

month_temp = sns.barplot(x = 'month', y = 'temp', data = df_forest,

                         order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], palette = 'winter');

month_temp.set(title = "Month Vs Temp Barplot", xlabel = "Months", ylabel = "Temperature");
df_forest.day.unique()
plt.rcParams['figure.figsize'] = [10, 10]

sns.set(style = 'whitegrid', font_scale = 1.3)

day = sns.countplot(df_forest['day'], order = ['sun' ,'mon', 'tue', 'wed', 'thu', 'fri', 'sat'], palette = 'spring')

day.set(title = 'Countplot for the days in the week', xlabel = 'Days', ylabel = 'Count');
plt.rcParams['figure.figsize'] = [8, 8]

sns.set(style = "white", font_scale = 1.3)

scat = sns.scatterplot(df_forest['temp'], df_forest['area'])

scat.set(title = "Scatter Plot of Area and Temperature", xlabel = "Temperature", ylabel = "Area");
# After Reducing the Skewness

plt.rcParams['figure.figsize'] = [8, 8]

sns.set(style = "white", font_scale = 1.3)

scat = sns.scatterplot(df_forest['temp'], df_forest['u_area'])

scat.set(title = "Scatter Plot of Area and Temperature", xlabel = "Temperature", ylabel = "Area");
# Setting Parameters

plt.rcParams['figure.figsize'] = [20, 10]

sns.set(style = 'white', font_scale = 1.3)

fig, ax = plt.subplots(1,2)



# Distribution Plots

area_dist = sns.distplot(df_forest['area'], ax = ax[0]);

area_dist_2 = sns.distplot(df_forest['u_area'], ax = ax[1]);

area_dist.set(title = "Skewed Area Distribution", xlabel = "Area", ylabel = "Density")

area_dist_2.set(title = "Reduced Skewness of Area Distribution", xlabel = "U_Area", ylabel = "Density");
# Correlation Heatmap of the features in the dataset

plt.rcParams['figure.figsize'] = [12, 10]

sns.set(font_scale = 1)

sns.heatmap(df_forest.corr(), annot = True);
data = norm.rvs(df_forest['area'])



# Fit a normal distribution to the data

mu, std = norm.fit(data)



plt.hist(data, bins=25, density=True, alpha=0.6, color='g')



# Plot the PDF

xmin, xmax = plt.xlim()

x = np.linspace(xmin, xmax, 100)

p = norm.pdf(x, mu, std)

plt.plot(x, p, 'k', linewidth=2)

title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)

plt.title(title)



plt.show()
# Reducing the skewness for the final training and dropping u_area

df_forest['area'] = np.log(df_forest['area'] + 1)

df_forest.drop(columns = 'u_area', inplace = True)



df_forest.head(10)
df_forest['day'].value_counts()
df_forest.describe()
# Changing categorical values into numerical values



# Months

df_forest['month'].replace({'jan' : 1, 'feb' : 2, 'mar' : 3, 'apr' : 4, 'may' : 5, 'jun' : 6,

                           'jul' : 7, 'aug' : 8, 'sep' : 9, 'oct' : 10, 'nov' : 11, 'dec' : 12},

                           inplace = True)



# Days

df_forest['day'].replace({'sun' : 1, 'mon' : 2, 'tue' : 3, 'wed' : 4, 'thu' : 5, 'fri' : 6, 'sat' : 7}, inplace = True)



# # Using Label Encoder for cat to num conversion

# categorical = list(df_forest.select_dtypes(include = ["object"]).columns)

# for i, column in enumerate(categorical) :

#     label = LabelEncoder()

#     df_forest[column] = label.fit_transform(df_forest[column])



df_forest.head(10)
target = df_forest['area']

features = df_forest.drop(columns = 'area')



X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.15, random_state = 196)



print ("Train data set size : ", X_train.shape)

print ("Test data set size : ", X_test.shape)
X_train.head()
# Linear Regression Model

model = LinearRegression()

model.fit(X_train, y_train)



# Predictions

predictions = model.predict(X_test)



# Scores

print ("Mean Squared Error : ", mean_squared_error(y_test, predictions))

print ("r2 Score : ", r2_score(y_test, predictions))
# Transforming data

poly = PolynomialFeatures(4)

poly_X_train = poly.fit_transform(X_train)

poly_X_test = poly.fit_transform(X_test)



model_2 = LinearRegression()

model_2.fit(poly_X_train, y_train)



# Predictions

predictions_poly = model_2.predict(poly_X_test)



# Scores

print ("Mean Squared Error : ", mean_squared_error(y_test, predictions_poly))

print ("r2 Score : ", r2_score(y_test, predictions_poly))
# Lasso regression

model_3 = Lasso(alpha = 100, max_iter = 10000) 

model_3.fit(X_train, y_train)



# Predictions

prediction = model_3.predict(X_test)



# Scores

print ("Mean Squared Error : ", mean_squared_error(y_test, prediction))

print ("r2 Score : ", r2_score(y_test, prediction))
# Ridge Regression

model_4 = Ridge(alpha = 500)

model_4.fit(X_train, y_train)



# Predictions

pred = model_4.predict(X_test)



# Scores

print ("Mean Squared Error : ", mean_squared_error(y_test, pred))

print ("r2 Score : ", r2_score(y_test, pred))
# ElasticNet

model_5 = ElasticNet(alpha = 100, max_iter = 10000)

model_5.fit(X_train, y_train)



# Predictions

pred1 = model_5.predict(X_test)



# Scores

print ("Mean Squared Error : ", mean_squared_error(y_test, pred1))

print ("r2 Score : ", r2_score(y_test, pred1))
# SVR

model_6 = SVR(C = 100, kernel = 'linear')

model_6.fit(X_train, y_train)



# Predictions

prediction = model_6.predict(X_test)



# Scores

print ("Mean Squared Error : ", mean_squared_error(y_test, prediction))

print ("r2 Score : ", r2_score(y_test, prediction))
prediction = np.exp(prediction - 1)

prediction