# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns  # seaborn for visualizationn
import matplotlib.pyplot as plt  # for visualization
from sklearn.linear_model import LinearRegression  #Import Linear regression model
from sklearn.model_selection import train_test_split  #To split the dataset into Train and test randomly
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Read data from CSV
ins_df = pd.read_csv("../input/insurance-premium-prediction/insurance.csv")
ins_df1 = ins_df.copy() #Take a copy of the original dataframe and play in the copied version.
ins_df.head()
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Know about the columns, data types, total rows, number of not null values in each column
ins_df1.info()
#Check the count of duplicate records and remove duplicate records
ins_df1.duplicated().sum()
ins_df1.drop_duplicates(inplace = True)
ins_df1.duplicated().sum()

d = {'yes' : 0, 'no' : 1}
ins_df1['smoker'] =ins_df1['smoker'].map(d)
ins_df1.head()

d = {'male' : 0, 'female' : 1}
ins_df1['sex'] =ins_df1['sex'].map(d)
ins_df1.head()
'''Describe the dataset by statistical measure for each column.Comparing mean and median for outliers. 
Age - Slight variation between mean and median.
bmi - Slight variation between mean and median.
expenses - High variation between mean and median. '''
ins_df1.describe().transpose()
# No outlier in age column.
ax = sns.boxplot(ins_df1['age'])
ax.set_title('Dispersion of Age')
plt.show(ax)
'''To ensure there are no outliers for bmi by box plot. But, there are few bmi values above 47 
which can be considered as outliers'''
ax = sns.boxplot(ins_df1['bmi'])
ax.set_title("Dispersion of bmi")
plt.show(ax)
ax = sns.boxplot(ins_df1['expenses'])
ax.set_title("Dispersion of Expenses")
plt.show(ax)
'''The scatter plot is not representing, when Age is increasing bmi is also increasing. 
Few data points of bmi is high at younger age compare to other data points. Those data points 
can be consider as outliers'''
ax = sns.scatterplot(x = 'age', y = 'bmi', data = ins_df1)
ax.set_title('Age vs BMI')
plt.show(ax)
plt.plot(ins_df1['bmi'],ins_df1['age'] )
plt.title('Age Vs BMI')
plt.xlabel('Age')
plt.ylabel('BMI')
plt.show()
# To understand the relationship between the Age and expenses with respect to bmi.
#----------------------------------------------------------------------------------
#Scatter plot clearly states that, when age is increasing expenses also increasing but has three different
#groups of expenses irrespective of bmi. Hence, BMI is not influencing the expenses with Age. 
plt.figure(figsize=(15,10))
ax = sns.scatterplot(x='age',y='expenses',hue = 'bmi',size = 'bmi', data=ins_df1)
ax = ax.set_title("Age vs Expenses by BMI")
plt.xlabel("Age")
plt.ylabel("Expenses")
plt.show(ax)
#Scatter plot clearly states that, Age with sex are not influencing the expenses.
plt.figure(figsize=(10,7))
ax = sns.scatterplot(x='age',y='expenses', hue='sex',style = 'sex',data=ins_df1)
ax.set_title("Age vs Expenses by Sex")
plt.show(ax)

#Both Age and smoker are highly influncing the expenses. Smoker yes
plt.figure(figsize=(10,7))
ax = sns.scatterplot(x='age',y='expenses', hue=ins_df1['smoker'],style = ins_df1['smoker'],size = ins_df1['smoker'], data=ins_df1)
ax.set_title("Age vs Expenses by Smoker")
plt.xlabel("Smoker (Yes - 1, No - 0)")
plt.ylabel("Expenses")
plt.show(ax)
#To understand the relationship of each independent variable with dependent variable.
#Age has positive side (30%) relationship against expenses
#bmi has positive side (20%) relationship against expenses
#Children has almost no relationship against expenses
#Smoker has strong positive relationship (78%) against expenses
#sex has no relationship against expenses
ins_df1.corr()
#Setting up correlation for our dataframe and passing it to seaborn heatmap function
sns.set(style="white")
corr = ins_df1.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr,annot=True, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5});
corr = ins_df1.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True), annot=True, fmt=".2f")
sns.set(font_scale=0.5)
plt.show()
del corr
#Swarm plot shows how smoker feature is influencing the expeneses compare with smoker and non-smoker
ax = sns.swarmplot(x='smoker',y='expenses',data=ins_df1)
ax.set_title("Smoker vs Expenses")
plt.xlabel("Smoker (Yes - 1, No - 0)")
plt.ylabel("Expenses")
plt.show(ax)
#These three features have relationship with expenses.
x = ins_df1[['age','bmi','smoker']]
y = ins_df1['expenses']
#train_test_split() to split the dataset into train and test set at random.
#test size data set should be 30% data
X_train,X_test,Y_train, Y_test = train_test_split(x,y,test_size=0.3,random_state=42)
#Creating an linear regression model object
model = LinearRegression()
#Training the model using training data set
model.fit(X_train, Y_train) 
#X_train_predict = model.predict(X_train)
#X_test_predict = model.predict(X_test)
print("Intercept value:", model.intercept_)
print("Coefficient values:", model.coef_)
coef_df = pd.DataFrame(list(zip(X_train.columns,model.coef_)), columns = ['Features','Predicted Coeff'])
coef_df
print("Features train data:\n",X_train.smoker)
#Predicting the Y value from the train set and test set.
Y_train_predict = model.predict(X_train)
Y_train_predict[0:5]

Y_test_predict = model.predict(X_test)
#Plot to see the actual expenses and predicted expenses from Train data set
ax = sns.scatterplot(Y_train,Y_train_predict)
ax.set_title("Actual Expenses vs Predicted Expenses")
plt.xlabel("Actual Expenses")
plt.ylabel("Predicted Expenses")
plt.show(ax)
#Train and predict the Y_train for the feature 'smoker'
smoker_model = LinearRegression()
smoker_model.fit(X_train[['smoker']], Y_train)
print("intercept:",smoker_model.intercept_, "coeff:", smoker_model.coef_)

#print("Train - Mean squared error:", np.mean((Y_train - model.predict(X_train)) ** 2))
smoker_df = pd.DataFrame(list(zip(Y_train, smoker_model.predict(X_train[['smoker']]))), columns = ['Actual Expenses','Predicted Expenses'])
smoker_df.head()
#X_train['smoker'].shape
#MSE for Train data set
print("MSE:",np.sqrt(mean_squared_error(Y_train, Y_train_predict)))
print("MSE only for Smoker:", np.sqrt(mean_squared_error(Y_train,smoker_model.predict(X_train[['smoker']]))))
#R-Squared value for Train data set
print("R-squared value:",round(r2_score(Y_train, Y_train_predict),3))
print("R-squared value only for smoker:", round(r2_score(Y_train,smoker_model.predict(X_train[['smoker']]))),3)
#Mean absolute error for Train data set
print("Mean absolute error:",mean_absolute_error(Y_train, Y_train_predict))
print("Mean absolute Error only for Smoker:", mean_absolute_error(Y_train,smoker_model.predict(X_train[['smoker']])))
print("MSE for Test data set")
print("MSE:",np.sqrt(mean_squared_error(Y_test, Y_test_predict)))
print("MSE only for Smoker:", np.sqrt(mean_squared_error(Y_test,smoker_model.predict(X_test[['smoker']]))))
_actual_n_predicted=pd.DataFrame({'Actual': Y_test, 'Predicted': Y_test_predict})
_actual_n_predicted_top5 = _actual_n_predicted.head(5)
## Actual vs Predicted Plot
_actual_n_predicted_top5.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
############ Doing it by Polynominal Linear Regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,r2_score
### API Documentation for https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
### Took Reference of following classes


def create_polynomial_regression_model(degree,X,y):
  "Creates a polynomial regression model for the given degree"
  
  X_train,X_test,Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=9)
  poly_features = PolynomialFeatures(degree=degree)
  
  # transforms the existing features to higher degree features.
  X_train_poly = poly_features.fit_transform(X_train)
  
  # fit the transformed features to Linear Regression
  poly_model = LinearRegression()
  poly_model.fit(X_train_poly, Y_train)
  
  # predicting on training data-set
  y_train_predicted = poly_model.predict(X_train_poly)
  
  # predicting on test data-set
  y_test_predict = poly_model.predict(poly_features.fit_transform(X_test))
  
  # evaluating the model on training dataset
  rmse_train = np.sqrt(mean_squared_error(Y_train, y_train_predicted))
  r2_train = r2_score(Y_train, y_train_predicted)
  
  # evaluating the model on test dataset
  rmse_test = np.sqrt(mean_squared_error(Y_test, y_test_predict))
  r2_test = r2_score(Y_test, y_test_predict)
  
  print("The model performance for the training set")
  print("-------------------------------------------")
  print("RMSE of training set is {}".format(rmse_train))
  print("R2 score of training set is {}".format(r2_train))
  
  print("\n")
  
  print("The model performance for the test set")
  print("-------------------------------------------")
  print("RMSE of test set is {}".format(rmse_test))
  print("R2 score of test set is {}".format(r2_test))
create_polynomial_regression_model(3,X_train,Y_train)
#X_train,X_test,Y_train, Y_test
## Let's break the dataset into test and training dataset
X_train,X_test,Y_train, Y_test = train_test_split(x,y,test_size=0.3,random_state=42)
### Training on whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, Y_train)
regressor.predict(X_test)
_actual_n_predicted=pd.DataFrame({'Actual': Y_test, 'Predicted': Y_test_predict})
_actual_n_predicted_top5 = _actual_n_predicted.head(5)
## Actual vs Predicted Plot
_actual_n_predicted_top5.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()