# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

#Importing Important Library To Dive into Machine Learning
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #Visualisation and Graphs
%matplotlib inline
import seaborn as sns #Another Library for Ploting and Visualisation
from sklearn import metrics


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



#Importing dataset
dataset = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
dataset.head(10)


#A quick look into dataset
print(dataset.head())           #First Few rows of dataset
print(dataset.info())           #overview of dataset
print(dataset.describe())       #Statistical information


#Checking for any missing value for Preprocessing the dataset
dataset.isnull().sum()


#Checking correlation among variables
fig, ax = plt.subplots(figsize=(10,10)) # Sample figsize in inches
sns.heatmap(dataset.corr(), annot=True, linewidths=.5, ax=ax)


#Description of columns in the dataset
dataset.columns


#Deciding Independent And Dependent Variable
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,11:12].values


#Encoding Categorical Variable 

#from sklearn.preprocessing import OneHotEncoder
#from sklearn.compose import ColumnTransformer
#columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
#X = columnTransformer.fit_transform(X)

#If DataFrame has dtype = object 
#"""
X = pd.DataFrame(X,dtype = float )
#"""
X

#Spliting dataset into Train and Test Set
#To train our machine/model and test the Machine output
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .20,random_state = 0)


#Avoiding Dummy variable trap if categorical variable present
#Taken care by library snd hence not neccessary
#X = X.iloc[:,1:]
#X
#fitting the Model(Linear) to the Given data
#Fitting The Model to dataset
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)


#Coefficient And Intercept videos
print(lr.coef_)
print(lr.intercept_)
#Predicting the model on X_test
y_pred = lr.predict(X_test)


#Comparing Actual and Predicted Values
compare = pd.DataFrame({'Actual':y_test.flatten(),'pred':y_pred.flatten()})
compare.head()
#BarPlot between Actual and Predicted
dataset1 = compare.head(25)
dataset1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
#Calculating MAE,MSE,RMSE
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#Building the optimal model using BackwardElimination
import statsmodels.regression.linear_model as lm
X = np.append(arr = np.ones((1599,1)).astype(int),values =X ,axis = 1)
X = pd.DataFrame(X,dtype = float)
X_opt = X.iloc[:,[0,1,2,3,4,5,6,7,8,9,10]]
regressor_ols = lm.OLS(endog = y,exog = X_opt).fit()
regressor_ols.summary()
 
X_opt = X.iloc[:,[0,1,2,5,6,7,8,9,10]]
regressor_ols = lm.OLS(endog = y,exog = X_opt).fit()
regressor_ols.summary()
