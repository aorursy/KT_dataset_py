# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn import metrics 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Importing Dataset
dataset = pd.read_csv("../input/redwinequality/datasets_4458_8204_winequality-red.csv")
dataset

#Quick look into Dataset
print(dataset.info())
print(dataset.describe())
print(dataset.head())


#Checking correlation among variables
fig, ax = plt.subplots(figsize=(10,10)) # Sample figsize in inches
sns.heatmap(dataset.corr(), annot=True, linewidths=.5, ax=ax)


#Dataset columns
dataset.columns


#Pairplot of data
sns.pairplot(dataset)


#Deciding Independent And Dependent Variable
X = dataset.iloc[:,6:7].values
y = dataset.iloc[:,11:12].values


#Splitting the Independent and Dependent variable into Train and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .20,random_state = 101)


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


#Viewing MAE,MSE,RMSE
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


#Visualising Our Train Set Model
plt.scatter(X_train, y_train,color='gray')
plt.plot(X_train,lr.predict(X_train),color='red')
plt.title("Prediction vs Actual(TrainSet)")
plt.xlabel("Alcohol")
plt.xlabel("Quality")


#Visualising Our Test Set Model
plt.scatter(X_test, y_test,color='gray')
plt.plot(X_train,lr.predict(X_train),color='red')
plt.title("Prediction vs Actual(TestSet)")
plt.xlabel("Alcohol")
plt.xlabel("Quality")
