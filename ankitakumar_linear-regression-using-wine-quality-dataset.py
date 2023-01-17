# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#creating a dataframe in pandas
df=pd.read_csv('../input/winequality-red.csv')
#printing the first 5 rows of the dataframe
df.head()
#computing summary statistics for each column of the dataframe
df.describe()
#Let's plot a scatter plot of alcohol against its quality
import matplotlib.pyplot as plt
plt.scatter(df['alcohol'],df['quality'])
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title('Plotting alcohol against the quality')
plt.grid(True)
plt.show()
#Plotting volatile acidity against quality
import matplotlib.pyplot as plt
plt.scatter(df['volatile acidity'],df['quality'])
plt.xlabel('Volatile Acidity')
plt.ylabel('Quality')
plt.title('Volatile Acidity against Quality')
plt.grid(True)
plt.show()
df.corr()
#This gives the pairwise correlation matrix
#import libraries
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression
#from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
df=pd.read_csv('../input/winequality-red.csv')
X = df[list(df.columns)[:-1]]
y=df['quality']
#divide the data into training and testing set 
X_train, X_test,y_train,y_test=train_test_split(X,y)
#Creating the linear regression model and fitting the data to it
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_prediction=regressor.predict(X_test)
#Printing the r score value
print('R-score is %s'%regressor.score(X_test,y_test))
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
df=pd.read_csv('../input/winequality-red.csv')
X = df[list(df.columns)[:-1]]
y=df['quality']
regressor=LinearRegression()
#Computing score using 5 fold cross validation method. cv is used to determine the folds ie 5
scores=cross_val_score(regressor,X,y,cv=5)
print(scores.mean())
print(scores)
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#Loading the boston housing dataset
data=load_boston()
#Splitting the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(data.data,
data.target)
X_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
#y_train = y_scaler.fit_transform(y_train)
X_test = X_scaler.transform(X_test)
#y_test = y_scaler.transform(y_test)
from sklearn.preprocessing import scale
import numpy as np
y_train = np.array(y_train).reshape(-1,1)
y_test = np.array(y_test).reshape(-1,1)
y_train = y_scaler.fit_transform(y_train)
y_test=y_scaler.fit_transform(y_test)
regressor = SGDRegressor(loss='squared_loss')
scores = cross_val_score(regressor, X_train, y_train, cv=5)
print('Cross validation r-squared scores:%s' %scores)
print('Average cross validation r-squared score:%s'%np.mean(scores))
y_test=y_test.flatten()
y_train=y_train.flatten()
regressor.fit(X_train, y_train)
print('Test set r-squared score:%s' %regressor.score(X_test, y_test))