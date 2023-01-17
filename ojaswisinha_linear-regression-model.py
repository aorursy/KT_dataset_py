# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv('/kaggle/input/random-linear-regression/train.csv')   #import the data
data.head(10)
data.isnull().any() #checck for null value 'y' is having null value
data.shape
data.describe()  #count of y shows only one value missing
data.fillna(49, inplace = True)   #fill with the mean value
data.isnull().any()  #check for null value again
X = np.array(data.iloc[:,0]) #X and y for training and test set, x will not hhave y and y will not have x

y = np.array(data.iloc[:,1])

X = X.reshape(-1,1)  #reshape 1D array to 2D

y = y.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)    #split the data
lm = LinearRegression()    #instantiate the model
lm.fit(X_train,y_train)   #fit the data
pred = lm.predict(X_test)   #predict the data
mse = mean_squared_error(y_test,pred)   #check the prediction error
print("Mean squared error: ", mse)

print("r2 : ", r2_score(y_test,pred))
plt.scatter(x = y_test, y = pred)    #relationship between actual vs, predicted value

plt.title("Actual vs Predicted")

plt.xlabel("Actual")

plt.ylabel("Predicted")
sns.distplot(y_test, bins = 10, hist= False, label = 'Distribution of data')
sns.jointplot(x= y_test, y=pred, kind="kde", color = 'violet')

sns.jointplot(x= y_test, y=pred, kind="scatter", color = 'green')
sns.jointplot(x= y_test, y=pred, kind="reg", color = "yellow")