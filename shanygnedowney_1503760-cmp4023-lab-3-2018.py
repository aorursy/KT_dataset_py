# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
import seaborn as sns
#matplotlib.style.use('ggplot')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = 3 * x -5 + rng.randn(50)
pd.DataFrame({
    'input':x,
    'output':y
})
data = pd.DataFrame({
    'input':x,
    'output':y
})
data
np.corrcoef(x, y)
data.corr() # possive linear relationship
plt.scatter(x, y)
plt.show()
dataTemp = data.copy()#Make a copy of data and assigning it to 'dataTemp'
dataTemp
#separate our data into dependent (Y) and independent(X) variables
X_data = dataTemp['input']
Y_data = dataTemp['output']

#Using the train_test_split method to perform a 70/30 test split
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.30)
# Create an instance of linear regression
reg = linear_model.LinearRegression()
#pdTemp(X_test)
pd.DataFrame(X_test)
# Cross validation train test
X_train = pd.DataFrame(X_train)
model = reg.fit(X_train,y_train)
pred = reg.predict(pd.DataFrame(X_test))
#View values in 70/30 cross validation test
print(pred)
print(model)
plt.scatter(X_test, pred,  color='blue')
plt.show()
#Print accuracy score
print('Accuracy Score:', model.score(pd.DataFrame(X_test),pd.DataFrame(y_test)))
# Provide a plot illustrating the Residuals
sns.set(style="darkgrid")
sns.residplot(reg.predict(X_train), reg.predict(X_train)-y_train,lowess=True, color="b")
sns.residplot(reg.predict(pd.DataFrame(X_test)),reg.predict(pd.DataFrame(X_test))-y_test,lowess=True, color="g")
plt.title('Residual Plot using Training (blue) and test (green) data ')
plt.ylabel('Residuals')
plt.xlabel("Predicted Values")
#Determine the Coefficient of Determination (R^2 ) of your model. Explain what this means

print('The Score:', model.score(pd.DataFrame(X_test),pd.DataFrame(y_test)))
reg.intercept_
reg.coef_
Y = 3.00609046 * 0.876343 + -4.9398425018604915
print(Y)
#Calculate and print the result of Mean Absolute Error
print("The Mean Absolute Error: %.2f" % metrics.mean_absolute_error(pd.DataFrame(y_test),pred))
#Calculate and print result of Mean Squared Error
print("The Mean Squared Error: %.2f" % metrics.mean_squared_error(pd.DataFrame(y_test),pred))
#Calculate and print the result of Root Mean Squared Error
print("The Root Mean Squared Error: %.2f" % np.sqrt(metrics.mean_squared_error(pd.DataFrame(y_test),pred)))
