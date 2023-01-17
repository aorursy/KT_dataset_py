# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
from sklearn.model_selection import train_test_split 
from sklearn import linear_model
from sklearn import linear_model

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = 2 * 4 + x + rng.randn(50)
pd.DataFrame({
    'input':x,
    'output':y
})
data = pd.DataFrame({
    'input':x,
    'output':y
})
data
pddf = pd.DataFrame
data.corr()
np.corrcoef(x,y)
plt.scatter(x,y)
plt.show()
data2 = data.copy()

# separate our data into dependent (Y) and independent(X) variables
X_data = data2['input']
Y_data = data2['output']

# We will split the data using a 70/30 split. 
# i.e. 70% of the data will be randomly chosen to train the model and 30% will be used to evaluate the model

X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.30)
# Create an instance of linear regression
reg = linear_model.LinearRegression()

pd.DataFrame(X_test)
X_train = pddf(X_train)
model = reg.fit(X_train,y_train)
predictions = reg.predict(pddf(X_test))
predictions
plt.scatter(X_test,predictions, color='black')
plt.show()
import seaborn as sns
sns.set(style="whitegrid")

# Plot the residuals after fitting a linear model
sns.residplot(X_test, y_test, lowess=True, color="b")
import seaborn as sns
sns.set(style="darkgrid")

# Plot the residuals after fitting a linear model
sns.residplot(reg.predict(X_train), reg.predict(X_train)-y_train, lowess=True, color="r")
sns.residplot(reg.predict(pddf(X_test)), reg.predict(pddf(X_test))-y_test, lowess=True, color="g")
plt.title('Residual Plot using Training (red) and test (green) data ')
plt.ylabel('Residuals')
#printing accuracy score
print('Score: ', model.score(pddf(X_test), pddf(y_test)))
reg.coef_
reg.intercept_
y = 0.99566088 * 7.890373 + 8.11269206815173
y
from sklearn import metrics

# (a) Mean Absolute Error
print('MAE: %.2f' % metrics.mean_absolute_error(pddf(y_test), predictions))

# (b) Mean Squared Error
print('MSE: %.2f' % metrics.mean_squared_error(pddf(y_test), predictions))

# (a) Root Mean Squared Error
print('RMSE: %.2f' % np.sqrt(metrics.mean_squared_error(pddf(y_test), predictions)))
