import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
# Input data files are available in the "../input/" directory.
from subprocess import check_output

data = pd.read_csv('../input/kc_house_data.csv', usecols = ['sqft_living', 'price'], sep = ',')
print(data.shape)
sft = data.iloc[:, 1:2].values
price = data.iloc[:, 0:1].values
sft_train, sft_test, price_train, price_test = train_test_split(sft, price, test_size = 0.2776, random_state = 0)
fig, ax = plt.subplots(1)
sns.regplot(x= sft_train, y= price_train, color = 'c', fit_reg = False, scatter = True)
plt.title('Square feet vs Price (Training set)')
plt.xlabel('Square feet')
plt.ylabel('Price')
ax.set_yticklabels([])
plt.show()
fig, ax = plt.subplots(1)
sns.regplot(x= sft_test, y= price_test, color = 'b', fit_reg = False, scatter = True)
plt.title('Square feet vs Price (Test set)')
plt.xlabel('Square feet')
plt.ylabel('Price')
ax.set_yticklabels([])
plt.show()
linreg = LinearRegression()
linreg.fit(sft_train, price_train)
print(linreg.coef_, linreg.intercept_)
predictions = linreg.predict(sft_test)
Errors = predictions - price_test
RSS = np.sum(np.square(Errors))
print(RSS)
fig, ax = plt.subplots(1)
sns.regplot(x= sft_test, y= price_test, color = 'b', fit_reg = False, scatter = True)
plt.scatter(x = sft_test, y = linreg.predict(sft_test), color = 'r')
plt.title('Square feet vs Price')
plt.xlabel('Square feet')
plt.ylabel('Price')
ax.set_yticklabels([])
plt.show()
sns.regplot(x= linreg.predict(sft_test), y= Errors, color= 'c', fit_reg= False, scatter= True)
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Checking scedasticity of linear model')
plt.show()
sns.distplot(Errors)
plt.ylabel('Residual magnitude')
plt.title('Residual distribution')