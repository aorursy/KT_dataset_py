# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 

import pandas as pd 

%matplotlib inline

import matplotlib.pyplot as plt  

import seaborn as sns

from scipy import stats

from scipy.stats import norm, skew 



sub1 = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.head(5)
test.head(5)
print(train.shape)

print(test.shape)

# their shapes
print(train.shape)

print(test.shape)


plt.style.use(style = 'ggplot')

plt.rcParams['figure.figsize'] = (10, 6)
train.SalePrice.describe()
# Checking skewness

print ("SKEW is:", train.SalePrice.skew())

plt.hist(train.SalePrice, color = 'blue')

plt.show()
target = np.log(train.SalePrice)

print ("SKEW is:", target.skew())

plt.hist(target, color = 'blue')

plt.show()
# Working with numeric types

numeric_features = train.select_dtypes(include = [np.number])

numeric_features.dtypes
corr = numeric_features.corr()
# 5 features most positevely correlated with SalePrice

print (corr['SalePrice'].sort_values(ascending = False)[:5])
# 5 features most negatively correlated with SalePrice

print (corr['SalePrice'].sort_values(ascending = False)[-5:])
train.OverallQual.unique()
quality_p = train.pivot_table(index = 'OverallQual', values = 'SalePrice', aggfunc = np.median)

quality_p
#Visualizing the pivot table

quality_p.plot(kind = 'bar', color = 'blue')

plt.xlabel('Overall Quality')

plt.ylabel('Median Sale Price')

plt.xticks(rotation = 0)

plt.show()
plt.scatter(x = train.GrLivArea, y = target)

plt.ylabel('Sales Price')

plt.xlabel('Above grade living area in square feet')

plt.show()



# We can see that median sales price strictly increases as Overall Quality increases
#Doing the previous step again for the garage area

plt.scatter(x = train.GarageArea, y = target)

plt.ylabel('Sale Price')

plt.xlabel('Garage Area')

plt.show()
train = train[train.GarageArea < 1200]



plt.scatter(x = train.GarageArea, y = np.log(train.SalePrice))

plt.xlim(-200, 1600) 

plt.ylabel('Sale Price')

plt.xlabel('Garage Area')

plt.show()
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending = False))

nulls.columns = ['Null Count']

nulls.index.name = 'Feature'

pd.set_option('display.max_rows', None)

nulls

# Handling missing values


print ("Unique values are:", train.MiscFeature.unique())
categoricals = train.select_dtypes(exclude = [np.number])

categoricals.describe()
print ("Original:\n")

print (train.Street.value_counts(), '\n')

# testing with Street data
# Using one-hot encoding to transform the data into a Boolean column

train.enc_street = pd.get_dummies(train.Street, drop_first = True)

test.enc_street = pd.get_dummies(train.Street, drop_first = True)

condition_pivot = train.pivot_table(index = 'SaleCondition', values = 'SalePrice', aggfunc = np.median)

condition_pivot.plot(kind = 'bar', color = 'blue')

plt.xlabel('Sale Condition')

plt.ylabel('Median Sale Price')

plt.xticks(rotation = 0)

plt.show()
data = train.select_dtypes(include = [np.number]).interpolate().dropna()

# check if all of the columns have 0 NULL values

sum(data.isnull().sum() != 0)
y = np.log(train.SalePrice)

x = data.drop(['SalePrice', 'Id'], axis = 1)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 42, test_size = .33)
from sklearn import linear_model

lr = linear_model.LinearRegression()



#Now, we fit the model

model = lr.fit(x_train, y_train)
print ("R-squared is: \n", model.score(x_test, y_test))
predictions = model.predict(x_test)



from sklearn.metrics import mean_squared_error

print ("RMSE is: \n", mean_squared_error(y_test, predictions))
actual_values = y_test

#Alpha helps to show overlapping data

plt.scatter(predictions, actual_values, alpha = .75, color = 'b')

plt.xlabel('Predicted Price')

plt.ylabel('Actual Price')

plt.title('Linear Regression Model')

plt.show()
submission = pd.DataFrame()

submission['id'] = test.Id



#Selecting the features from the test data

feats = test.select_dtypes(include = [np.number]).drop(['Id'], axis = 1).interpolate()



#Generating our predictions

predictions = model.predict(feats)



#Transforming the predictions to the correct form

final_predictions = np.exp(predictions)



print ("Original predictions are: \n", predictions[: 5], "\n")

print ("Final predictions are: \n", final_predictions[: 5])
submission['SalePrice'] = final_predictions

submission.head()
submission.to_csv('submission1.csv', index = False)