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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn import linear_model



# read the train and test file

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')





# check out the size of the data

print("Train data shape:", train.shape)

print("Test data shape:", test.shape)



print(train.head())



plt.style.use(style='ggplot')

plt.rcParams['figure.figsize'] = (10, 6)



print("Sale Price: \n", train.SalePrice.describe())



# to plot a histogram of SalePrice

print("Skew is:", train.SalePrice.skew())

plt.hist(train.SalePrice, color='blue')

plt.title(" Histogram of Sale Price ")

plt.show()



target = np.log(train.SalePrice)

print("\n Skew is:", target.skew())

plt.hist(target, color='blue')

plt.title("Sale Price")

plt.show()





# return a subset of columns matching the specified data types

numeric_features = train.select_dtypes(include=[np.number])

print(numeric_features.dtypes)



# displays the correlation between the columns and examine the correlations between the features and the target.

corr = numeric_features.corr()

print("1")

print(corr['SalePrice'].sort_values(ascending=False)[:5], '\n')

print(corr['SalePrice'].sort_values(ascending=False)[-5:])

print("2")

# investigate the relationship between OverallQual and SalePrice

quality_pivot = train.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)

print(quality_pivot)



quality_pivot.plot(kind='bar', color='blue')

plt.xlabel('Overall Quality')

plt.ylabel('Median Sale Price')

plt.xticks(rotation=0)

plt.title("Overall Quality Vs Sale Price")

plt.show()





# visualizing the relationship between the Ground Living Area(GrLivArea) and SalePrice

plt.scatter(x=train['GrLivArea'], y=target)

plt.ylabel('Sale Price')

plt.xlabel('Above grade (ground) living area square feet')

plt.title("Ground Living Area Vs Sale Price")

plt.show()



# Null value determination

nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])

nulls.columns = ['Null Count']

nulls.index.name = 'Feature'

print(nulls)



categoricals = train.select_dtypes(exclude=[np.number])

print("Categoricals: \n", categoricals.describe())



train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)

test['enc_street'] = pd.get_dummies(test.Street, drop_first=True)



print('Encoded: \n')

print(train.enc_street.value_counts())  # Pave and Grvl values converted into 1 and 0



# encode  SaleCondition as a new feature by using a similar method that we used for Street above

def encode(x): return 1 if x == 'Partial' else 0

train['enc_condition'] = train.SaleCondition.apply(encode)

test['enc_condition'] = test.SaleCondition.apply(encode)



# exploring this newly modified feature as a plot.

condition_pivot = train.pivot_table(index='enc_condition', values='SalePrice', aggfunc=np.median)

condition_pivot.plot(kind='bar', color='blue')

plt.xlabel('Encoded Sale Condition')

plt.ylabel('Median Sale Price')

plt.xticks(rotation=0)

plt.title("Sale Condition")

plt.show()





data = train.select_dtypes(include=[np.number]).interpolate().dropna()



# Checking if the all of the columns have 0 null values.

print(sum(data.isnull().sum() != 0))





Y = np.log(train.SalePrice)

X = data.drop(['SalePrice', 'Id'], axis=1)

# exclude ID from features since Id is just an index with no relationship to SalePrice.



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=.33)



# Linear Regression Model

lr = linear_model.LinearRegression()



# ---- fit the model / Model fitting

model = lr.fit(X_train, Y_train)





# Calculating Coefficient of determination

print("R^2 is: \n", model.score(X_test, Y_test))



# using the model we have built to make predictions on the test data set.

predictions = model.predict(X_test)



# calculating the Root-mean-square deviation

print('RMSE is: \n', mean_squared_error(Y_test, predictions))



# relationship between predictions and actual_values graphically with a scatter plot.

actual_values = Y_test

plt.scatter(predictions, actual_values, alpha=.75,

            color='b')  # alpha helps to show overlapping data

plt.xlabel('Predicted Price')

plt.ylabel('Actual Price')

plt.title('Linear Regression Model')

plt.show()





# creating a csv that contains the predicted SalePrice for each observation in the test.csv dataset.

submission = pd.DataFrame()

submission['Id'] = test.Id



feats = test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()



# generating predictions

predictions = model.predict(feats)



# transforming the predictions to the correct form

final_predictions = np.exp(predictions)



# checking the difference

print("Original predictions are: \n", predictions[:10], "\n")

print("Final predictions are: \n", final_predictions[:10])



# assigning these predictions and check

submission['SalePrice'] = final_predictions

print("Submission Head: \n", submission.head())



# exporting to a .csv file

submission.to_csv('submission.csv', index=False)


