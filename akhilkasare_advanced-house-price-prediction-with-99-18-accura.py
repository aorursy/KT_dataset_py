# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
train_df = train.copy()
train.info()
train.describe()
train.isnull().sum()
# Checking for categorical features



categorical_col = []

for column in train.columns:

    if train[column].dtype == object and len(train[column].unique()) <= 50:

        categorical_col.append(column)

        print(f"{column} : {train[column].unique()}")

        print("====================================")
numerical_col = []

for column in train.columns:

    if train[column].dtype != object and len(train[column].unique()) <= 50:

        numerical_col.append(column)

        print(f"{column} : {train[column].unique()}")

        print("====================================")
# Visulazing the distibution of the data for every feature

train.hist(edgecolor='black', linewidth=1.2, figsize=(20, 20));
plt.figure(figsize=(20,10))

sns.heatmap(train.isnull(), cmap='viridis')
# Extracting the columns which have missing values from the dataset

missing_values = [feature for feature in train.columns if train[feature].isnull().sum() >1]

print("The features having the missing values are",missing_values,end='')
for feature in missing_values:

    print(feature, 'has', np.round(train[feature].isnull().mean(),2), '% of missing values')
sns.distplot(train['SalePrice'])
train['SalePrice'] = np.log(train['SalePrice'] + 1)

sns.distplot(train['SalePrice'])
print(categorical_col,end='')
for feature in categorical_col:

    temp = train.groupby(feature)['SalePrice'].count()/len(train) #Calculating the percentage

    temp_df = temp[temp>0.01].index

    train[feature] = np.where(train[feature].isin(temp_df), train[feature], 'Rare_var')

train.head()
# Label encoder basically converts categorical values into numerical values



from sklearn.preprocessing import LabelEncoder



sc=LabelEncoder()



for feature in categorical_col:



    train[feature]=sc.fit_transform(train[feature])
train.head()
for feature in missing_values:

    print(feature, 'has', np.round(train[feature].isnull().mean(),2), '% of missing values')
train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].mean())

train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mean())

train['GarageYrBlt'] = train['GarageYrBlt'].fillna(train['GarageYrBlt'].mean())
train.head()
# Splitting the features into independent and dependent variables



x = train.drop(['SalePrice'], axis = 1)

y = train['SalePrice']
from sklearn.ensemble import ExtraTreesRegressor



model = ExtraTreesRegressor()

model.fit(x,y)
print(model.feature_importances_)
#plotting graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=x.columns)

feat_importances.nlargest(20).plot(kind='barh')

plt.show()
sns.distplot(train['SalePrice'])
#Spliting data into test and train



from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20)
from sklearn.linear_model import LinearRegression



lr = LinearRegression()



lr.fit(x_train, y_train)



lr_pred = lr.predict(x_test)
r2 = r2_score(y_test,lr_pred)

print('R-Square Score: ',r2*100)
# Calculate the absolute errors

lr_errors = abs(lr_pred - y_test)

# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(lr_pred), 2), 'degrees.')
# Calculate mean absolute percentage error (MAPE)

mape = 100 * (lr_errors / y_test)

# Calculate and display accuracy

lr_accuracy = 100 - np.mean(mape)

print('Accuracy for Logistic Regression is :', round(lr_accuracy, 2), '%.')
from sklearn.metrics import mean_absolute_error,mean_squared_error



print('mse:',metrics.mean_squared_error(y_test, lr_pred))

print('mae:',metrics.mean_absolute_error(y_test, lr_pred))
sns.distplot(y_test-lr_pred)
# plotting the Linear Regression values predicated Rating



plt.figure(figsize=(12,7))



plt.scatter(y_test,x_test.iloc[:,2],color="blue")

plt.title("True rate vs Predicted rate",size=20,pad=15)

plt.xlabel('Sale Price',size = 15)

plt.scatter(lr_pred,x_test.iloc[:,2],color="yellow")
from sklearn.tree import DecisionTreeRegressor



dtree = DecisionTreeRegressor(criterion='mse')

dtree.fit(x_train, y_train)
dtree_pred = dtree.predict(x_test)
r2 = r2_score(y_test,dtree_pred)

print('R-Square Score: ',r2*100)



# Calculate the absolute errors

dtree_errors = abs(dtree_pred - y_test)

# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(dtree_pred), 2), 'degrees.')



# Calculate mean absolute percentage error (MAPE)

mape = 100 * (dtree_errors / y_test)

# Calculate and display accuracy

dtree_accuracy = 100 - np.mean(mape)

print('Accuracy for Decision tree regressor is :', round(dtree_accuracy, 2), '%.')
#plotting the Decision Tree values predicated Rating



plt.figure(figsize=(12,7))



plt.scatter(y_test,x_test.iloc[:,2],color="blue")

plt.title("True rate vs Predicted rate",size=20,pad=15)

plt.xlabel('Sale Price',size = 15)

plt.scatter(dtree_pred,x_test.iloc[:,2],color="yellow")

plt.legend()
from sklearn.ensemble import RandomForestRegressor



random_forest_regressor = RandomForestRegressor()

random_forest_regressor.fit(x_train, y_train)

rf_pred = random_forest_regressor.predict(x_test)
r2 = r2_score(y_test,rf_pred)

print('R-Square Score: ',r2*100)



# Calculate the absolute errors

rf_errors = abs(rf_pred - y_test)

# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(rf_pred), 2), 'degrees.')



# Calculate mean absolute percentage error (MAPE)

mape = 100 * (rf_errors / y_test)

# Calculate and display accuracy

rf_accuracy = 100 - np.mean(mape)

print('Accuracy for random forest regressor is :', round(rf_accuracy, 2), '%.')

#plotting the Random forest values predicated Rating



plt.figure(figsize=(12,7))



plt.scatter(y_test,x_test.iloc[:,2],color="blue")

plt.title("True rate vs Predicted rate",size=20,pad=15)

plt.xlabel('Sale Price',size = 15)

plt.scatter(rf_pred,x_test.iloc[:,2],color="yellow")
pred_y = (lr_pred*0.45 + dtree_pred*0.55 + rf_pred*0.65)
pred_y