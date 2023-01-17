import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
train_data = pd.read_csv("../input/bengaluru-house-price-data/Bengaluru_House_Data.csv")

test_data = pd.read_csv("../input/housepricetestdata/Predicting-House-Prices-In-Bengaluru-Test-Data.csv")
train_data.head()
test_data.head()
train_data.describe()
test_data.describe()
train_data.isnull().sum()
test_data.isnull().sum()
train_data.dtypes
test_data.dtypes
def change_to_float(area_size):

    if isinstance(area_size, str):

        area_size = area_size.split('Sq.')[0]

        area_size = area_size.split('Perch')[0]

        area_size = area_size.split('Acres')[0]

        area_size = area_size.split('Guntha')[0]

        area_size = area_size.split('Grounds')[0]

        area_size = area_size.split('Cents')[0]

        area_size = area_size.split('-')

        area_size = list(map(float,area_size))

        area_size = sum(area_size)  / len(area_size)

    return area_size
train_data['total_sqft'] = train_data['total_sqft'].apply(lambda x : change_to_float(x))
size_mode = train_data['size'].mode()[0]

train_data.loc[train_data['size'].isna(), 'size'] = size_mode
train_data['total_sqft'] = train_data['total_sqft'].astype('float64')
train_data['size'] = train_data['size'].apply(lambda x: x.split(' ')[0])
train_data['size'] = train_data['size'].astype('float64')
train_data.head()
test_data['total_sqft'] = test_data['total_sqft'].apply(lambda x : change_to_float(x))
size_mode = test_data['size'].mode()[0]

test_data.loc[test_data['size'].isna(), 'size'] = size_mode
test_data['total_sqft'] = test_data['total_sqft'].astype('float64')
test_data['size'] = test_data['size'].apply(lambda x: x.split(' ')[0])
test_data['size'] = test_data['size'].astype('float64')
test_data.head()
traincorr = train_data.corr()
testcorr = test_data.corr()
plt.figure(figsize=(10,10))

sns.heatmap(traincorr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':10}, cmap='Greens')
plt.figure(figsize=(10,10))

sns.heatmap(testcorr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':10}, cmap='Greens')
med_bal_train = train_data['balcony'].median ()

print (med_bal_train)
med_bal_test = test_data['balcony'].median ()

print (med_bal_test)
med_bath_train = train_data['bath'].median ()

print (med_bath_train)
med_bath_test = test_data['bath'].median ()

print (med_bath_test)
mod_bal_train = train_data['balcony'].mode ()

print (mod_bal_train)
mod_bal_test = test_data['balcony'].mode ()

print (mod_bal_test)
train_data.drop(['society'],axis=1, inplace = True)

train_data.head()
test_data.drop(['society'],axis=1, inplace = True)

test_data.head()
train_data.loc[train_data['balcony'].isna(), 'balcony'] = med_bal_train
test_data.loc[test_data['balcony'].isna(), 'balcony'] = med_bal_test
train_data.loc[train_data['bath'].isna(), 'bath'] = med_bath_train
test_data.loc[test_data['bath'].isna(), 'bath'] = med_bath_test
train_data.isnull().sum()
test_data.isnull().sum()
X_train = train_data[['size', 'total_sqft', 'bath', 'balcony']]

y_train = train_data[['price']]

X_test = test_data[['size', 'total_sqft', 'bath', 'balcony']]
#outlier analysis

for column_name in X_train.columns:

    q1 = X_train[column_name].quantile(0.25)

    q3 = X_train[column_name].quantile(0.75)

    iqr = q3-q1

    upper = q3 +1.5*iqr

    lower = q1 -1.5*iqr

    outliner_df = X_train.loc[(X_train[column_name] < lower)| (X_train[column_name] > upper)]

    print('Percentage of outlinear in {0} is {1}'.format(column_name, outliner_df.shape[0]/X_train.shape[0]))
from scipy.stats import normaltest
normaltest(train_data[['size','total_sqft','bath','balcony']])

#p-value should be lesser than 0.05 means the data is not normally distributed
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X_train, y_train)
y_test = lr.predict(X_test)
lr.intercept_
y_train.shape
y_test.shape
from sklearn.preprocessing import RobustScaler
rs = RobustScaler()
rs.fit(X_train)

rs.fit(X_test)

X_train_scaled = rs.transform(X_train)

X_test_scaled = rs.transform(X_test)
# Train the model using the training sets 

lr.fit(X_train_scaled, y_train)
y_test_scaled = lr.predict(X_test_scaled)
lr.intercept_