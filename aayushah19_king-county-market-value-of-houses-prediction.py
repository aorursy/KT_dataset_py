import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
#Loading the training data into a variable named data
data = pd.read_csv('../input/king-county-housing-dataset/train.csv')

#Loading the test data into a variable named test
test = pd.read_csv('../input/king-county-housing-dataset/test.csv')
data.shape
#Dropping the following columns in both data and test
data.drop(['Usecode', 'censusblockgroup', 'Latitude', 'Longitude', 
           'BGMedYearBuilt', 'BGPctKids', 'ViewType'], axis=1, inplace = True)
test.drop(['Usecode', 'censusblockgroup', 'Latitude', 'Longitude', 
           'BGMedYearBuilt', 'BGPctKids', 'ViewType'], axis=1, inplace = True)
#Looking at the number of NA's in each column
data.isna().sum()
#Filled the NA's in GarageSquareFeet with 0 in data and test
data['GarageSquareFeet'] = data['GarageSquareFeet'].fillna(0)
test['GarageSquareFeet'] = test['GarageSquareFeet'].fillna(0)
#Filled the NA's in BGMedHomeValue with the mean of the BGMedHomeValue
data['BGMedHomeValue'] = data['BGMedHomeValue'].fillna(data['BGMedHomeValue'].mean())
test['BGMedHomeValue'] = test['BGMedHomeValue'].fillna(test['BGMedHomeValue'].mean())
#Making the train data with no NA's
train = data[(data.BGMedRent.notna())]
#Making the validation set with the observations that have NA's
validation = data[data.BGMedRent.isna()]

#Dividing the validation set into target and explanatory variables
validation_y = validation['BGMedHomeValue'].reset_index(name='BGMedHomeValue').drop(columns='index')
validation_x = validation.loc[:, validation.columns != 'BGMedRent'].drop(columns=['TransDate', 
                                                                                  'PropertyID', 
                                                                                  'SaleDollarCnt', 'ZoneCodeCounty'])

#Dividing the training set into target and explanatory variables
y = train['BGMedHomeValue'].reset_index(name='BGMedHomeValue').drop(columns='index')
x = train.loc[:, train.columns != 'BGMedRent'].drop(columns=['TransDate', 
                                                             'PropertyID', 
                                                             'SaleDollarCnt', 'ZoneCodeCounty'])

#Using train-test-split to divide the training data into train and validation
x_train, x_validation, y_train, y_validation = train_test_split(x, y)

#Using Linear Regression
reg = LinearRegression()
reg.fit(x_train, y_train)
predict = reg.predict(x_validation)

print('The mean absolute error is:', metrics.mean_absolute_error(y_validation, predict))
print('The root mean square error is:', np.sqrt(metrics.mean_squared_error(y_validation, predict)))

#Predicting the values of the validation set
predict_validation = reg.predict(validation_x)
indexes = data[data['BGMedRent'].isna()].index.values

#Putting the values in the indexes where the NA's are present
for i, element in enumerate(indexes):
    data.iloc[element, 12] = predict_validation[i]
#Making the train data with no NA's
train = test[(test.BGMedRent.notna())]
#Making the validation set with the observations that have NA's
validation = test[test.BGMedRent.isna()]

#Dividing the validation set into target and explanatory variables
validation_y = validation['BGMedHomeValue'].reset_index(name='BGMedHomeValue').drop(columns='index')
validation_x = validation.loc[:, validation.columns != 'BGMedRent'].drop(columns=['TransDate', 
                                                                                  'PropertyID', 
                                                                                  'SaleDollarCnt', 'ZoneCodeCounty'])

#Dividing the training set into target and explanatory variables
y = train['BGMedHomeValue'].reset_index(name='BGMedHomeValue').drop(columns='index')
x = train.loc[:, train.columns != 'BGMedRent'].drop(columns=['TransDate', 
                                                             'PropertyID', 
                                                             'SaleDollarCnt', 'ZoneCodeCounty'])

#Using train-test-split to divide the training data into train and validation
x_train, x_validation, y_train, y_validation = train_test_split(x, y)

#Using Linear Regression
reg = LinearRegression()
reg.fit(x_train, y_train)
predict = reg.predict(x_validation)

print('The mean absolute error is:', metrics.mean_absolute_error(y_validation, predict))
print('The root mean square error is:', np.sqrt(metrics.mean_squared_error(y_validation, predict)))

#Predicting the values of the validation set
predict_validation = reg.predict(validation_x)
indexes = test[test['BGMedRent'].isna()].index.values

#Putting the values in the indexes where the NA's are present
for i, element in enumerate(indexes):
    test.iloc[element, 12] = predict_validation[i]
data.isna().sum()
#Using the pandas datetime format to get the month
data['Month'] = pd.to_datetime(data['TransDate']).dt.strftime('%m')
#Displaying the count of each month
data.groupby(data.Month).size().reset_index(name='Count')
#Dropping month
data.drop(['Month'], inplace=True, axis=1)
#Correlation
data.corr().head()
#Making the target variable as SaleDollarCnt
y = data['SaleDollarCnt']
#Making the rest of it as explanatory
x = data.loc[:, data.columns != 'SaleDollarCnt'].drop(columns=['TransDate', 'PropertyID', 'ZoneCodeCounty'])
x1 = test.loc[:, test.columns != 'SaleDollarCnt'].drop(columns=['TransDate', 'PropertyID', 'ZoneCodeCounty'])

#Stored the mean and standard deviations of all the columns
mean_x = x.mean()
std_x = x.std()

mean_x1 = x1.mean()
std_x1 = x1.std()

print(std_x1, std_x)

#Used mean normalization
x = ((x - mean_x) / std_x).to_numpy()
#Two empty lists are initialized for storing the results
aape, mape = [], []

#Used linear_regession
linear_reg = LinearRegression()

#Kfold cross validation is used with 10 splits
cv = KFold(n_splits=10, random_state=42, shuffle=False)

#Going over every iteration and fitted and predicted the values in every iteration
for train_index, validation_index in cv.split(x):
    x_train, x_validation, y_train, y_validation = x[train_index], x[validation_index], y[train_index], y[validation_index]
    linear_reg.fit(x_train, y_train)
    predict_linear_reg = linear_reg.predict(x_validation)
    aape.append((abs(predict_linear_reg - y_validation)/y_validation).mean())
    mape.append((abs(predict_linear_reg - y_validation)/y_validation).median())
    
print('AAPE for linear regression is:', np.mean(aape))
print('MAPE for linear regression is:', np.median(mape))
#Two empty lists are initialized for storing the results
aape, mape = [], []

#Used support vector regession
support_reg = SVR(kernel='poly', degree=5, gamma='scale')

#Kfold cross validation is used with 10 splits
cv = KFold(n_splits=10, random_state=42, shuffle=False)

#Going over every iteration and fitted and predicted the values in every iteration
for train_index, validation_index in cv.split(x):
    x_train, x_validation, y_train, y_validation = x[train_index], x[validation_index], y[train_index], y[validation_index]
    support_reg.fit(x_train, y_train)
    predict_support_reg = support_reg.predict(x_validation)
    aape.append((abs(predict_support_reg - y_validation)/y_validation).mean())
    mape.append((abs(predict_support_reg - y_validation)/y_validation).median())
    
print('AAPE for support vector regression is:', np.mean(aape))
print('MAPE for support vector regression is:', np.median(mape))
#Two empty lists are initialized for storing the results
aape, mape = [], []

#Used adaboost_regession
adaboost_reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=200, learning_rate=0.8)

#Kfold cross validation is used with 10 splits
cv = KFold(n_splits=10, random_state=42, shuffle=False)

#Going over every iteration and fitted and predicted the values in every iteration
for train_index, validation_index in cv.split(x):
    x_train, x_validation, y_train, y_validation = x[train_index], x[validation_index], y[train_index], y[validation_index]
    adaboost_reg.fit(x_train, y_train)
    predict_adaboost_reg = adaboost_reg.predict(x_validation)
    aape.append((abs(predict_adaboost_reg - y_validation)/y_validation).mean())
    mape.append((abs(predict_adaboost_reg - y_validation)/y_validation).median())
    
print('AAPE for Adaboost regression is:', np.mean(aape))
print('MAPE for Adaboost regression is:', np.median(mape))
#Two empty lists are initialized for storing the results
aape, mape = [], []

#Used randomforest
random_forest = RandomForestRegressor(n_estimators = 1000, random_state=42)

#Kfold cross validation is used with 10 splits
cv = KFold(n_splits=10, random_state=42, shuffle=False)

#Going over every iteration and fitted and predicted the values in every iteration
for train_index, validation_index in cv.split(x):
    x_train, x_validation, y_train, y_validation = x[train_index], x[validation_index], y[train_index], y[validation_index]
    random_forest.fit(x_train, y_train)
    predict_random_forest = random_forest.predict(x_validation)
    aape.append((abs(predict_random_forest - y_validation)/y_validation).mean())
    mape.append((abs(predict_random_forest - y_validation)/y_validation).median())
    
print('AAPE for Random forest regression is:', np.mean(aape))
print('MAPE for Random forest regression is:', np.median(mape))