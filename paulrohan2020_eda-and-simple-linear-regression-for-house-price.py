import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.ensemble import RandomForestRegressor

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import mean_squared_log_error
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df_train.head()
df_test.head()
df_train.describe()
df_train.tail()
# Get the overall concise summary of the DataFrame

df_train.info()
numeric_features = df_train.select_dtypes(include=[np.number])

numeric_features.columns
numeric_features.head()
# Now, as we will be predicting 'SalePrice' lets see description of that column

df_train['SalePrice'].describe()
sns.distplot(df_train['SalePrice'])
# first wo quickly view all the column names in the data

# print(df_train.columns)

# for above I could also use - data.columns

# Checking - 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train['GrLivArea']], axis=1)

data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0, 800000), s=32)
# Checking - 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train['TotalBsmtSF']], axis=1)

data.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0, 800000))
year_feature = [feature for feature in numeric_features if 'Yr' in feature or 'Year' in feature]

year_feature
# Checking - 'YearBuilt'

data = pd.concat([df_train['SalePrice'], df_train['YearBuilt']], axis=1)

data.plot.scatter(x = 'YearBuilt', y = 'SalePrice', ylim=(0, 800000))
# Checking - 'YearBuilt'

data = pd.concat([df_train['SalePrice'], df_train['GarageYrBlt']], axis=1)

data.plot.scatter(x = 'GarageYrBlt', y = 'SalePrice', ylim=(0, 800000))
# Checking - 'YearBuilt'

data = pd.concat([df_train['SalePrice'], df_train['YearRemodAdd']], axis=1)

data.plot.scatter(x = 'YearRemodAdd', y = 'SalePrice', ylim=(0, 800000))
# Checking - 'YearBuilt'

data = pd.concat([df_train['SalePrice'], df_train['YrSold']], axis=1)

data.plot.scatter(x = 'YrSold', y = 'SalePrice', ylim=(0, 800000))

# Checking - TotRmsAbvGrd' i.e. total rooms above grade

data = pd.concat([df_train['SalePrice'], df_train['TotRmsAbvGrd']], axis=1)

data.plot.scatter(x = 'TotRmsAbvGrd', y = 'SalePrice', ylim=(0, 800000))

data = pd.concat([df_train['SalePrice'], df_train['OverallQual']], axis=1)

f, ax = plt.subplots(figsize=(14, 8))

fig = sns.boxplot(x='OverallQual', y='SalePrice', data=data)

fig.axis(ymin= 0, ymax=800000)


# Now lets view the top 8 correlated features with the sale price:

corr_mat = df_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corr_mat, vmax=0.8, square=True)
top_correlations = df_train.corr()

top_feature_columns = top_correlations['SalePrice'][top_correlations['SalePrice'].values > 0.2].index.values

top_feature_columns
# Handling Missing Values for 19 features which have missing values mentioned above

df_train['GarageYrBlt'] = df_train['GarageYrBlt'].fillna(0)

# filling in missing GarageYrBuilt values with zeros.  

# But this may not be the most logical approach - refer to this discussion below for mor perspective

# https://www.kaggle.com/c/house-prices-advanced-regression-techniques/discussion/60143



# similary fillingup na valuse for couple of other features

df_train['LotFrontage'] = df_train.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

df_train['MasVnrArea'] = df_train['MasVnrArea'].fillna(0)



heat_map_with_top_correlated_features = np.append(top_feature_columns[-12:], np.array(['SalePrice']))

pearson_correlation_coefficients = np.corrcoef(df_train[heat_map_with_top_correlated_features[::-1]].T)

plt.figure(figsize=(16,16))

sns.set(font_scale=1)

with sns.axes_style('white'):

    sns.heatmap(pearson_correlation_coefficients, yticklabels=heat_map_with_top_correlated_features[::-1], xticklabels=heat_map_with_top_correlated_features[::-1], fmt='.2f', annot_kws={'size': 10}, annot=True, square=True, cmap=None)

    

    
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt', 'FullBath', 'OpenPorchSF', 

        'WoodDeckSF', '2ndFlrSF', 'YearRemodAdd', 'MasVnrArea', 'LotFrontage', 'LotArea']

sns.pairplot(df_train[cols], height=4)

plt.show()
total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
def loss_func(y_predicted, y_actual):

    squared_error = (y_predicted - y_actual) ** 2

    sum_squared_error = squared_error.sum()

    size = len(y_actual)

    return 1/(2*size) * sum_squared_error
x = df_train['GrLivArea']

y = df_train['SalePrice']



x = (x - x.mean()) / x.std()

x = np.c_[np.ones(x.shape[0]), x]

x.shape
x = df_train['GrLivArea']

y = df_train['SalePrice']
x = (x - x.mean()) / x.std()

print(x.shape)

# (1460, ) i.e. it will be an array like

# [0 1 2 3 4 5 6 7 8.... 1459]



x = np.c_[np.ones(x.shape[0]), x]

print(x.shape)

# (1460, 2)
# Now below I am defining a basic Linear regression function



class SimpleLinearRegression:



    def get_predictions(self, X):

        return np.dot(X, self._W)



    def _get_gradient_descent_step(self, X, targets, learning_rate):

        predictions = self.get_predictions(X)



        error = predictions - targets

        gradient = np.dot(X.T, error)/len(x)



        # now update the W

        self._W -= learning_rate * gradient





    def fit(self, X, y, iterations_num=1000, learning_rate=0.01):

            self._W = np.zeros(X.shape[1])



            self._history_of_cost = []

            self._w_history = [self._W]



            for i in range(iterations_num):

                predictions = self.get_predictions(X)

                cost = loss_func(predictions, y)



                self._history_of_cost.append(cost)



                self._get_gradient_descent_step(x, y, learning_rate)



                self._w_history.append(self._W.copy())



            return self

house_price_linear_result = SimpleLinearRegression()

house_price_linear_result.fit(x, y, iterations_num=2000, learning_rate=0.01)

print(house_price_linear_result._W)
# Now plot the Cost function

plt.title('Kaggle House Price Cost Function')

plt.xlabel('No of Iterations')

plt.ylabel('House Price Cost')

plt.plot(house_price_linear_result._history_of_cost)

plt.show()

# We have already taken care of NA values for some the below features

x = df_train[['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 

'YearBuilt', 'FullBath', 'OpenPorchSF', 'WoodDeckSF', '2ndFlrSF', 'YearRemodAdd', 

              'MasVnrArea', 'LotFrontage', 'LotArea']]



# just like before let's standardize our independent variable data

x = (x - x.mean()) / x.std()

x = np.c_[np.ones(x.shape[0]), x]



house_price_multi_variate_linear_result = SimpleLinearRegression()

house_price_multi_variate_linear_result.fit(x, y, iterations_num = 2000, learning_rate=0.01)



print(house_price_multi_variate_linear_result._W)
plt.title('Kaggle House Price Cost Function-Multivariate Case')

plt.xlabel('No of Iterations')

plt.ylabel('House Price Cost')

plt.plot(house_price_multi_variate_linear_result._history_of_cost)

plt.show()


train_target_label = df_train['SalePrice']

# labels are dependent variables whose values are to be predicted. 

top_feature_columns_modified = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt', 'FullBath', 'OpenPorchSF', 

        'WoodDeckSF', '2ndFlrSF', 'YearRemodAdd', 'MasVnrArea', 'LotFrontage', 'LotArea']



training_sample_df = df_train[top_feature_columns_modified]

test_sample_df = df_test[top_feature_columns_modified]



training_sample_df.head()
test_sample_df.head()
imputer = SimpleImputer(strategy = 'median')



# During fit() the imputer learns about the mean, median etc of the data,

# which is then applied to the missing values during transform().

imputer.fit(training_sample_df)

imputer.fit(test_sample_df)



# Note - sklearn.preprocessing.Imputer.fit_transform returns a new array, it doesn't alter the argument array.

training_sample_df = imputer.transform(training_sample_df)

test_sample_df = imputer.transform(test_sample_df)
scaler = StandardScaler()



# Again, during fit() the imputer learns about the mean, median etc of the data,

# which is then applied to the missing values during transform().

scaler.fit(training_sample_df)

scaler.fit(test_sample_df)



training_sample_df = scaler.transform(training_sample_df)

test_sample_df = scaler.transform(test_sample_df)
# Syntax of train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.33, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(training_sample_df, train_target_label, random_state=42, train_size = 0.8 )



X_train.shape
X_test.shape
training_sample_df.shape
test_sample_df.shape
model = LinearRegression()

# we have to fit this model to our data, in other words, we have to make it “learn” using our training data.

# The syntax of the fit function is just plain model.fit(X_train, Y_train)

model.fit(training_sample_df, train_target_label)
print('Intercept is ', model.intercept_)



# For retrieving the slope (coefficient of x). This will be an array of values.

print("Slope i.e. coefficient of x is ", model.coef_)
# regression = model.fit(training_sample_df, train_target_label)

print("Regression score is", model.score(training_sample_df, train_target_label))

print('train_target_label is ', train_target_label)
y_predict = model.predict(X_test)
rmse_simple_linear = np.sqrt(metrics.mean_squared_error(y_test, y_predict))

print('Root Mean Square Error is ', rmse_simple_linear)



# MSE_Log

rmse_log_simple_linear = np.sqrt(mean_squared_log_error(y_test, y_predict))

print('Root Mean Square Log Error is ', rmse_log_simple_linear)

prediction_on_test_data = model.predict(test_sample_df)

# print("Regression score on test sample data is", model.score(test_sample_df, train_target_label ))

testID = df_test['Id']

predict_submission = pd.DataFrame()

predict_submission['ID'] = testID

predict_submission['SalePrice'] = prediction_on_test_data

predict_submission
model_ridge = Ridge(alpha=0.5)

model_ridge.fit(training_sample_df, train_target_label)

y_predict_ridge = model_ridge.predict(X_test)



mse_linear_ridge = np.sqrt(metrics.mean_squared_error(y_test, y_predict_ridge))

print('MSE of Linear Ridge is ', mse_linear_ridge)



# Log Error

mse_log_linear_ridge = np.sqrt(mean_squared_log_error(y_test, y_predict_ridge))

print('MSE Log of Linear Ridge is ', mse_log_linear_ridge)

model_lasso = Lasso(alpha=33)

model_lasso.fit(training_sample_df, train_target_label)

y_predict_lasso = model_lasso.predict(X_test)



mse_linear_lasso = np.sqrt(metrics.mean_squared_error(y_test, y_predict_lasso))

print('MSE of Lasso Regression is ', mse_linear_lasso)



# Log Error

mse_log_linear_lasso = np.sqrt(mean_squared_log_error(y_test, y_predict_lasso))

print('MSE Log of Lasso Regression is ', mse_log_linear_lasso)
RFR = RandomForestRegressor(max_depth=50)

RFR.fit(training_sample_df, train_target_label)



y_predict_random_forest = RFR.predict(X_test)



mse_random_forest = np.sqrt(metrics.mean_squared_error(y_test, y_predict_random_forest))

print('MSE Random Forest is ', mse_random_forest)



# Log Error

mse_log_random_forest = np.sqrt(mean_squared_log_error(y_test, y_predict_random_forest))

print('MSE Log Random Forest is ', mse_log_random_forest)
# Y_test_predicted_for_submission = RFR.predict(df_test)

# Y_test_predicted_for_submission = RFR.predict(X_test)

Y_test_predicted_for_submission = RFR.predict(test_sample_df)



indexes = np.arange(df_test.shape[0]+2, 2*df_test.shape[0]+2)

print('Indexex ', indexes)



# output_for_submission = pd.DataFrame({'Id': test_sample_df.Id,

#                        'SalePrice': Y_test_predicted_for_submission})



# output_for_submission = pd.DataFrame({'Id': indexes,

#                        'SalePrice': Y_test_predicted_for_submission})



# output_for_submission = pd.DataFrame({'Id': df_test.Id,

#                        'SalePrice': Y_test_predicted_for_submission})



# print(output_for_submission)



# *****************

testID = df_test['Id']

output_for_submission = pd.DataFrame()

output_for_submission['ID'] = testID

output_for_submission['SalePrice'] = Y_test_predicted_for_submission

output_for_submission.to_csv('submission.csv', index=False)
