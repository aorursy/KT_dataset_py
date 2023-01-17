import numpy as np # NumPy is the fundamental package for scientific computing



import pandas as pd # Pandas is an easy-to-use data structures and data analysis tools

pd.set_option('display.max_columns', None) # To display all columns



import matplotlib.pyplot as plt # Matplotlib is a python 2D plotting library

%matplotlib inline 

# A magic command that tells matplotlib to render figures as static images in the Notebook.



import seaborn as sns # Seaborn is a visualization library based on matplotlib (attractive statistical graphics).

sns.set_style('whitegrid') # One of the five seaborn themes

import warnings

warnings.filterwarnings('ignore') # To ignore some of seaborn warning msg



from scipy import stats



from sklearn import linear_model # Scikit learn library that implements generalized linear models

from sklearn import neighbors # provides functionality for unsupervised and supervised neighbors-based learning methods

from sklearn.metrics import mean_squared_error # Mean squared error regression loss

from sklearn import preprocessing # provides functions and classes to change raw feature vectors



from math import log
data = pd.read_csv("../input/kc_house_data.csv", parse_dates = ['date']) # load the data into a pandas dataframe

data.head(2) # Show the first 2 lines
data.drop(['id', 'date'], axis = 1, inplace = True)
data['basement_present'] = data['sqft_basement'].apply(lambda x: 1 if x > 0 else 0) # Indicate whether there is a basement or not

data['renovated'] = data['yr_renovated'].apply(lambda x: 1 if x > 0 else 0) # 1 if the house has been renovated
categorial_cols = ['floors', 'view', 'condition', 'grade']



for cc in categorial_cols:

    dummies = pd.get_dummies(data[cc], drop_first=False)

    dummies = dummies.add_prefix("{}#".format(cc))

    data.drop(cc, axis=1, inplace=True)

    data = data.join(dummies)
dummies_zipcodes = pd.get_dummies(data['zipcode'], drop_first=False)

dummies_zipcodes.reset_index(inplace=True)

dummies_zipcodes = dummies_zipcodes.add_prefix("{}#".format('zipcode'))

dummies_zipcodes = dummies_zipcodes[['zipcode#98004','zipcode#98102','zipcode#98109','zipcode#98112','zipcode#98039','zipcode#98040']]

data.drop('zipcode', axis=1, inplace=True)

data = data.join(dummies_zipcodes)



data.dtypes
from sklearn.cross_validation import train_test_split

train_data, test_data = train_test_split(data, train_size = 0.8, random_state = 10)
# A function that take one input of the dataset and return the RMSE (of the test data), and the intercept and coefficient

def simple_linear_model(train, test, input_feature):

    regr = linear_model.LinearRegression() # Create a linear regression object

    regr.fit(train.as_matrix(columns = [input_feature]), train.as_matrix(columns = ['price'])) # Train the model

    RMSE = mean_squared_error(test.as_matrix(columns = ['price']), 

                              regr.predict(test.as_matrix(columns = [input_feature])))**0.5 # Calculate the RMSE on test data

    return RMSE, regr.intercept_[0], regr.coef_[0][0]
RMSE, w0, w1 = simple_linear_model(train_data, test_data, 'sqft_living')

print ('RMSE for sqft_living is: %s ' %RMSE)

print ('intercept is: %s' %w0)

print ('coefficient is: %s' %w1)
input_list = data.columns.values.tolist() # list of column name

input_list.remove('price')

simple_linear_result = pd.DataFrame(columns = ['feature', 'RMSE', 'intercept', 'coefficient'])



# loop that calculate the RMSE of the test data for each input 

for p in input_list:

    RMSE, w1, w0 = simple_linear_model(train_data, test_data, p)

    simple_linear_result = simple_linear_result.append({'feature':p, 'RMSE':RMSE, 'intercept':w0, 'coefficient': w1}

                                                       ,ignore_index=True)

simple_linear_result.sort_values('RMSE').head(10) # display the 10 best estimators
# A function that take multiple features as input and return the RMSE (of the test data), and the  intercept and coefficients

def multiple_regression_model(train, test, input_features):

    regr = linear_model.LinearRegression() # Create a linear regression object

    regr.fit(train.as_matrix(columns = input_features), train.as_matrix(columns = ['price'])) # Train the model

    RMSE = mean_squared_error(test.as_matrix(columns = ['price']), 

                              regr.predict(test.as_matrix(columns = input_features)))**0.5 # Calculate the RMSE on test data

    return RMSE, regr.intercept_[0], regr.coef_ 
print ('RMSE: %s, intercept: %s, coefficients: %s' %multiple_regression_model(train_data, 

                                                                             test_data, ['sqft_living','bathrooms','bedrooms']))

print ('RMSE: %s, intercept: %s, coefficients: %s' %multiple_regression_model(train_data, 

                                                                             test_data, ['sqft_above','view#0','bathrooms']))

print ('RMSE: %s, intercept: %s, coefficients: %s' %multiple_regression_model(train_data, 

                                                                             test_data, ['bathrooms','bedrooms']))

print ('RMSE: %s, intercept: %s, coefficients: %s' %multiple_regression_model(train_data, 

                                                                             test_data, ['view#0','grade#12','bedrooms','sqft_basement']))

print ('RMSE: %s, intercept: %s, coefficients: %s' %multiple_regression_model(train_data, 

                                                                             test_data, ['sqft_living','bathrooms','view#0']))
train_data['sqft_living_squared'] = train_data['sqft_living'].apply(lambda x: x**2) # create a new column in train_data

test_data['sqft_living_squared'] = test_data['sqft_living'].apply(lambda x: x**2) # create a new column in test_data

print ('RMSE: %s, intercept: %s, coefficients: %s' %multiple_regression_model(train_data, 

                                                                             test_data, ['sqft_living','sqft_living_squared']))
# we're first going to add more features into the dataset.



# sqft_living cubed

train_data['sqft_living_cubed'] = train_data['sqft_living'].apply(lambda x: x**3) 

test_data['sqft_living_cubed'] = test_data['sqft_living'].apply(lambda x: x**3) 



# bedrooms_squared: this feature will mostly affect houses with many bedrooms.

train_data['bedrooms_squared'] = train_data['bedrooms'].apply(lambda x: x**2) 

test_data['bedrooms_squared'] = test_data['bedrooms'].apply(lambda x: x**2)



# bedrooms times bathrooms gives what's called an "interaction" feature. It is large when both of them are large.

train_data['bed_bath_rooms'] = train_data['bedrooms']*train_data['bathrooms']

test_data['bed_bath_rooms'] = test_data['bedrooms']*test_data['bathrooms']



# Taking the log of squarefeet has the effect of bringing large values closer together and spreading out small values.

train_data['log_sqft_living'] = train_data['sqft_living'].apply(lambda x: log(x))

test_data['log_sqft_living'] = test_data['sqft_living'].apply(lambda x: log(x))



train_data.shape
# split the train_data to include a validation set (train_data2 = 60%, validation_data = 20%, test_data = 20%)

train_data_2, validation_data = train_test_split(train_data, train_size = 0.75, random_state = 50)
# A function that take multiple features as input and return the RMSE (of the train and validation data)

def RMSE(train, validation, features, new_input):

    features_list = list(features)

    features_list.append(new_input)

    regr = linear_model.LinearRegression() # Create a linear regression object

    regr.fit(train.as_matrix(columns = features_list), train.as_matrix(columns = ['price'])) # Train the model

    RMSE_train = mean_squared_error(train.as_matrix(columns = ['price']), 

                              regr.predict(train.as_matrix(columns = features_list)))**0.5 # Calculate the RMSE on train data

    RMSE_validation = mean_squared_error(validation.as_matrix(columns = ['price']), 

                              regr.predict(validation.as_matrix(columns = features_list)))**0.5 # Calculate the RMSE on train data

    return RMSE_train, RMSE_validation 
input_list = train_data_2.columns.values.tolist() # list of column name

input_list.remove('price')



# list of features included in the regression model and the calculated train and validation errors (RMSE)

regression_greedy_algorithm = pd.DataFrame(columns = ['feature', 'train_error', 'validation_error'])  

i = 0

temp_list = []



# a while loop going through all the features in the dataframe

while i < len(train_data_2.columns)-1:

    

    # a temporary dataframe to select the best feature at each iteration

    temp = pd.DataFrame(columns = ['feature', 'train_error', 'validation_error'])

    

    # a for loop to test all the remaining features

    for p in input_list:

        RMSE_train, RMSE_validation = RMSE(train_data_2, validation_data, temp_list, p)

        temp = temp.append({'feature':p, 'train_error':RMSE_train, 'validation_error':RMSE_validation}, ignore_index=True)

        

    temp = temp.sort_values('train_error') # select the best feature using train error

    best = temp.iloc[0,0]

    temp_list.append(best)

    regression_greedy_algorithm = regression_greedy_algorithm.append({'feature': best, 

                                                  'train_error': temp.iloc[0,1], 'validation_error': temp.iloc[0,2]}, 

                                                 ignore_index=True) # add the feature to the dataframe

    input_list.remove(best) # remove the best feature from the list of available features

    i += 1

regression_greedy_algorithm
greedy_algo_features_list = regression_greedy_algorithm['feature'].tolist()[:24] # select the first 30 features

test_error, _, _ = multiple_regression_model(train_data_2, test_data, greedy_algo_features_list)

print ('test error (RMSE) is: %s' %test_error)
input_feature = train_data.columns.values.tolist() # list of column name

input_feature.remove('price')



for i in [1,10]:

    ridge = linear_model.Ridge(alpha = i, normalize = True) # initialize the model

    ridge.fit(train_data.as_matrix(columns = input_feature), train_data.as_matrix(columns = ['price'])) # fit the train data

    print ('test error (RMSE) is: %s' %mean_squared_error(test_data.as_matrix(columns = ['price']), 

                              ridge.predict(test_data.as_matrix(columns = [input_feature])))**0.5) # predict price and test error
ridgeCV = linear_model.RidgeCV(alphas = np.linspace(1.0e-10,1,num = 100), normalize = True, store_cv_values = True) # initialize the model

ridgeCV.fit(train_data.as_matrix(columns = input_feature), train_data.as_matrix(columns = ['price'])) # fit the train data

print ('best alpha is: %s' %ridgeCV.alpha_) # get the best alpha

print ('test error (RMSE) is: %s' %mean_squared_error(test_data.as_matrix(columns = ['price']), 

                              ridgeCV.predict(test_data.as_matrix(columns = [input_feature])))**0.5) # predict price and test error
for i in [0.01,0.1,1,250,500,1000]:

    lasso = linear_model.Lasso(alpha = i, normalize = True) # initialize the model

    lasso.fit(train_data.as_matrix(columns = input_feature), train_data.as_matrix(columns = ['price'])) # fit the train data

    print (lasso.sparse_coef_.getnnz) # number of non zero weights

    print ('test error (RMSE) is: %s' %mean_squared_error(test_data.as_matrix(columns = ['price']), 

                              lasso.predict(test_data.as_matrix(columns = [input_feature])))**0.5) # predict price and test error
lassoCV = linear_model.LassoCV(normalize = True) # initialize the model (alphas are set automatically)

lassoCV.fit(train_data.as_matrix(columns = input_feature), np.ravel(train_data.as_matrix(columns = ['price']))) # fit the train data

print ('best alpha is: %s' %lassoCV.alpha_) # get the best alpha

print ('number of non zero weigths is: %s' %np.count_nonzero(lassoCV.coef_)) # number of non zero weights

print ('test error (RMSE) is: %s' %mean_squared_error(test_data.as_matrix(columns = ['price']), 

                              lassoCV.predict(test_data.as_matrix(columns = [input_feature])))**0.5) # predict price and test error
# normalize the data

train_X = train_data.as_matrix(columns = input_feature)

scaler = preprocessing.StandardScaler().fit(train_X)

train_X_scaled = scaler.transform(train_X)

test_X = test_data.as_matrix(columns = [input_feature])

test_X_scaled = scaler.transform(test_X)



knn = neighbors.KNeighborsRegressor(n_neighbors=10, weights='distance') # initialize the model

knn.fit(train_X_scaled, train_data.as_matrix(columns = ['price'])) # fit the train data

print ('test error (RMSE) is: %s' %mean_squared_error(test_data.as_matrix(columns = ['price']), 

                              knn.predict(test_X_scaled))**0.5) # predict price and test error