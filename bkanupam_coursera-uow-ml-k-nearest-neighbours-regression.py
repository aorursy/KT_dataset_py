# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import os
from IPython.display import display

# dictionary with dataset column names and their corresponding data types
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 
              'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 
              'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 
              'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
# change directory to the location where the data for the notebook is located
os.chdir('/kaggle/input/knearestneighboursregression-dataset')
train = pd.read_csv('kc_house_data_small_train.csv', dtype = dtype_dict)
print('Top 5 rows of training dataset')
display(train.head())
test = pd.read_csv('kc_house_data_small_test.csv', dtype = dtype_dict)
validate = pd.read_csv('kc_house_data_validation.csv', dtype = dtype_dict)

# add a column vector of all ones to the begining of a feature matrix
def add_one_vector(X):
    one_vector = np.ones(len(X)).reshape(len(X), 1)
    return np.concatenate((one_vector, X), axis=1)


def extract_features(df, datatype_dict):
    """
    Extract input and output features from a dataframe and return a tuple of input features matrix and output 
    feature vector
    :param df: dataframe
    :param datatype_dict: dictionary with dataset column names and their corresponding data types
    :return: a tuple of input features matrix and output feature vector
    """    
    # remove the columns that are not numeric i.e. int, floats etc.
    # train.dtypes gives a pandas with index as column names and value as column data types. We filter this
    # series to remove columns of type object
    numeric_cols = pd.Series(train.dtypes).where(lambda col_dtype: col_dtype != 'object').dropna()
    feature_names = list(numeric_cols.keys().values)
    # price is the output variable
    feature_names.remove('price')
    # extract the input features from the dataframe as a numpy 2d array
    input_features = add_one_vector(df[feature_names].values)
    output_variable = df.loc[:, 'price'].values
    return input_features, output_variable
train_input_features, train_output = extract_features(train, dtype_dict)
cv_input_features, cv_output = extract_features(validate, dtype_dict)
test_input_features, test_output = extract_features(test, dtype_dict)
def normalize_features(input_features):
    norm = np.sqrt(np.sum(input_features**2, axis=0))
    normalized_features = input_features / norm
    return normalized_features, norm

norm_train_input_features, train_norm = normalize_features(train_input_features)
norm_cv_input_features = cv_input_features / train_norm
norm_test_input_features = test_input_features / train_norm
print(norm_test_input_features[0])
print(norm_train_input_features[9])
np.sqrt(np.sum((norm_train_input_features[9] - norm_test_input_features[0])**2))
distance = {}
for i in range(10):
    distance[i] = np.sqrt(np.sum((norm_train_input_features[i] - norm_test_input_features[0])**2))
print(distance)
distance_sorted = sorted(distance.items(), key=lambda item: item[1])
print(distance_sorted)
diff = norm_train_input_features[:] - norm_test_input_features[0]
diff[-1].sum()
def compute_distance(training_examples, query_house):
    """
    Vectorized implementation of calculating the distance of a query house from each of the training examples
    :param training_examples: a matrix or numpy 2d array consisting of training data (input features)
    :param query_house: the query house
    :return: numpy 2d array whose first column is the training row index and second column is the distance from
    the query house
    """
    # subtract the query house row from each training example row
    diff_matrix = training_examples - query_house
    # now for each row in the matrix (which corresponds to each training example), calculate the sum of
    # squares of feature values ( this is done by using axis = 1 in the 2d array )
    distance = np.sqrt(np.sum(diff_matrix**2, axis=1))
    index = np.arange(0, len(distance))    
    return np.concatenate((index.reshape(-1, 1), distance.reshape(-1, 1)), axis=1)
rowindex_distance = compute_distance(norm_train_input_features[:], norm_test_input_features[2])

def get_min_distance_row_index(rowindex_distance):
    rowindex = rowindex_distance[:, 0]
    distance = rowindex_distance[:, 1]
    min_distance = np.amin(distance)
    min_distance_index = np.where(distance == min_distance)
    return rowindex[min_distance_index], min_distance

min_row_index, min_distance = get_min_distance_row_index(rowindex_distance)
print(int(min_row_index[0]), min_distance)
# Since the query house ( i.e. the third test example) is closest to the 382nd training example, hence the predicted 
# value of the query house will be same as the price of the 382nd training example
train_output[382]
def k_nearest_neighbours(k, training_examples, query_house):
    rownumber_distance = compute_distance(training_examples, query_house)
    # sort the 2d array on the index column in ascending order
    # You can call .argsort() on the column you want to sort, and it will give you an array of row indices 
    # that sort that particular column which you can pass as an index to your original array.
    rownumber_distance_sorted = rownumber_distance[rownumber_distance[:, 1].argsort()]
    return rownumber_distance_sorted[0:k, :]
knn_rowindex_distance = k_nearest_neighbours(4, norm_train_input_features[:], norm_test_input_features[2])
print(knn_rowindex_distance[:, 0].astype(int))
def predict_house_price_custom(k, train_input_features, train_output, query_house):
    k_rowindex_distance = k_nearest_neighbours(k, train_input_features, query_house)
    # get the rowindex of the k nearest neighbours and get their corresponding prices
    k_rowindex = k_rowindex_distance[:, 0].astype(int)
    # get the mean of the k nearest house prices, this is the predicted price of the query house
    return np.mean(train_output[k_rowindex])
query_house_predicted_price = []

for query_house_index in range(10):
    query_house = norm_test_input_features[query_house_index]
    predicted_price = predict_house_price_custom(10, norm_train_input_features, train_output, query_house)
    query_house_predicted_price.append((query_house_index, predicted_price))

# sort on the basis of predicted prices in ascending order    
sorted_query_house_predicted_price = sorted(query_house_predicted_price, key=lambda item:item[1])
print(sorted_query_house_predicted_price)
print('\nHouse index in test set of 10 houses with lowest predicted price: {}'
      .format(sorted_query_house_predicted_price[0][0]))
def get_optimized_kvalue(list_kvalues, predict_house_price):
    k_rss = []
    for k in list_kvalues:    
        queryhouseindex_predictedprice_actualprice = []
        for query_house_index in range(len(norm_cv_input_features)):
            query_house = norm_cv_input_features[query_house_index]
            actual_price = cv_output[query_house_index]
            predicted_price = predict_house_price(k, norm_train_input_features, train_output, query_house)
            queryhouseindex_predictedprice_actualprice.append([query_house_index, predicted_price, actual_price])
        queryhouseindex_predictedprice_actualprice = np.array(queryhouseindex_predictedprice_actualprice)            
        # now calculate the residual sum of squares ( (predicted value - actual value)**2 ) over the entire validation set
        price_diff = (queryhouseindex_predictedprice_actualprice[:, 1] - queryhouseindex_predictedprice_actualprice[:, 2])**2
        rss = np.sum(price_diff)
        k_rss.append((k, rss))   
        print('k: {} --> rss: {}'.format(k, rss))
    sorted_k_rss = sorted(k_rss, key=lambda item:item[1])    
    return sorted_k_rss[0][0]

optimized_k = get_optimized_kvalue(np.arange(15)+1, predict_house_price_custom)
print('\n The value of k that minimizes RSS on validation data is: {}'.format(optimized_k))

from sklearn.neighbors import KNeighborsRegressor

def predict_house_price_scikit(k, train_input_features, train_output, query_house):
    k10_regressor = KNeighborsRegressor(n_neighbors = k, weights='distance')
    k10_regressor.fit(train_input_features, train_output)    
    return k10_regressor.predict(query_house.reshape(1, -1))

predict_house_price_scikit(4, norm_train_input_features, train_output, norm_test_input_features[2])
optimized_k_scikit = get_optimized_kvalue(np.arange(15)+1, predict_house_price_scikit)
print('\n The value of k (using scikit nearest neighbors) that minimizes RSS on validation data is: {}'
      .format(optimized_k_scikit))

