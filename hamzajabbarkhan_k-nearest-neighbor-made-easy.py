from IPython.display import Image

Image("../input/euclidean-distance-formula/euclidean distance.PNG")
import pandas as pd 

pd.options.display.max_columns = 40 
#our dataset does not have any headers. So make sure we make list of them first 

columns = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 

        'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 

        'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-rate', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
car_data = pd.read_csv('../input/sample-dataset-for-k-nearest-neighbors/imports-85.data', names = columns)
car_data.head()
nominal_columns = ['make', 'fuel-type','aspiration','num-of-doors','body-style','drive-wheels','engine-location','engine-type','num-of-cylinders','fuel-system']
car_copy = car_data.copy()
car_copy = car_copy.drop(columns = nominal_columns)
car_copy.columns
car_copy = car_copy.drop(columns = ['symboling'])
car_copy.columns
car_copy.isnull().sum()
car_copy['normalized-losses'][0]
type(car_copy['normalized-losses'][0])
import numpy as np 
car_copy = car_copy.replace('?',np.nan)
car_copy.isnull().sum()
car_copy.info()
car_copy = car_copy.astype('float')
car_copy.info()
car_copy.isnull().sum()
#Tip : if you ever find yourself stuck in trying to figure out the parameters of the function or method, just press shift + tab tab in jupyter notebook

car_copy = car_copy.dropna(subset = ['price'])
car_copy.isnull().sum()
car_copy.shape
car_copy = car_copy.drop(columns = ['normalized-losses'])
car_copy.shape
car_copy.isnull().sum()
car_copy = car_copy.fillna(car_copy.mean())
car_copy.isnull().sum()
car_copy.head()
price_column = car_copy['price']
car_copy = (car_copy - car_copy.min())/(car_copy.max() - car_copy.min())
car_copy.head()
car_copy['price'] = price_column 
car_copy.head()
from sklearn.neighbors import KNeighborsRegressor 

from sklearn.metrics import mean_squared_error 
#instantiate the KNN class.

#I would suggest visiting the documentation of the KNN class and understanding what the parameters are.

#the parameters we will be focusing on are n_neighbors = 5 by default, algorithm = 'auto' by default and p = 2 by default. Leave them as it is for now.

knn = KNeighborsRegressor()

#before we split, make sure you randomize the dataset.

np.random.seed(1)

#make sure you set the random_seed to 1. Otherwise everytime you restart your kernel let's say, you would get different results.

index_shuffled = np.random.permutation(car_copy.index)

shuffled_car_data = car_copy.loc[index_shuffled]

#We will use 75% of the data for training and the remaining 25% for testing. 

train_index = int(shuffled_car_data.shape[0] * 0.75)

training_data = shuffled_car_data.iloc[0:train_index]

testing_data = shuffled_car_data.iloc[train_index:]

knn.fit(training_data[['bore']], training_data['price'])

prediction = knn.predict(testing_data[['bore']])

mse = mean_squared_error(testing_data['price'], prediction)

mse**(1/2)

columns_testing = shuffled_car_data.columns.tolist()
columns_testing
columns_testing.remove('price')

col_list = []

for x in range(len(columns_testing)): 

    col_list.append(columns_testing[x])

    knn2 = KNeighborsRegressor()

    knn2.fit(training_data[col_list], training_data['price'])

    predictions = knn2.predict(testing_data[col_list])

    mse = mean_squared_error(testing_data['price'], predictions)

    rmse = mse**(1/2)

    print(("For {0} features, mse is {1}, rmse is {2}").format(x+1,mse,rmse))

    print('')
k_list = [1,2,3,4,5,6,7,8,9,10]

for x in k_list: 

    knn3 = KNeighborsRegressor(n_neighbors=x)

    knn3.fit(training_data[col_list], training_data['price'])

    k_predictions = knn3.predict(testing_data[col_list])

    mse = mean_squared_error(testing_data['price'], k_predictions)

    rmse = mse**(1/2)

    print(("For k = {0}, MSE is {1}, RMSE is {2}").format(x, mse, rmse)) 

    print('')
def knn_lever_tuning(feature_list, k_list, df):

    

    np.random.seed(1)

    shuffled_index = np.random.permutation(df.index)

    new_df = df.loc[shuffled_index]

    train_index = int(new_df.shape[0]*0.75)

    train_data = new_df.iloc[0:train_index]

    test_data = new_df.iloc[train_index:]

    

    

    k_result = {}

    

    for y in k_list: 

        model = KNeighborsRegressor(n_neighbors = y)

        model.fit(train_data[feature_list], train_data['price'])

        price_prediction = model.predict(test_data[feature_list])

        mse = mean_squared_error(test_data['price'], price_prediction)

        rmse = mse**(1/2)

        k_result[y] = rmse

    return k_result

    
col = []

final_result = {}

for x in columns_testing:

    col.append(x)

    error_values = knn_lever_tuning(col,[1,2,3,4,5,6,7,8,9,10], car_copy)

    number_features = '{0} features'.format(len(col))

    final_result[number_features] = error_values

features_k_df = pd.DataFrame.from_dict(final_result)

features_k_df



#in the dataframe below, the index is the k values
import seaborn as sns 

import matplotlib.pyplot as plt 



plt.figure(figsize = (12,12))

sns.heatmap(features_k_df)

from sklearn.model_selection import KFold 

from sklearn.model_selection import cross_val_score 
kf = KFold(n_splits = 5, shuffle = True, random_state = 1)

model_final = KNeighborsRegressor()

cvs = cross_val_score(estimator = model_final, X = car_copy[columns_testing], y = car_copy['price'], scoring = 'neg_mean_squared_error', cv = kf)
cvs
cvs = np.absolute(cvs)
rmse = np.sqrt(cvs)

rmse
k_folds = [3,5,7,9,10,11,13]



for fold in k_folds: 

    kf = KFold(n_splits = fold, shuffle = True, random_state = 1)

    knn_model = KNeighborsRegressor()

    cross_score = cross_val_score(estimator = knn_model,X = car_copy[columns_testing], y = car_copy['price'], scoring = 'neg_mean_squared_error', cv = kf)

    rmses = np.sqrt(np.absolute(cross_score))

    avg_rmse = np.mean(rmses)

    std_rmse = np.std(rmses)

    print('for {0} folds, average rmse : {1}, standard deviation of rmse : {2}'.format(str(fold),str(avg_rmse),str(std_rmse)))

    