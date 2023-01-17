import numpy as np



# Set the random seed for reproducability

np.random.seed(42)
import pandas as pd
# Reads in the csv-files and creates a dataframe using pandas



# base_set = pd.read_csv('data/housing_data.csv')

# benchmark = pd.read_csv('data/housing_test_data.csv')

# sampleSubmission = pd.read_csv('data/sample_submission.csv')
base_set = pd.read_csv('../input/dat158-2019/housing_data.csv')

benchmark = pd.read_csv('../input/dat158-2019/housing_test_data.csv')

sample_submission = pd.read_csv('../input/dat158-2019/sample_submission.csv')
%matplotlib inline



import matplotlib.pyplot as plt

import seaborn as sns
base_set.head()
benchmark.head()
base_set.info()
benchmark.info()
base_set.describe()
correlations = base_set.corr()

correlations["median_house_value"]
base_set.hist(bins=50, figsize=(15,15))

plt.show()
base_set.plot(kind="scatter", 

           x="longitude", 

           y="latitude", 

           alpha=0.4,

           s=base_set["population"]/100, 

           label="population",

           c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,

           figsize=(15,7))

plt.legend()
from pandas.plotting import scatter_matrix

attributes = ['median_house_value', 'median_income',

             'total_rooms', 'housing_median_age']

scatter_matrix(base_set[attributes], figsize=(12,8))
# There are null values in total_bedrooms, we fill those with the median

def fill_null(dataset, column):

    values = {column: dataset[column].median()}

    

    return dataset.fillna(values)



# For these particular sets, there are only null values in the 'total_bedrooms' column.

base_set = fill_null(base_set, 'total_bedrooms')

benchmark = fill_null(benchmark, 'total_bedrooms')
benchmark = benchmark.drop(columns=['Id'])
base_set.isnull().any()
benchmark.isnull().any()
labels_column = 'median_house_value'



X = base_set.drop(columns=[labels_column])

Y = pd.DataFrame(base_set[labels_column], columns=[labels_column])
X.head()
Y.head()
def derive_datapoints(dataset):

    dataset['bedrooms_per_room'] = dataset['total_bedrooms'] / dataset['total_rooms']



    dataset['rooms_per_household'] = dataset['total_rooms'] / dataset['households']

    dataset['bedrooms_per_household'] = dataset['total_bedrooms'] / dataset['households']

    dataset['population_per_household'] = dataset['population'] / dataset['households']

    

    return dataset



X = derive_datapoints(X)

benchmark = derive_datapoints(benchmark)



# One-hot encoding

X = pd.get_dummies(X)

benchmark = pd.get_dummies(benchmark)
# Some housekeeping, we need to ensure the test set has the same columns as the training set

# The missing columns will be the onehot-encoded values

missing_columns = set( X.columns ) - set( benchmark.columns )



# We fill the values in the missing columns with 0, as they are one-hot encoded values that don't exist in the set

for column in missing_columns:

    benchmark[column] = 0



# Ensure the order of column in the test set is in the same order than in train set

benchmark = benchmark[X.columns]
X.head()
benchmark.head()
from sklearn.model_selection import train_test_split



train_to_valtest_ratio = .2

validate_to_test_ratio = .5



# First split our main set

(X_train,

 X_validation_and_test,

 Y_train,

 Y_validation_and_test) = train_test_split(X, Y, test_size=train_to_valtest_ratio)



# Then split our second set into validation and test

(X_validation,

 X_test,

 Y_validation,

 Y_test) = train_test_split(X_validation_and_test, Y_validation_and_test, test_size=validate_to_test_ratio)
from sklearn.preprocessing import MinMaxScaler, StandardScaler



X_scaler = StandardScaler().fit(X_train)

def scale_dataset_X(dataset):

    return X_scaler.transform(dataset)



X_train_scaled = scale_dataset_X(X_train)

X_validation_scaled = scale_dataset_X(X_validation)

X_test_scaled = scale_dataset_X(X_test)
from keras.models import Sequential

from keras.layers import Dense



model = Sequential([

    Dense(15, activation='relu', input_dim=X_train.shape[1]),

    Dense(15, activation='relu'),

    Dense(60, activation='relu'),

    Dense(120, activation='relu'),

    Dense(60, activation='relu'),

    Dense(15, activation='relu'),

    Dense(1),

])



model.summary()
import keras.backend as K



def rmse(y_true, y_pred):

    return K.sqrt(K.mean(K.square(y_pred - y_true)))



model.compile(optimizer='adadelta', # adam, sgd, adadelta

              loss=rmse,

              metrics=[rmse, 'mae'])
from keras.callbacks import EarlyStopping



early_stopper = EarlyStopping(patience=3)



training_result = model.fit(X_train_scaled, Y_train,

                            batch_size=32,

                            epochs=250,

                            validation_data=(X_validation_scaled, Y_validation),

                            callbacks=[early_stopper])
# Plot model accuracy over epoch

plt.plot(training_result.history['mean_absolute_error'])

plt.plot(training_result.history['val_mean_absolute_error'])

plt.title('Model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()



# Plot model loss over epoch

plt.plot(training_result.history['loss'])

plt.plot(training_result.history['val_loss'])

plt.title('Model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
validate_result = model.test_on_batch(X_validation_scaled, Y_validation)

validate_result
test_result = model.test_on_batch(X_test_scaled, Y_test)

test_result
from sklearn.ensemble import RandomForestRegressor



rfr_model = RandomForestRegressor()

rfr_model.fit(X_train, Y_train)



# Get the mean absolute error on the validation data

rfr_predictions = rfr_model.predict(X_test)



rfr_error =  np.sqrt(np.mean((rfr_predictions - Y_test['median_house_value']) ** 2))

rfr_error
import re



regex = re.compile(r"[|]|<", re.IGNORECASE)



# XGBoost does not support some of the column names



X_train.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_train.columns.values]

X_test.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_test.columns.values]



from xgboost.sklearn import XGBRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RandomizedSearchCV



import scipy.stats as st



one_to_left = st.beta(10, 1)  

from_zero_positive = st.expon(0, 50)



xgb_reg = XGBRegressor(nthreads=-1)



xgb_gs_params = {  

    "n_estimators": st.randint(3, 40),

    "max_depth": st.randint(3, 40),

    "learning_rate": st.uniform(0.05, 0.4),

    "colsample_bytree": one_to_left,

    "subsample": one_to_left,

    "gamma": st.uniform(0, 10),

    'reg_alpha': from_zero_positive,

    "min_child_weight": from_zero_positive,

}



xgb_gs = RandomizedSearchCV(xgb_reg, xgb_gs_params, n_jobs=1)  

xgb_gs.fit(X_train.as_matrix(), Y_train)  



xgb_model = xgb_gs.best_estimator_ 



xgb_predictions = xgb_model.predict(X_test.as_matrix())



xgb_error =  np.sqrt(np.mean((xgb_predictions - Y_test['median_house_value']) ** 2))

xgb_error
print(f'NN RMSE:                            {test_result[0]}')

print(f'RandomForestRegressor RMSE:         {rfr_error}')

print(f'XGBRegressor RMSE:                  {xgb_error}')
# Scale test data

benchmark_scaled = scale_dataset_X(benchmark)
benchmark.head()
X.head()
median_house_value = model.predict(benchmark_scaled)
len(median_house_value)
median_house_value
submission = pd.DataFrame({

    'Id': [i for i in range(len(median_house_value))],

    'median_house_value': median_house_value.flatten()

})
submission.head()
# Stores a csv file to submit to the kaggle competition

submission.to_csv('submission.csv', index=False)