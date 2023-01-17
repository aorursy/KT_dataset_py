# import your libraries

import warnings

warnings.filterwarnings('ignore')

warnings.simplefilter('ignore')

from xgboost import XGBRegressor, plot_importance

import numpy as np 

import pandas as pd 

from matplotlib import pyplot as plt

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.feature_selection import SelectFromModel 

import os
# check what we have in input

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# load the datasets from the csv files using pandas

train_data = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')

test_data = pd.read_csv('../input/home-data-for-ml-course/test.csv', index_col='Id')
# drop the 'Id' index column to make it easy to index by default integer increment

# train_data.reset_index(drop=True,inplace=True)

train_data.head(5)
# drop rows where SalesPrice is null

train_data.dropna(axis='index',subset=['SalePrice'],inplace=True)



# drop rows where GarageYrBlt is null

train_data.dropna(axis='index',subset=['GarageYrBlt'],inplace=True)



# get the target feature

target = train_data.SalePrice 



# remove the sales price feature from the input data and remove categorical values

train_data.drop(columns=['SalePrice'],inplace=True) 

train = train_data.select_dtypes(exclude=['object']) # exclude categorical features

test = test_data.select_dtypes(exclude=['object'])



# create training and validation chunks

train_x, valid_x, train_y, valid_y = train_test_split(train,target,train_size=0.8,test_size=0.2, random_state=200)
# view the shape of the remaining features

print(f'>>> Training data shape: {train_x.shape}')
# print the columns with missing values

columns_na = train_x.columns[train_x.isna().any()].tolist()

print(f'>>> Columns with missing values:')

column_na_count = (train_x.isna().sum()) # returns a series with feature: na_count

print(f'{column_na_count[column_na_count > 0]}') 

#print(f'{column_na_count}')
def score_droprow_vs_imputation(X,y,x_valid, y_valid):

    model = XGBRegressor(n_estimators=330,random_state=100) # use other default parameters

    model.fit(X,y)

    # predict with the validation data

    predictions = model.predict(x_valid)

    return mean_absolute_error(y_valid,predictions)
# use imputation and return the MEA

imputer = SimpleImputer(strategy='most_frequent') # you can try other strategies too

imputed_train_x = imputer.fit_transform(train_x)

imputed_valid_x = imputer.transform(valid_x)

# extract column names

column_names = list(train_x) 

#fill in the column names and form pandas DF again

imputed_train_x = pd.DataFrame(data=imputed_train_x,columns=column_names)

imputed_valid_x = pd.DataFrame(data=imputed_valid_x,columns=column_names)

# check the score

print(f'>>> MAE (imputation): {score_droprow_vs_imputation(imputed_train_x,train_y,imputed_valid_x,valid_y)}')
# lets check the score for dropping rows with missing values

print(f'>>> Shape of training data: {train_x.shape}')

na_row_id_train = np.nonzero(pd.isna(train_x).any(1))[0]

na_row_id_valid = np.nonzero(pd.isna(valid_x).any(1))[0]

print(f'>>> Number of rows to be dropped on training data: {len(na_row_id_train)}')

print(f'>>> Number of rows to be dropped on validation data: {len(na_row_id_valid)}')



# use boolean access to get the required not null rows

train_index = np.ones(len(train_x.index),dtype=bool)

valid_index = np.ones(len(valid_x.index),dtype=bool)

# set the rows to be dropped to False

train_index[na_row_id_train] = False

valid_index[na_row_id_valid] = False

# get the non null rows

reduced_train_x, reduced_train_y = train_x.loc[train_index], train_y.loc[train_index]

reduced_valid_x, reduced_valid_y = valid_x.loc[valid_index], valid_y.loc[valid_index]

# print the new shapes

print(f'>>> New data shape for training: {reduced_train_x.shape},\t {reduced_train_y.shape}')

print(f'>>> New data shape for validation:{reduced_valid_x.shape},\t {reduced_valid_y.shape}')



# print MAE dropping rows

print(f'>>> MAE (row dropping): {score_droprow_vs_imputation(reduced_train_x,reduced_train_y,reduced_valid_x,reduced_valid_y)}')
# create the model and plot the feature imortance for the reduced dataset

model = XGBRegressor(n_estimators=330, random_state=150)

model.fit(reduced_train_x,reduced_train_y)

fig, (ax1, ax2) = plt.subplots(figsize=(30, 15),nrows=1,ncols=2,sharey=False)

plot_importance(model,ax=ax1,color='m')

ax1.set_title('Dropping Rows Feature Importance')

# fit on the imputed data

model.fit(imputed_train_x,train_y)

plot_importance(model,ax=ax2,color='g')

ax2.set_title('Imputation Feature Importance')

plt.show()
# create and fit the model on the reduced dataset

model = XGBRegressor(n_estimators=330, random_state=120)

model.fit(reduced_train_x,reduced_train_y)

# generate predictions fromthe validation data

predictions = model.predict(reduced_valid_x)

print(f'>>> MAE (base model): {mean_absolute_error(predictions,reduced_valid_y)}')

# get the threshold iterable by sorting the feature importance

bounds = np.sort(model.feature_importances_)

num_of_features_reduced = []

mae_reduced = []

final_threshold = 0.014168278314173222

for threshhold in bounds:

    selected = SelectFromModel(model,threshold=threshhold,prefit=True)

    selected_train_x = selected.transform(reduced_train_x) # get a reduced version of the train_x

    num_of_features = selected_train_x.shape[1]

    # now train a new model with the selected_train_x

    selected_model = XGBRegressor(n_estimators=330,random_state=120)

    selected_model.fit(selected_train_x,reduced_train_y)

    # validate our reduced model

    selected_valid_x = selected.transform(reduced_valid_x)

    predictions = selected_model.predict(selected_valid_x)

    mae = mean_absolute_error(reduced_valid_y,predictions)

    mae_reduced.append(mae)

    num_of_features_reduced.append(num_of_features)

    print(f'>>> Threshold = {threshhold}\t number of features = {num_of_features}\t MAE = {mae}')
# create and fit the model on the imputed dataset

model = XGBRegressor(n_estimators=330, random_state=120)

model.fit(imputed_train_x,train_y)

# generate predictions from the validation data

predictions = model.predict(imputed_valid_x)

print(f'>>> MAE (base model): {mean_absolute_error(predictions,valid_y)}')

# get the threshold iterable by sorting the feature importance

bounds = np.sort(model.feature_importances_)

num_of_features_imp = []

mae_imp = []

for threshhold in bounds:

    selected = SelectFromModel(model,threshold=threshhold,prefit=True)

    selected_train_x = selected.transform(imputed_train_x) # get a reduced version of the train_x

    num_of_features = selected_train_x.shape[1]

    # now train a new model with the selected_train_x

    selected_model = XGBRegressor(n_estimators=330,random_state=120)

    selected_model.fit(selected_train_x,train_y)

    # validate our reduced model

    selected_valid_x = selected.transform(imputed_valid_x)

    predictions = selected_model.predict(selected_valid_x)

    mae = mean_absolute_error(valid_y,predictions)

    num_of_features_imp.append(num_of_features)

    mae_imp.append(mae)

    print(f'>>> Threshold = {threshhold}\t number of features = {num_of_features}\t MAE = {mae}')
# let us plot the comparison

fig, ax = plt.subplots(figsize=(30, 15))

ax.plot(num_of_features_reduced,mae_reduced,color='green', marker='o',linewidth=4, markersize=12,label='Reduced Data')

ax.plot(num_of_features_imp,mae_imp,color='red', marker='*',linewidth=4, markersize=12,label='Imputed Data')

ax.grid(True)

ax.set_title('MAE vs Number of Features')

ax.set_xlabel('Number of Features')

ax.set_ylabel('MAE')

ax.legend()
# create and fit the model on the reduced dataset for test prediction

model = XGBRegressor(n_estimators=330, random_state=120)

model.fit(reduced_train_x,reduced_train_y)

# sort the feature importance

bounds = np.sort(model.feature_importances_)

final_threshold = 0.014168278314173222 # we got this threshold from our plot above

# select the m

selected = SelectFromModel(model,threshold=final_threshold,prefit=True)

# get a reduced version of the train_x

selected_train_x = selected.transform(reduced_train_x) 

# confirm that num_of_features == 14

assert selected_train_x.shape[1] == 14 

# now train a new model with the selected_train_x

selected_model = XGBRegressor(n_estimators=330,random_state=120)

selected_model.fit(selected_train_x,reduced_train_y)
# check if there are still NaN values in the test data

column_na_count = (test.isna().sum()) # returns a series with feature: na_count

print(f'{column_na_count[column_na_count > 0]}') 

print(f'>>> Test shape: {test.shape}')
# get the imputer from training data

test_imputer = SimpleImputer(strategy='most_frequent') # you can try other strategies too

# get the most frequent from reduced train

imputed_train_x = pd.DataFrame(data=test_imputer.fit_transform(reduced_train_x),

                               columns=list(reduced_train_x)) 

imputed_test_x = pd.DataFrame(data=test_imputer.transform(test),columns=list(test))



# transform the test data into a reduced version using the SelectFromModel

selected_test = selected.transform(imputed_test_x)

test_predictions = selected_model.predict(selected_test)



# prepare data for submission

results = pd.DataFrame({'Id':test.index,

                       'SalePrice':test_predictions})

results.to_csv('submission.csv',index=False)