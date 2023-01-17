# My forecasting COVID-19 confirmed cases and fatalities between March 25 and April 22 

# My submission scored 1.81929



import numpy as np 

import pandas as pd 



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_absolute_error



# model

from catboost import Pool

from catboost import CatBoostRegressor



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# load training and testing data 

subm = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')

train_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv', index_col='Id', parse_dates=True)

test_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv', index_col='ForecastId', parse_dates=True)
# see testing data

test_data
# ...and training data

train_data
train_data.describe()
test_data.describe()
train_data.describe(include=['O'])
test_data.describe(include=['O'])
train_data.shape
test_data.shape
# detect missing values in training

train_data.isna().sum()
# ...in testing data

test_data.isna().sum()
# Count number of unique elements in the object

train_data.nunique()
# separate the vector correct answers ('ConfirmedCases' and 'Fatalities') from the training data

train_data.dropna(axis=0, subset=['ConfirmedCases', 'Fatalities'], inplace=True)

y_conf = train_data.ConfirmedCases

train_data.drop(['ConfirmedCases'], axis=1, inplace=True)

y_fatal = train_data.Fatalities

train_data.drop(['Fatalities'], axis=1, inplace=True)

# Select categorical columns in training and testing data

categorical_cols = [cname for cname in train_data.columns if

                    train_data[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in train_data.columns if 

                train_data[cname].dtype in ['int64', 'float64']]
# replace missing values in training and testing data

# as we saw above, the data are absent only in 'Province/State'

train_data.fillna('-', inplace=True)

test_data.fillna('-',inplace=True)
train_data.shape
# perform LabelEncoder with categorical data (categorical_cols)

encodering = LabelEncoder()



encod_train_data = train_data.copy()

encod_test_data = test_data.copy()

for col in categorical_cols:

    encod_train_data[col] = encodering.fit_transform(train_data[col])

    encod_test_data[col] = encodering.fit_transform(test_data[col])
# split encod_train_data into training(X_train) and validation(X_valid) data

# and split vector correct answers ('ConfirmedCases')

X_train, X_valid, y_train, y_valid = train_test_split(encod_train_data, y_conf, train_size=0.8, 

                                                      test_size=0.2, random_state=0)
# select model and install parameters

model = CatBoostRegressor(iterations=80, 

                          depth=16, 

                          learning_rate=0.04, 

                          loss_function='RMSE')
# determine the best metrics for the model

def get_score(iterations):

    model = CatBoostRegressor(iterations=iterations, 

                          depth=8, 

                          learning_rate=0.04,

                          loss_function='RMSE')

    scores = cross_val_score(model, X_train, y_train, cv=3)



    return scores.mean()
#results = {}

#for x in [60, 70, 80, 100]:

    #results[x] = get_score(x)
#results
# train the model

model.fit(X_train,y_train)
# preprocessing of validation data, get predictions

preds = model.predict(X_valid)



print('MAE:', mean_absolute_error(y_valid, preds))
x_list = [X_train, X_valid]

y_list = [y_train, y_valid]



scoring = list(map(lambda x,y: round(model.score(x,y)*100, 2), x_list, y_list)) 

scoring
# get predictions test data

final_preds_conf = model.predict(encod_test_data)
# split encod_train_data into training(X_train) and validation(X_valid) data

# and split vector correct answers ('Fatalities')

X_train_f, X_valid_f, y_train_f, y_valid_f = train_test_split(encod_train_data, y_fatal, train_size=0.8, 

                                                      test_size=0.2, random_state=0)
# train the model

model.fit(X_train_f,y_train_f)
# preprocessing of validation data, get predictions

preds = model.predict(X_valid_f)



print('MAE:', mean_absolute_error(y_valid_f, preds))
x_list_f = [X_train_f, X_valid_f]

y_list_f = [y_train_f, y_valid_f]



scoring = list(map(lambda x,y: round(model.score(x,y)*100, 2), x_list_f, y_list_f)) 

scoring
# get predictions test data

final_preds_fatal = model.predict(encod_test_data)
# combine predictions 'ConfirmedCases' and 'Fatalities'

output = pd.DataFrame({'ForecastId': test_data.index,

                       'ConfirmedCases': final_preds_conf,

                       'Fatalities': final_preds_fatal})
# replace negative values with 0, because the predictions of 'ConfirmedCases' and 'Fatalities' cannot be negative

output.loc[output['ConfirmedCases'] < 0,'ConfirmedCases'] = 0

output.loc[output['Fatalities'] < 0,'Fatalities'] = 0
# and save test predictions to file

output.to_csv('submission.csv', index=False)