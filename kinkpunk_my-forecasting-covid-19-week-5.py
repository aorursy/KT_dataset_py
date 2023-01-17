import numpy as np 

import pandas as pd 



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler 



import time

from tqdm import tqdm



# model

from catboost import Pool

from catboost import CatBoostRegressor

from catboost import CatBoostClassifier

#from xgboost import XGBRegressor

#from sklearn.ensemble import GradientBoostingRegressor

#from sklearn.ensemble import BaggingRegressor



#plot

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# load training and testing data 

subm = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/submission.csv')

training_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/train.csv', index_col='Id', parse_dates=True)

testing_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv', index_col='ForecastId', parse_dates=True)

happiest_data = pd.read_csv('/kaggle/input/world-happiness/2019.csv')
# load additional data 

happiest_data.rename(columns={'Country or region':'Country_Region'}, inplace=True)
testing_data.info(), training_data.info()
training_data.describe(include=['O'])
training_data.loc[training_data['Country_Region'] == 'US']
# add information to the training data from happiest_data

train_data = training_data.copy()

train_data = train_data.merge(happiest_data, how='left', left_index=True, on=['Country_Region'])

train_data.index = training_data.index
# ... and to the test data

test_data = testing_data.copy()

test_data = test_data.merge(happiest_data, how='left', on=['Country_Region'])

test_data.index = testing_data.index
# see testing data

test_data
# detect missing values in training

train_data.isna().sum()
# Convert data in integer

train_data['Date']= pd.to_datetime(train_data['Date']).dt.strftime("%m%d").astype(int)

test_data['Date']= pd.to_datetime(test_data['Date']).dt.strftime("%m%d").astype(int)
# define the minimum and maximum dates after convertion in train data

train_data['Date'].min(), train_data['Date'].max()
# define the minimum and maximum dates after convertion in train data

test_data['Date'].min(), test_data['Date'].max()
# create a list with dates that intersect in the training and test data

drop_date = [i for i in range(test_data['Date'].min(), train_data['Date'].max()+1)]
# see it

#drop_date
# throw out the dates coinciding with the test data from the train data

train_data = train_data.loc[~train_data['Date'].isin(drop_date)]
#check the minimum and maximum dates 

train_data['Date'].min(), train_data['Date'].max()
# separate the vector correct answers from the training data

y = train_data.TargetValue

train_data.drop(['TargetValue'], axis=1, inplace=True)
# Select categorical columns in training and testing data

categorical_cols = [cname for cname in train_data.columns if

                    train_data[cname].dtype == "object"]
# Select non type columns in training and testing data

non_cols = [cname for cname in train_data.columns if

                    train_data[cname].dtype == None]
non_cols, categorical_cols
# replace missing values in training and testing data

train_data[categorical_cols] = train_data[categorical_cols].fillna('-')

test_data[categorical_cols] = test_data[categorical_cols].fillna('-')
train_data.isna().sum()
# replace missing non type values in training and testing data

train_data[non_cols] = train_data[non_cols].fillna(0)

test_data[non_cols] = test_data[non_cols].fillna(0)
train_data.isna().sum()
# perform LabelEncoder with categorical data (categorical_cols)

state_encoder = LabelEncoder()

counrty_encoder = LabelEncoder()

ord_encoder = OrdinalEncoder()

encod_train_data = train_data.copy()

encod_test_data = test_data.copy()



    

encod_train_data[categorical_cols] = ord_encoder.fit_transform(train_data[categorical_cols])

encod_test_data[categorical_cols] = ord_encoder.transform(test_data[categorical_cols])

encod_train_data.loc[120], encod_test_data.loc[120]
def rmse_score(learning_rate):

    rmse = np.sqrt(-cross_val_score(CatBoostRegressor(iterations=2000, 

                          depth=9, 

                          learning_rate=learning_rate, 

                          loss_function='RMSE',

                          #random_seed=random_seed,

                          verbose=False),X_train, y_train, scoring="neg_mean_squared_error", cv = 3))

    return(rmse)
# metrics = [0.01, 0.04, 0.4]

# results = {}

# for x in tqdm(metrics):

#    results[x] = rmse_score(x)

#    time.sleep(1)
# results
# plt.figure(figsize=(12,8))

# for i in results:

#    sns.lineplot(data=results[i], label=i)
# for x in metrics:

#    print(x, results[x].mean())
# split encod_train_data into training(X_train) and validation(X_valid) data

# and split vector correct answers ('ConfirmedCases')

X_train, X_valid, y_train, y_valid = train_test_split(encod_train_data, y, train_size=0.95, 

                                                      test_size=0.05, random_state=0)
# select model and install parameters

model = CatBoostRegressor(iterations=12000, 

                          depth=9, 

                          learning_rate=0.04, 

                          loss_function='RMSE',

                          verbose=False)
# train the model

model.fit(X_train,y_train, plot = True)
# preprocessing of validation data, get predictions

preds = model.predict(X_valid)



print('MAE:', mean_absolute_error(y_valid, preds))
# make the prediction using the resulting model

preds = model.predict(X_valid)



print('MSE:', mean_squared_error(y_valid, preds))
x_list = [X_train, X_valid]

y_list = [y_train, y_valid]



scoring = list(map(lambda x,y: round(model.score(x,y)*100, 2), x_list, y_list)) 

scoring
# get predictions test data

final_preds = model.predict(encod_test_data)
test_id = test_data.index.astype(str)

quant_05 = pd.DataFrame({'ForecastId_Quantile': test_id + '_0.05', 'TargetValue': 0.85*final_preds})

quant_50 = pd.DataFrame({'ForecastId_Quantile': test_id + '_0.5', 'TargetValue': final_preds})

quant_95 = pd.DataFrame({'ForecastId_Quantile': test_id + '_0.95', 'TargetValue': 1.15*final_preds})

all_predict = pd.concat([quant_05, quant_50, quant_95])
all_predict.describe()
all_predict.to_csv("submission.csv",index=False)