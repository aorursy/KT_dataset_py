import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 



np.set_printoptions(suppress=True)



sns.set_palette("dark")

sns.set_style("whitegrid")
data = pd.read_csv("../input/all_data.csv")
data.head()
data.dtypes
import datetime as dt

data['date'] = data.apply(lambda x: dt.datetime.fromtimestamp(x['timestamp']), axis = 1)

data['year'] = data.apply(lambda x: dt.datetime.fromtimestamp(x['timestamp']).year, axis = 1)
plt.subplots(figsize=(15, 9))

plt.plot(data['date'], data['transactions'])

plt.xlabel("Date", fontsize = 15)

plt.ylabel("Number of Transactions", fontsize = 15)
plt.subplots(figsize=(15, 9))

plt.plot(data['date'], data['price_USD'])

plt.xlabel("Date", fontsize = 15)

plt.ylabel("Price_USD", fontsize = 15)
plt.subplots(figsize = (8, 8))

correlation_object = data.iloc[:, 1:8].corr()

sns.heatmap(correlation_object)
testing_data = data.loc[data['year'] == 2017]

training_data = data.loc[data['year'] != 2017]



del data
def remove_marketcapvalue():

    global testing_data, training_data

    testing_data.drop('market-cap-value', axis = 1, inplace = True)

    training_data.drop('market-cap-value', axis = 1, inplace = True)

    

remove_marketcapvalue()



training_data.head()
#Plot and remove the date column. 

plt.subplots(figsize=(15, 9))

plt.plot(training_data['date'], training_data['price_USD'])

plt.xlabel("Date", fontsize = 15)

plt.ylabel("Price_USD", fontsize = 15)
def drop_date():

    global training_data, testing_data

    training_data.drop('date', inplace = True, axis = 1)

    testing_data.drop('date', inplace = True, axis = 1)

    

drop_date()

training_data.dtypes

#what ever calculations are to be done. It's better to exclude 2017 since 2017 is not finished yet and the groupings' values might not be accurate. 

sns.pairplot(training_data.iloc[:, 1:], hue = "year")
#Lets come up with new features on time stamp like date, month etc. 



def date_features(dataframe):

    dataframe['day'] = dataframe.apply(lambda x: dt.datetime.fromtimestamp(x['timestamp']).day, axis = 1)

    dataframe['month'] = dataframe.apply(lambda x: dt.datetime.fromtimestamp(x['timestamp']).month, axis = 1)

    dataframe['week'] = dataframe.apply(lambda x: np.ceil(x['day']/7), axis = 1)

    return(dataframe)



training_data = date_features(training_data)

testing_data = date_features(testing_data)
training_data.groupby('month')['price_USD'].median().plot(kind = 'bar')

plt.ylabel("Median of Month", fontsize = 15)

plt.xlabel("Month", fontsize = 15)
#Let's create a new feature which describes whether the given month is 5|6|7

def is_month_567(dataframe):

    dataframe['is_month_567'] = [1 if x in [5,6,7] else 0 for x in dataframe['month']]

    return(dataframe)



training_data = is_month_567(training_data)

testing_data = is_month_567(testing_data)



training_data['is_month_567'].value_counts().to_frame().plot(kind = "bar")

plt.title("Count values")
#Preparing for modelling. 

def prepare_modelling():

    global training_data, testing_data

    concat_data = pd.concat([training_data, testing_data])

    target = concat_data['price_USD']

    concat_data.drop(['timestamp', 'price_USD'], axis = 1, inplace = True)

    

    #get index of test and train sets. 

    train_index = concat_data[concat_data['year'] != 2017].index.tolist()

    test_index = list(set(train_index).symmetric_difference(set(range(len(concat_data)))))

    

    #dummify data. 

    def dummified_data(dataframe):

        dataframe = pd.concat([pd.get_dummies(dataframe['year'], prefix= 'year', prefix_sep='_').reset_index(drop = True),  dataframe.reset_index(drop = True)], axis = 1)

        dataframe = pd.concat([pd.get_dummies(dataframe['month'], prefix='month', prefix_sep='_').reset_index(drop = True),  dataframe.reset_index(drop = True)], axis = 1)

        dataframe = pd.concat([pd.get_dummies(dataframe['day'], prefix='day', prefix_sep='_').reset_index(drop = True),  dataframe.reset_index(drop = True)], axis = 1)

        dataframe = pd.concat([pd.get_dummies(dataframe['week'], prefix='week', prefix_sep='_').reset_index(drop = True),  dataframe.reset_index(drop = True)], axis = 1)

    

        r_cols = ['year', 'month', 'day', 'week']

        dataframe.drop(r_cols, axis = 1, inplace = True)

    

        return(dataframe)

    concat_data = dummified_data(concat_data)

    

    train_X = concat_data.loc[train_index]

    train_Y = target[train_index]

    

    test_X = concat_data.loc[test_index]

    test_Y = target[test_index]

    

    return(train_X, train_Y, test_X, test_Y)





train_X, train_Y, test_X, test_Y = prepare_modelling()

    
import xgboost as xgb



from sklearn import cross_validation, model_selection

xgbfolds = model_selection.KFold(n_splits=5)



xgb_dtrain = xgb.DMatrix(train_X, train_Y)

xgb_dtest = xgb.DMatrix(test_X)



xgb_params = {'learning_rate' : 0.03, 

             'subsample' : 0.7,

             'max_depth' : 5,

             'colsample_bytree' : 0.8,

              'objective': 'reg:linear',

              'eval_metric': 'rmse',

             'silent': 0

             }

xgb_obj = xgb.cv(params = xgb_params, dtrain = xgb_dtrain, early_stopping_rounds=10,

                       verbose_eval=True, show_stdv=False, folds = xgbfolds, num_boost_round = 9999)
xgb_obj.plot(figsize = (15,10))
xgb = xgb.train(params = xgb_params, dtrain = xgb_dtrain, num_boost_round = 80)



predictions = xgb.predict(xgb_dtest)



from sklearn.metrics import mean_squared_error

mean_squared_error(y_true = test_Y, y_pred = predictions)