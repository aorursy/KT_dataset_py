import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import pylab

import seaborn as sns

from scipy.stats import skew

from scipy.stats import kurtosis

from scipy.stats.stats import pearsonr
data = pd.read_csv("../input/train.csv")

data = data.sample(frac=1, random_state=1).reset_index(drop=True) #Did this so train/test split would not be biased, since it's a time series

data.shape
data.head(n=5)
data.dtypes
plt.rcParams['figure.figsize'] = (6, 6)

data.hist(bins=10)

plt.tight_layout()

plt.show()
train = data.iloc[:-1000, :]

test = data.iloc[-1000:, :]

print(data.shape, train.shape, test.shape)
all_data = pd.concat((train.loc[:,'datetime':'windspeed'],

                      test.loc[:,'datetime':'windspeed']))
all_data.shape
from scipy.special import boxcox,inv_boxcox



plt.rcParams['figure.figsize'] = (8.0, 3.5)

prices = pd.DataFrame({".count":train["count"], "boxcox(count)":boxcox(train["count"], 0.384)})



prices.hist()
all_data.datetime = all_data.datetime.apply(pd.to_datetime)

all_data['month'] = all_data.datetime.apply(lambda x : x.month)

all_data['dow'] = all_data.datetime.apply(lambda x : x.weekday())

all_data['hour'] = all_data.datetime.apply(lambda x : x.hour)

all_data.head()
cats = ['holiday','season','weather','workingday', 'month', 'hour']



for cat in cats:

    all_data[cat] = all_data[cat].astype('str')
all_data = all_data.drop('datetime', axis=1)
#filling NA's with the mean of the column:

all_data = all_data.fillna(all_data.mean())
all_data.sample(n=4)
#MEA scorer

def mae(y, y_):

    return sum(abs(y-y_)) / len(y)
#RMSE scorer

from sklearn.metrics import mean_squared_error

def rmse(y, y_):

    return mean_squared_error(y, y_)**0.5
#RMSLE scorer

def rmsle(y, y_):

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))

    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))

    calc = (log1 - log2) ** 2

    return np.sqrt(np.mean(calc))
from sklearn import cross_validation, grid_search, linear_model, metrics, pipeline, preprocessing
train_labels = train['count'].values

train_data = all_data[:train.shape[0]]

test_labels = test['count'].values

test_data = all_data[train.shape[0]:]
print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)
binary_data_columns = ['holiday', 'workingday']

binary_data_indices = np.array([(column in binary_data_columns) for column in train_data.columns], dtype = bool)



categorical_data_columns = ['weather', 'month', 'dow','season'] 

categorical_data_indices = np.array([(column in categorical_data_columns) for column in train_data.columns], dtype = bool)



numeric_data_columns = ['atemp','temp', 'humidity', 'windspeed', 'hour']

numeric_data_indices = np.array([(column in numeric_data_columns) for column in train_data.columns], dtype = bool)
transformer_list = [        

            #binary

            ('binary_variables_processing', preprocessing.FunctionTransformer(lambda data: data[:, binary_data_indices])), 

                    

            #numeric

            ('numeric_variables_processing', pipeline.Pipeline(steps = [

                ('selecting', preprocessing.FunctionTransformer(lambda data: data[:, numeric_data_indices])),

                ('scaling', preprocessing.StandardScaler(with_mean = 0))            

                        ])),

        

            #categorical

            ('categorical_variables_processing', pipeline.Pipeline(steps = [

                ('selecting', preprocessing.FunctionTransformer(lambda data: data[:, categorical_data_indices])),

                ('hot_encoding', preprocessing.OneHotEncoder(handle_unknown = 'ignore'))            

                        ])),

        ]
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()



estimator = pipeline.Pipeline(steps = [       

    ('feature_processing', pipeline.FeatureUnion(transformer_list=transformer_list)),

    ('model_fitting', regressor)

    ]

)



estimator.fit(train_data, train_labels)

predicted = estimator.predict(test_data)



print("MAE:  ", mae(test_labels, predicted))

print("RMSE: ", rmse(test_labels, predicted))

print("RMSLE:", rmsle(test_labels, predicted))

print("R2:   ", estimator.score(test_data, test_labels))
pylab.figure(figsize=(5, 5))

pylab.subplot(1,1,1)

pylab.grid(True)

pylab.xlim(-100,1100)

pylab.ylim(-100,1100)

pylab.scatter(estimator.predict(train_data), train_labels, alpha=0.5, color = 'red')

pylab.scatter(predicted, test_labels, alpha=0.5, color = 'blue')

pylab.title('linear model')
plt.rcParams['figure.figsize'] = (5, 5)

plt.scatter(predicted, (test_labels - predicted))
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor()



estimator = pipeline.Pipeline(steps = [       

    ('feature_processing', pipeline.FeatureUnion(transformer_list=transformer_list)),

    ('model_fitting', regressor)

    ]

)



estimator.fit(train_data, train_labels)

predicted = estimator.predict(test_data)



print("MAE:  ", mae(test_labels, predicted))

print("RMSE: ", rmse(test_labels, predicted))

print("RMSLE:", rmsle(test_labels, predicted))

print("R2:   ", estimator.score(test_data, test_labels))
pylab.figure(figsize=(5, 5))

pylab.subplot(1,1,1)

pylab.grid(True)

pylab.xlim(-100,1100)

pylab.ylim(-100,1100)

pylab.scatter(estimator.predict(train_data), train_labels, alpha=0.5, color = 'red')

pylab.scatter(predicted, test_labels, alpha=0.5, color = 'blue')

pylab.title('random forest model')
plt.rcParams['figure.figsize'] = (5, 5)

plt.scatter(predicted, (test_labels - predicted))
cv_lmbdas = {}

lmbdas = list(np.linspace(-1, 1, num=10)) #Start broad, narrow down

lmbdas.sort()



for lmbda in lmbdas:

    estimator.fit(train_data, boxcox(train_labels, lmbda))

    predicted = inv_boxcox(estimator.predict(test_data), lmbda)

    cv_lmbdas[lmbda] = rmsle(test_labels, predicted)
#Find winning lambda

print(min(cv_lmbdas, key=cv_lmbdas.get)," : ", cv_lmbdas[min(cv_lmbdas, key=cv_lmbdas.get)])
pylab.figure(figsize=(12, 4))

series = pd.Series(cv_lmbdas, index=lmbdas)

series.plot(title = "Compare lambdas")

plt.xlabel("lmbda")

plt.ylabel("rmsle")
best_lambda = min(cv_lmbdas, key=cv_lmbdas.get)
estimator.fit(train_data, boxcox(train_labels, best_lambda))

predicted = inv_boxcox(estimator.predict(test_data), best_lambda)



print("MAE:  ", mae(test_labels, predicted))

print("RMSE: ", rmse(test_labels, predicted))

print("RMSLE:", rmsle(test_labels, predicted))

print("R2:   ", estimator.score(test_data, boxcox(test_labels, best_lambda)))
pylab.subplot(1,1,1)

pylab.grid(True)

pylab.xlim(-100,1100)

pylab.ylim(-100,1100)

pylab.scatter(inv_boxcox(estimator.predict(train_data), best_lambda), train_labels, alpha=0.5, color = 'red')

pylab.scatter(predicted, test_labels, alpha=0.5, color = 'blue')

pylab.title('random forest model - boxcox')
plt.scatter(predicted, (test_labels - predicted))