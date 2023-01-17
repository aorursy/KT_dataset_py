# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Load dataset

data = pd.read_csv('../input/Automobile_data.csv')

# List all features = columns

list(data.columns)
# Ensure that columns necessary for regression contain only numeric values

data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')

data['price'] = pd.to_numeric(data['price'], errors='coerce')



# Handle missing data: remove rows that contain missing data

data.dropna(subset=['horsepower', 'price'], inplace=True)



# data.dtypes

#data.head()
# import stats library

from scipy.stats import pearsonr
# Calculate a Pearson correlation coefficient and the p-value for testing non-correlation

pearsonr(data['horsepower'], data['price'])
# import plot library

from bokeh.io import output_notebook

from bokeh.plotting import figure, show, ColumnDataSource



# enable output to notebook cell

output_notebook()
source = ColumnDataSource(data=dict(

    x=data['horsepower'],

    y=data['price'],

    make=data['make']

))



# add tooltips to show infos for each datapoint

tooltips = [

    ('make', '@make'),

    ('horsepower', '$x'),

    ('price', '$y{$0}')

]



# create figure

p = figure(plot_width=600, plot_height=400, tooltips=tooltips)

# add axis labels

p.xaxis.axis_label = 'Horsepower'

p.yaxis.axis_label = 'Price'



# show datapoints as circles

p.circle('x', 'y', source=source, size=8, color='blue', alpha=0.5)



# show figure

show(p)
# import machine learning library

from sklearn.model_selection import train_test_split
# split dataset into training data and test data: 75% / 25%

train, test = train_test_split(data, test_size=0.25)
from sklearn import linear_model

model = linear_model.LinearRegression()

# reshape first array to 2D for .fit() method

training_x = np.array(train['horsepower']).reshape(-1, 1)

training_y = np.array(train['price'])



# perform linear fit/regression

model.fit(training_x, training_y)

# turn coefficient array which contains only one number into a number

slope = np.asscalar(np.squeeze(model.coef_))

intercept = model.intercept_

print('slope: ', slope, 'intercept: ', intercept)
# Add best fit line to figure

from bokeh.models import Slope



best_fit = Slope(gradient=slope, y_intercept=intercept, line_color='red', line_width=3)

p.add_layout(best_fit)

show(p)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# define a function to generate a prediction and then compare the desired metrics

def predict_metrics(lr, x, y):

    pred = lr.predict(x)

    mae = mean_absolute_error(y, pred)

    mse = mean_squared_error(y, pred)

    r2 = r2_score(y, pred)

    return mae, mse, r2



training_mae, training_mse, training_r2 = predict_metrics(model, training_x, training_y)



# calculate with test data for comparison

test_x = np.array(test['horsepower']).reshape(-1, 1)

test_y = np.array(test['price'])

test_mae, test_mse, test_r2 = predict_metrics(model, test_x, test_y)



print('training mean error: ', training_mae, 'training mse: ', training_mse, 'training r2: ', training_r2)

print('test mean error: ', test_mae, 'test mse: ', test_mse, 'test r2: ', test_r2)
cols = ['horsepower', 'engine-size', 'peak-rpm', 'length', 'width', 'height']

#preprocess the data

for col in cols:

    data[col] = pd.to_numeric(data[col], errors='coerce')



data.dropna(subset=['price', 'horsepower'], inplace=True)



for col in cols:

    print(col, pearsonr(data[col], data['price']))
# drop peak-rpm & height from further analysis as they are weakly correlated



# split data into train & test set

model_cols = ['horsepower', 'engine-size', 'length', 'width']

multi_x = np.column_stack(tuple(data[col] for col in model_cols))

multi_train_x, multi_test_x, multi_train_y, multi_test_y = train_test_split(multi_x, data['price'], test_size=0.25)
# fit the model

multi_model = linear_model.LinearRegression()

multi_model.fit(multi_train_x, multi_train_y)

multi_intercept = multi_model.intercept_

multi_coeffs = dict(zip(model_cols, multi_model.coef_))

print('intercept: ', multi_intercept)

print('coefficients: ', multi_coeffs)
# calculate error metrics

multi_train_mae, multi_train_mse, multi_train_r2 = predict_metrics(multi_model, multi_train_x, multi_train_y)

multi_test_mae, multi_test_mse, multi_test_r2 = predict_metrics(multi_model, multi_test_x, multi_test_y)



print('training mean error: ', multi_train_mae, 'training mse: ', multi_train_mse, 'training r2: ', multi_train_r2)

print('test mean error: ', multi_test_mae, 'test mse: ', multi_test_mse, 'test r2: ', multi_test_r2)