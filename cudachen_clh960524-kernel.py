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
df_train = pd.read_csv('/kaggle/input/timeseries/Train.csv')

df_train.head()
df_train['Datetime'] = pd.to_datetime(df_train['Datetime'], format='%d-%m-%Y %H:%M')

df_train.head()
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(rc={'figure.figsize':(30,4)})



sns.lineplot(x='Datetime', y='Count', data=df_train)

plt.show()
split_date = '2014-04-01'

train_split = df_train.loc[df_train['Datetime'] <= split_date].copy()

test_split = df_train.loc[df_train['Datetime'] > split_date].copy()
train_split.head()
test_split.head()
sns.set(rc={'figure.figsize':(30,4)})



sns.lineplot(x='Datetime', y='Count', data=train_split)

sns.lineplot(x='Datetime', y='Count', data=test_split)

plt.show()
def create_features(df, label=None):

    """

    Creates time seires from datetime index.

    """

    df['date'] = df.index

    df['hour'] = df['Datetime'].dt.hour

    df['day_of_week'] = df['Datetime'].dt.dayofweek

    df['quarter'] = df['Datetime'].dt.quarter

    df['month'] = df['Datetime'].dt.month

    df['year'] = df['Datetime'].dt.year

    df['day_of_year'] = df['Datetime'].dt.dayofyear

    df['day_of_month'] = df['Datetime'].dt.day

    df['week_of_year'] = df['Datetime'].dt.weekofyear

    

    X = df[['hour', 'day_of_week', 'quarter', 'month', 'year',

           'day_of_year', 'day_of_month', 'week_of_year']]

    if label:

        y = df[label]

        return X, y

    

    return X
X_train, y_train = create_features(train_split, label='Count')

X_test, y_test = create_features(test_split, label='Count')
import xgboost as xgb



reg = xgb.XGBRegressor(n_estimators=1000)

reg.fit(X_train, y_train,

       eval_set = [(X_train, y_train), (X_test, y_test)],

       early_stopping_rounds=50,

       verbose=True)
from xgboost import plot_importance, plot_tree



_ = plot_importance(reg, height=0.9)
sns.set(rc={'figure.figsize':(30,4)})



predict_split = test_split.copy()

predict_split['Count'] = reg.predict(X_test)



sns.lineplot(x='Datetime', y='Count', data=train_split)

sns.lineplot(x='Datetime', y='Count', data=test_split)

sns.lineplot(x='Datetime', y='Count', data=predict_split)

plt.show()
# Plot the forcast with the actuals

sns.set(rc={'figure.figsize':(30,4)})



sns.lineplot(x='Datetime', y='Count', data=test_split)

sns.lineplot(x='Datetime', y='Count', data=predict_split)



#sns.ylim('2014-04', '2014-05')



plt.show()
from sklearn.metrics import mean_squared_error, mean_absolute_error
mean_squared_error(y_true=test_split['Count'],

                  y_pred=predict_split['Count'])
mean_absolute_error(y_true=test_split['Count'],

                  y_pred=predict_split['Count'])
RMSE = np.sqrt(mean_squared_error(y_true=test_split['Count'],

                  y_pred=predict_split['Count']))

print(RMSE)
def create_window_data(data, window_size):

    X = []

    y = []

    i = 0

    while(i + window_size) < len(data):

        X.append(data[i:i + window_size])

        y.append(data[i + window_size])

        i += 1

    assert len(X) == len(y)

    return X, y



window_size = 7

X_train, y_train = create_window_data(train_split['Count'].values, window_size)

X_test, y_test = create_window_data(test_split['Count'].values, window_size)
xgb_model = xgb.XGBRegressor()
from sklearn.model_selection import cross_val_score,KFold

from sklearn.model_selection import GridSearchCV



tweaked_model = GridSearchCV(

    xgb_model,   

    {

        'max_depth':[1,2,5,10,20],

        'n_estimators':[20,30,50,70,100],

        'learning_rate':[0.1,0.2,0.3,0.4,0.5]

    },   

    cv = 5,   

    verbose = 1,

    n_jobs = -1,

    scoring = 'neg_mean_squared_error')



tweaked_model.fit(X_train, y_train)

print('Best: %f using %s'%(tweaked_model.best_score_, tweaked_model.best_params_))
tweaked_model.best_params_
y_predict = tweaked_model.predict(X_test)
'''

sns.set(rc={'figure.figsize':(30,4)})



sns.lineplot(data=y_test)

sns.lineplot(data=y_predict)

'''

plt.plot(y_test, color='blue', label='actual')

plt.plot(y_predict, color='red', label='predicted')

plt.legend(loc='best')

plt.title('RMSE:%.4f' % np.sqrt(mean_squared_error(y_true=y_test,

                  y_pred=y_predict)))

plt.show()

print('the next day predict is %.f' % y_predict[-1])
print('RMSE:%.4f' % np.sqrt(mean_squared_error(y_true=y_test,

                  y_pred=y_predict)))
print(test_split['Datetime'].values.shape)

print(y_predict.shape)



df_submit = pd.DataFrame({"Datetime": test_split['Datetime'].iloc[:y_predict.shape[0]], "Count": y_predict})



df_submit.to_csv("submit.csv", index=False)