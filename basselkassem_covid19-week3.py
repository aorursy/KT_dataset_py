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
import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import lightgbm as lgb

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_squared_log_error



mpl.rcParams['axes.grid']=True

pd.options.display.max_rows = 99





train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/submission.csv')



train.Date = pd.to_datetime(train.Date)

test.Date = pd.to_datetime(test.Date)



def fill_province(row):

    if pd.isna(row['Province_State']):

        row['Province_State'] = '_PROVINCE_' + row['Country_Region']

    return row



train = train.apply(fill_province, axis = 1)

test = test.apply(fill_province, axis = 1)



def extract_time_features(df):

    df['Day'] = df['Date'].dt.day

    df['Day_of_Week'] = df['Date'].dt.dayofweek

    df['Day_of_Year'] = df['Date'].dt.dayofyear

    df['Week_of_Year'] = df['Date'].dt.weekofyear

    df['Days_im_Month'] = df['Date'].dt.days_in_month



extract_time_features(train)

extract_time_features(test)



train_col_to_delete = ['Id', 'ConfirmedCases', 'Fatalities', 'Country_Region', 'Province_State', 'Date' ]

test_col_to_delete = ['ForecastId', 'Date', 'Country_Region', 'Province_State']

validation_duration = 2

validation_duration = np.timedelta64(validation_duration - 1, 'D')

print(validation_duration)



def train_val_split(df, display = False):

    split_thr = df['Date'].max() - validation_duration 

    df_train = df[df['Date'] < split_thr ]

    X_train = df_train.drop(columns = train_col_to_delete)

    y_cc_train = df_train[['ConfirmedCases']]

    y_fa_train = df_train[['Fatalities']]



    df_val= df[df['Date'] >= split_thr ]

    X_val = df_val.drop(columns = train_col_to_delete)

    y_cc_val = df_val[['ConfirmedCases']]

    y_fa_val = df_val[['Fatalities']]



    if display:

        print('data shape:', df.shape)

        print('train shape:', df_train.shape)

        print('val shape:', df_val.shape)

    return(X_train, y_cc_train, y_fa_train, X_val, y_cc_val, y_fa_val)



def plot_feature_importance(model, X):

    feat_importance = pd.DataFrame(sorted(zip(model.feature_importance(importance_type = 'gain'), X.columns)), columns=['Score','Feature'])

    feat_importance = feat_importance.sort_values(by = "Score", ascending = False)

    plt.figure(figsize = (8, 8))

    sns.barplot(x = "Score", y = "Feature", data = feat_importance)

    plt.title('LightGBM Features')

    plt.tight_layout()

    plt.show()

    return feat_importance.reset_index(drop = True)



def create_model(X_train, y_train, X_val, y_val, draw_metics = False):

    n_estimators = 100

    params = {

  'metric': 'rmse',

  'objective': 'mse',

  'verbose': 0, 

  'learning_rate': 0.99,

    }

    d_train = lgb.Dataset(X_train, y_train)

    d_valid = lgb.Dataset(X_val, y_val)

    watchlist = [d_train, d_valid]

    evals_result = {}

    model = lgb.train(params,

                    d_train, 

                    n_estimators,

                    valid_sets = watchlist, 

                    evals_result = evals_result, 

                    early_stopping_rounds = 10,

                    verbose_eval = 0,

                    )

    if draw_metics:

        lgb.plot_metric(evals_result) 

    return model



def rmse(y, y_hat):

    return np.sqrt(mean_squared_error(y, y_hat))



def rmsle(y, y_hat):

    y_hat = np.where(y_hat < 0, 0, y_hat)

    return 'rmsle', np.sqrt(mean_squared_log_error(y, y_hat))



def evaluate_model(model, X_train, y_train, X_val, y_val): 

    y_hat = model.predict(X_train)

    print('Training error;', rmsle(y_train, y_hat))

    y_val_hat = model.predict(X_val)

    print('Validation error:', rmsle(y_val, y_val_hat))





# Modeling for all country



submission = pd.DataFrame(columns = submission.columns)



for country in train['Country_Region'].unique():

    print('start modeling for ', country, '...')

    provinces = train[train['Country_Region'] == country]['Province_State'].unique()

    for province in provinces:

        country_df = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]

        X_train, y_cc_train, y_fa_train, X_val, y_cc_val, y_fa_val = train_val_split(country_df)

        model_cc = create_model(X_train, y_cc_train, X_val, y_cc_val)

        model_fa = create_model(X_train, y_fa_train, X_val, y_fa_val)



        test_df = test[(test['Country_Region'] == country) & (test['Province_State'] == province)]

        forcast_id = test_df['ForecastId'].values.tolist()



        X_test = test_df.drop(columns=test_col_to_delete)

        y_cc_hat = model_cc.predict(X_test)

        y_fa_hat = model_fa.predict(X_test)



        test_res = pd.DataFrame(columns=submission.columns)

        test_res['ForecastId'] = forcast_id

        test_res['ConfirmedCases'] = y_cc_hat

        test_res['Fatalities'] = y_fa_hat

        submission = submission.append(test_res)



for col in ['ConfirmedCases', 'Fatalities']:

    submission.loc[submission[col] < 0, col] = 0



submission.to_csv('submission' + '.csv', index = 0)