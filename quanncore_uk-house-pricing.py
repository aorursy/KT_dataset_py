import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime as dt





import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px





import math

import datetime as datetime

from fbprophet import Prophet

from sklearn.metrics import mean_absolute_error as MAE

from sklearn.metrics import r2_score as R2

import xgboost









%matplotlib inline
# utils functions



# remove outliers by using percentile ranges 

def remove_outlier_by_group(df, by, column, percentile_range=[25, 75]):

    # outlier detection with percentile 

    def is_outlier(x, percentile_range=[25, 75]):

        quartile_1, quartile_3 = np.percentile(x, percentile_range)

        iqr = quartile_3 - quartile_1

        lower_bound = quartile_1 - (iqr * 1.5)

        upper_bound = quartile_3 + (iqr * 1.5)

        return (x > upper_bound) | (x < lower_bound)



    removed_df = df[~df.groupby(by)[column].apply(is_outlier, percentile_range=percentile_range)]

    print(f'Removed # points: {len(df) - len(removed_df)}')

    

    return removed_df



# get desired columns from renamed dummy columns

def filter_cols(col_list, desired_columns):

    filtered_col_list = []

    for n_col in col_list:

        n_col_split = n_col.split('_')

        if any(o_col in n_col_split[0] for o_col in desired_columns):

            filtered_col_list.append(n_col)

    

    return filtered_col_list

           

# build a prophet model

def build_model(train_df, test_df, original_regressor_list, log_transform=True):

    # there should me more hyperparam tuning in ideal case

    m = Prophet(seasonality_mode='multiplicative')

    # because col names has been changed on dummy variable creation, select all cols with original names

    regressor_col_list = filter_cols(train_df.columns, original_regressor_list)

    for col in regressor_col_list:

        print(f'Adding a regressor: {col}')

        m.add_regressor(col, mode='multiplicative')

    

    if log_transform:

        train_df['y'] = np.log(train_df['y'] + 1)

        test_df['y'] = np.log(test_df['y'] + 1)

    

    m.fit(train_df)

    

    future_df = test_df.drop(['y'], axis=1)

    past_df = train_df.drop(['y'], axis=1)



    forecast_test = m.predict(future_df)

    forecast_train = m.predict(past_df)

    

    if log_transform:

        train_df['y'] = np.exp(train_df['y']) - 1

        test_df['y'] = np.exp(test_df['y']) - 1

        forecast_train['yhat'] = np.exp(forecast_train['yhat']) - 1

        forecast_test['yhat'] = np.exp(forecast_test['yhat']) - 1

        

    forecast_train.loc[:,'y'] = train_df.loc[:,'y']

    forecast_test.loc[:,'y'] = test_df.loc[:,'y']

    # clip negative values

    forecast_train.loc[:,'yhat'] = forecast_train.yhat.clip(lower=0)

    forecast_train.loc[:,'yhat_lower'] = forecast_train.yhat_lower.clip(lower=0)

    forecast_test.loc[:,'yhat'] = forecast_test.yhat.clip(lower=0)

    forecast_test.loc[:,'yhat_lower'] = forecast_test.yhat_lower.clip(lower=0)

    

    complete_forecast = pd.concat([forecast_train, forecast_test], axis=0)

    

    return m, complete_forecast, forecast_train, forecast_test



# plot result of time series forecasted by prophet

def plot_complete_data(train_df, test_df, seperation_date=dt.datetime(2016, 1, 1)):

    f, ax = plt.subplots(figsize=(14, 8))

    ax.plot(train_df['ds'], train_df['y'], 'ko', markersize=3)

    ax.plot(train_df['ds'], train_df['yhat'], color='steelblue', lw=0.5)

    ax.fill_between(train_df['ds'], train_df['yhat_lower'], train_df['yhat_upper'], 

                    color='steelblue', alpha=0.3)

    

    ax.plot(test_df['ds'], test_df['y'], 'ro', markersize=3)

    ax.plot(test_df['ds'],test_df['yhat'], color='coral', lw=0.5)

    ax.fill_between(test_df['ds'], test_df['yhat_lower'], test_df['yhat_upper'], color='coral', alpha=0.3)

    

    ax.axvline(seperation_date, color='0.8', alpha=0.7)

    ax.grid(ls=':', lw=0.5)

    

    return f



# plot joint distribution for given datasets (observed, forecasted)

def plot_joint_distr(df, x='yhat', y='y', title=None):

    g = sns.jointplot(x=x, y=y, data = df, kind="reg", color="0.4")

    

    g.fig.set_figwidth(8)

    g.fig.set_figheight(8)



    ax = g.fig.axes[1]

    

    if title is not None: 

        ax.set_title(title, fontsize=16)



    ax = g.fig.axes[0]



    ax.text(100, 2500, "R = {:+4.2f}\nMAE = {:4.1f}".format(df.loc[:,[y, x]].corr().iloc[0,1], 

                                                            MAE(df.loc[:,y].values, df.loc[:,x].values)), 

            fontsize=16)



    ax.set_xlabel("model's estimates", fontsize=15)

    

    ax.set_ylabel("observations", fontsize=15)

    

    ax.grid(ls=':')



    [l.set_fontsize(13) for l in ax.xaxis.get_ticklabels()]

    [l.set_fontsize(13) for l in ax.yaxis.get_ticklabels()];



    ax.grid(ls=':')
# read dataframe and cast the columns to desired types

uk_housing_prices = pd.read_csv('../input/ukhouseprice/pp-complete.csv', names=["Price", "DoT", "Property Type", "Old/New", "Duration", "Town/city"],

                               usecols=[1, 2, 4, 5, 6, 11,], index_col=False, parse_dates=["DoT"], 

                                )

dtype = {"Price" : float,  "Property Type": str, 'Old/New': str, "Duration": str, "Town/city": str}

uk_housing_prices = uk_housing_prices.astype(dtype)
uk_housing_prices.head()
uk_housing_prices.isnull().sum()
uk_housing_prices.describe()
# extract day, month and year and other useful features

uk_housing_prices['day'] = pd.DatetimeIndex(uk_housing_prices['DoT']).day

uk_housing_prices['month'] = pd.DatetimeIndex(uk_housing_prices['DoT']).month

uk_housing_prices['year'] = pd.DatetimeIndex(uk_housing_prices['DoT']).year

uk_housing_prices['quarter'] = pd.DatetimeIndex(uk_housing_prices['DoT']).quarter



# subsample in each year of each month

fraction = .02

uk_housing_prices = uk_housing_prices.groupby(['year', 'month'], group_keys=False).apply(pd.DataFrame.sample, frac=fraction).reset_index(drop=True)



# add a new column indicating whether a sale occured in London or not

uk_housing_prices['IsLondon'] = 0

uk_housing_prices.loc[uk_housing_prices['Town/city'].str.contains("london", case=False), 'IsLondon'] = 1
uk_housing_prices['Price'].describe()
uk_housing_prices.nlargest(10, 'Price')
fig = px.histogram(uk_housing_prices, x="Price", nbins=50, log_y =True)

fig.show()
fig = px.box(uk_housing_prices, y="Price", log_y =True)

fig.show()
fig = px.box(uk_housing_prices, x='Old/New', y="Price", log_y =True)

fig.show()
fig = px.box(uk_housing_prices, x='IsLondon', y="Price", log_y =True)

fig.show()
temp = uk_housing_prices.groupby('year')['year', 'Price'].mean()

fig = px.bar(temp, x="year", y="Price", title='Mean of Price over years')

fig.show()



temp = uk_housing_prices.groupby(['year','month'])['Price'].mean()

fig = px.line(temp, y="Price", title='Mean of Price over months for all years')

fig.show()
r_uk_housing_prices = remove_outlier_by_group(uk_housing_prices, 'year', 'Price', percentile_range=[25, 75]).reset_index(drop=True)

fig = px.histogram(r_uk_housing_prices, x="Price", nbins=50, log_y =True)

fig.show()

# rename the columns so that prophet model can work with them

uk_housing_prices_dummy = r_uk_housing_prices.rename(columns={'DoT': 'ds', 'Price': 'y'})



# apply simple one hot encoding for categorial features

uk_housing_prices_dummy = pd.get_dummies(uk_housing_prices_dummy[["y", "ds", "day", "month", "year", "quarter", 

                                                            "Property Type", "Old/New", "Duration", "IsLondon"]], drop_first=True)



# split train/test set

train_df = uk_housing_prices_dummy[uk_housing_prices_dummy['ds'] < '2016-01-01'].sort_values('ds').reset_index(drop=True)

test_df = uk_housing_prices_dummy[uk_housing_prices_dummy['ds'] > '2016-01-01'].sort_values('ds').reset_index(drop=True)
# columns will be used as regressors

original_regressor_list = ['IsLondon', 'Duration', 'Property Type']



# build, train the model and get forecasted results for train and test set

# take the log of prices  and use multiplicative decomposition because of its non-stationary nature

m, complete_forecast, forecast_train, forecast_test = build_model(train_df, test_df, original_regressor_list, log_transform=True)
complete_forecast.sort_values(['ds'], ascending=False).head(50)
m.plot(complete_forecast)
plot_complete_data(forecast_train, forecast_test)
plot_joint_distr(forecast_train, title='Training set')
plot_joint_distr(forecast_test, title='Test set')
# fit a boosting tree regressor

desired_cols = ['day', 'month', 'year', 'quarter', 'IsLondon', 'Duration', 'Property Type']

filtered_cols = filter_cols(train_df.columns, desired_cols)

model = xgboost.XGBRegressor(colsample_bytree=0.4,

                             gamma=0,                 

                             learning_rate=0.07,

                             max_depth=3,

                             min_child_weight=1.5,

                             n_estimators=10000,                                                                    

                             reg_alpha=0.75,

                             reg_lambda=0.45,

                             subsample=0.6,

                             seed=42)

model.fit(train_df[filtered_cols],train_df['y'])
# check the metrics on test set

preds = model.predict(test_df[filtered_cols])

print(f'MAE: {MAE(test_df["y"], preds)}, R2: {R2(test_df["y"], preds)}')