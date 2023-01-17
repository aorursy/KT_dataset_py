!pip install nose
#from google.colab import drive

#drive.mount('/content/drive')
import pickle

from urllib.request import urlopen



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



np.random.seed(42)

np.random.RandomState(42)



import warnings

import sklearn.exceptions

warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=FutureWarning)



import seaborn as sns



#import unittest

from nose.tools import *



import time



import statsmodels.api as sm

from scipy import stats

from scipy.stats import norm, skew, kurtosis



from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV,cross_val_score 

from sklearn import preprocessing

from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, AdaBoostRegressor

from xgboost.sklearn import XGBRegressor

from sklearn.metrics import mean_absolute_error,r2_score, mean_squared_log_error,mean_squared_error, make_scorer

london= pd.read_csv('https://www.dropbox.com/s/z0twef3ygv8budm/london_merged.csv?dl=1')

assert_is_not_none(london)

assert_is_instance(london, pd.DataFrame)

assert_equal(london.shape, (17414,10))

london.head()
london.dtypes
london.describe()
london.info()
london['timestamp'] = pd.to_datetime(london['timestamp'], format ="%Y-%m-%d %H:%M:%S")
london['weather_code'].unique()

#1 = Clear ; mostly clear but have some values with haze/fog/patches of fog/ fog in vicinity

#2 = scattered clouds / few clouds

#3 = Broken clouds

#4 = Cloudly

#7 = Rain/ light Rain shower/ Light rain

#10 = rain with thunderstorm

#26 = snowfall

#94 = Freezing Fog

weather_dict = {1 : 100,  2 : 100,  3 : 100,  4 : 100, 

                7 : 200, 10 : 200, 26 : 200, 94 : 200}

london['weather_code']=london['weather_code'].replace(weather_dict)
london['count_log'] = np.log1p(london['cnt'])
def plot_distribution(london,columns,**kwargs ):

    """

    Function to plot a dataframe feature distribution.

    Input: 

    df - pandas DataFrame.

    columns - list of the df columns that should be plotted

    Output: multiple feature histograms

    

    Note: String columns should be dropped before passing to the function.

    """

    assert_is_not_none(london)

    assert_is_instance(london, pd.DataFrame)

    

    fig, axes = plt.subplots(ncols=len(columns),figsize=(30,5))

    for axs, col in zip(axes, columns):

        sns.distplot(london[col], ax=axs,**kwargs)

    fig.suptitle('Feature distribution')

    

plot_distribution(london,london.drop(['timestamp'],axis=1).columns, kde=False)
london.head()
def plot_by_time(london,column,main_title,yaxis):

    """

    Plot values by timestamp.

    Input:

    london - pandas DataFrame

    column - name of the column to plot

    main_title - string for title of the plot

    yaxis -string for yaxis label

    

    """

    assert_true(column in set(london.columns))

    

    

    ax = london.plot(x='timestamp',y=column, rot=90)

    plt.title(main_title)

    plt.ylabel(yaxis)

    plt.show()



#ax.set_xticklabels(pd.to_datetime(london.timestamp), rotation=90)

plot_by_time(london,'t1','Temperature by date','Temperature in C')

plot_by_time(london,'t2','Temperature "feels like" by date','Temperature in C')

plot_by_time(london,'hum','humidity by date','% humidity')

plot_by_time(london,'wind_speed','Wind speed by date','wind km/h')

plot_by_time(london,'cnt','Bike shares by date','count')

plot_by_time(london,'count_log','Bike shares by date','log of count')
london_non_weekend = london[london['is_weekend'] == 0]

london_non_weekend = london_non_weekend.drop(['is_holiday','is_weekend'],axis=1)

london_is_weekend = london[london['is_weekend'] == 1]

london_is_weekend = london_is_weekend.drop(['is_holiday','is_weekend'],axis=1)

london_is_holiday = london[london['is_holiday'] == 1]

london_is_holiday = london_is_holiday.drop(['is_holiday','is_weekend'],axis=1)

london_non_holiday = london[london['is_holiday'] == 0]

london_non_holiday = london_non_holiday.drop(['is_holiday','is_weekend'],axis=1)
def hourly_plot(df,title):

    """

    Function for plotting bike shares by hour.

    input: 

    df - pandas dataframe 

    title - main title of the plot

    

    """

    assert_true('timestamp' in set(df.columns))

    assert_true('cnt' in set(df.columns))

    

    df.groupby(by=df.timestamp.dt.hour)['cnt'].mean().plot()

    plt.title(title)

    plt.ylabel('count')

    plt.legend(['workdays','weekend','holidays'],loc=2, fontsize = 'medium')

    

hourly_plot(london_non_weekend, 'Bike shares by hour')

hourly_plot(london_is_weekend, 'Bike shares by hour')

hourly_plot(london_is_holiday, 'Bike shares by hour')

london.corr()
london_is_weekend.corr()
london_non_weekend.corr()
def plot_heatmap_corr(df):

    """

    Plot correlation heatmap from pandas dataframe.

    Input:pandas dataframe

    

    Output: Correlation heatmap

    

    """

    assert_is_not_none(df)

    assert_is_instance(df, pd.DataFrame)

    

    f, ax = plt.subplots(figsize=(8, 8))

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(df.corr(), cmap=cmap, vmax=1.0, vmin=-1.0, center=0,

                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.title('Correlation matrix')

    

plot_heatmap_corr(london)
def add_non_workday(df):

    """

    The function is grouping "is_holiday" and "is_weekend" columns into a new one "is_non_workday"

    Input: pandas DataFrame that contains columns 'is_holiday','is_weekend'

    """

    assert_true('is_holiday' in set(df.columns))

    assert_true('is_weekend' in set(df.columns))

    

    df['is_non_workday'] = df['is_holiday'] + df['is_weekend']

    df = df.drop(['is_holiday','is_weekend'],axis=1)

    

    assert_true('is_non_workday' in set(df.columns))

    

    return df
def add_month(df):

    """

    The function is extracting the month of a timestamp into a new column.

    Input: pandas DataFrame that contains 'timestamp' column

    """

    assert_true('timestamp' in set(df.columns))

    

    df['month'] = df['timestamp'].dt.month

    

    assert_true('month' in set(df.columns))

    return df
def add_year(df):

    """

    The function is extracting the year of a timestamp into a new column.

    Input: pandas DataFrame that contains 'timestamp' column

    """

    assert_true('timestamp' in set(df.columns))

    

    df['year'] = df['timestamp'].dt.year

    

    assert_true('year' in set(df.columns))

    return df
def add_day_of_week(df):

    """

    The function is extracting the day of the week of a timestamp into a new column.

    Input: pandas DataFrame that contains 'timestamp' column

    """

    assert_true('timestamp' in set(df.columns))

    

    df['day']=df['timestamp'].dt.dayofweek

    

    assert_true('day' in set(df.columns))

    return df

def add_hour(df):

    """

    The function is extracting the hour of a timestamp into a new column.

    Input: pandas DataFrame that contains 'timestamp' column

    """

    assert_true('timestamp' in set(df.columns))

    

    df['hour'] = df['timestamp'].dt.hour

    

    assert_true('hour' in set(df.columns))

    return df
def add_encode(df, column, max_value):

    """

    The function is encoding time series cyclical features with sin and cos.

    Input: 

    ---------

    df - pandas DataFrame

    column - column name

    max_value - column max value

    Output: 

    -----------

    -same dataframe with _sin and _cos columns added

    """

    assert_true(column in set(df.columns))

    

    df[column + '_sin'] = np.sin(2 * np.pi * df[column] / max_value)

    df[column + '_cos'] = np.cos(2 * np.pi * df[column] / max_value)

    

    assert_true((column + '_sin') in set(df.columns))

    assert_true((column + '_cos') in set(df.columns))

    return df
london=add_non_workday(london)

add_month(london)

add_day_of_week(london)

add_hour(london)

add_year(london)

add_encode(london,'hour',23)

add_encode(london,'month',12)
london[london['count_log']<6].groupby(by='season').count()
london[london['count_log']<6].groupby(by='day').count()
london[london['count_log']<6].groupby(by='month').count()
london[london['count_log']<6].groupby(by='hour').count()
london[london['count_log']<6].groupby(by='is_non_workday').count()
def add_night_hours(df):

    """

    The function is creating a new column "is_night". It requires a column with hours.

    If the hour is from 8:00 to 20:00 the data is classified as 0, if not in this interval it's 1.

    The purpose is try to to catch the sunlight status, but it's not taking in acount the time of the year and summertime.

    

    Input:

    df- pandas DataFrame, containing 'hour' column

    Output :

    Pandas DataFrame with added "is_night column"

    """

    assert_true('hour' in set(df.columns))

    

    df['is_night'] = 0

    df.loc[(df['hour'] < 8) | (df['hour'] > 20), 'is_night'] = 1

    

    assert_true('is_night' in set(df.columns))

    return df
add_night_hours(london)

def sns_hist(data,title):

    """"

    Function to plot seaborn histogram.

    """

    assert_is_not_none(data)

    

    sns.distplot(data, fit=norm)

    plt.title(title)

    plt.ylabel('Density')
sns_hist(london[london['is_night']==1]['count_log'],"Histogram of logarithm transformed count of bike shares during the night.")
sns_hist(london[london['is_night']==0]['count_log'],"Histogram of logarithm transformed count of bike shares during the day.")
london_df = london.drop(['timestamp','cnt'],axis=1)

X = london_df.drop(['count_log'], axis=1)

y = london_df['count_log']

scaler_x = preprocessing.MinMaxScaler()

X =  pd.DataFrame(scaler_x.fit_transform(X), columns = X.columns)
def df_split(df,train_percent):

    """

    Function to split DataFrame/Series on percentage value.

    Input:

    -------

    -df - pandas DataFrame

    -train_percent - the percentage /100 of the data that we want to split. value between [0,1]

    

    Output:

    --------

    -two pandas DataFrames

    """

    assert_is_not_none(df)

    assert_greater(train_percent,0)

    assert_less(train_percent,1)

    

    split_index = int(train_percent * len(df))

    train = df.iloc[:split_index]

    test = df.iloc[split_index:]

    return train,test
X_train,X_test = df_split(X,0.7)

y_train,y_test = df_split(y,0.7)
X_train.shape
y_train.shape
X_test.shape
y_test.shape
london.shape
sns_hist(pd.DataFrame(y_train),"Histogram of logarithm transformed count of bike shares train set")
sns_hist(pd.DataFrame(y_test),"Histogram of logarithm transformed count of bike shares test set")

#sns.distplot(pd.DataFrame(y_test), fit=norm)

#plt.title("Histogram of logarithm transformed count of bike shares test set")

#plt.ylabel('Density')
def rmsle(y, y_pred):

    """

    Root squared logarithmic loss function.

    The function is trimming the negative values and replace them with 0. After that calculates the RMSLE.

    Input:

    y - true values

    y_pred - predicted values



    Output: RMSLE

    """

    assert(y.shape == y_pred.shape)

    y = np.expm1(y)

    y_pred=y_pred.clip(min=0)

    y_pred=y_pred.clip(max=10)

    y_pred = np.expm1(y_pred)

    return np.sqrt(mean_squared_log_error( y, y_pred))





rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

tscv = TimeSeriesSplit(n_splits=5)
columns = ['weather_code','season','is_non_workday']

X_dummies = pd.get_dummies(X_train, columns=columns)

X_dummies = X_dummies.drop(['day','year','month'],axis=1)

def test_algorithms():

    

    """

    A custom function designed for cross validation score on "london bike sharing dataset".

    The function is comparing the following algorithms :

    "LinearRegression", "Random Forrest", "XGBoost", "SVR" , "AdaBoost","BaggingRegressor",

    on "London bicycle sharing dataset" with mean RMSLE error, standard deviation on RMSLE and execution time. 

    The output of this function is a pandas dataframe.

    """

    names = ["LinearRegression", "Random Forrest", "XGBoost", "SVR" , "AdaBoost",

             "BaggingRegressor"]

    

    regressors = [

        LinearRegression(),

        RandomForestRegressor(random_state=42),

        XGBRegressor(objective ='reg:squarederror',random_state=42),

        SVR(),

        AdaBoostRegressor(random_state=42),

        BaggingRegressor(random_state=42)    

        ]

    

    data_X = [X_dummies,X_train ,X_train ,X_dummies ,X_train ,X_train ]

    reg_columns=['algorithm','score_rmsle_mean','score_std','time']

    reg_performance = pd.DataFrame(columns=reg_columns)

    print('Please wait 1-2 minutes for all algorithms to complete.')

    for name, regressor, X_trains in zip(names, regressors,data_X):

        time_start = time.time()

        cv_results = cross_val_score(regressor, X_trains,y_train, cv=5 ,scoring = rmsle_scorer)

        time_end = round(time.time() - time_start,3)

        mean_score = round(-cv_results.mean(),4)

        std_score = round(cv_results.std(),4)

        t= pd.DataFrame([[name,mean_score,std_score,time_end]],columns = reg_columns)

        reg_performance = reg_performance.append(t, ignore_index=True)

        print(name , ' RMSLE = ',mean_score , 'with std=',std_score ," execution_time =  ", time_end,"s")

    return reg_performance

reg_performance = test_algorithms()

reg_performance
def plot_performance(df,x,y):

    """

    Function to bar plot algorithm performance.

    Inputs

    -------

    - df: DataFrame generated by test_algorithms() function

    - x: x column name, usually 'algorithm'

    - y: y column name - 'score_rmsle_mean' or 'time'

    

    Returns

    -------

    Barplot of the performance

    """

    assert_is_not_none(df)

    assert_true(x in set(df.columns))

    assert_true(y in set(df.columns))

    

    

    sns.barplot(data=df, x=x,y=y)

    plt.title('Performance of different algorithms')

    plt.xticks( rotation='vertical')





plot_performance(reg_performance, x='algorithm',y='score_rmsle_mean')

#################CAUTION LONG_RUNNING_CODE#####################################



"""

The following code is used to optimize 3 machine learning algorithms and export

the trained models to a pickle file. Uncomment if needed to check the results.



"""





def save_pickle(model):

    """

    Export sklearn trained model to a file.

    Input:

    ----------

    - model:sklearn trained model

    """

    pickle.dump(model, open(model.__class__.__name__ + '.model', 'wb'))

    print('Model saved as ' + model.__class__.__name__ + '.model')



###############################################################################

############################XGBoost############################################

#grid_values = {'learning_rate': [0.001, 0.01, 0.1, 0.3],

#               'n_estimators':[100,700,1000,1200],

#               'max_depth': [3,5,8,10],

#               #'min_child_weight': [1,2,3,4],

#               #'gamma':[0, 0.1, 0.2, 0.3]

#               #'reg_alpha': [0.1,1,200,500],#L1

#               #'reg_lambda':[1,200,500],#L2

#               }

#

#

#

#grid_xgb = GridSearchCV(XGBRegressor(objective = 'reg:squarederror',early_stopping_rounds=20, eval_metric="rmse",random_state=42),

#                        n_jobs=-1, param_grid = grid_values,cv=tscv,scoring = rmsle_scorer)

#grid_xgb.fit(X_train, y_train)

#save_pickle(grid_xgb.best_estimator_)

#pd.DataFrame(grid_xgb.cv_results_).to_csv('XGB_cv_results.csv')

##############################################################################

######################RandomForrest###########################################

#

#grid_values_rf = {'n_estimators': [10,100,500,750,1000,1200],

#                  'max_depth' : [5,7,8,9],

#                  'min_samples_leaf': [5,10,25,50],

#                #'max_features': ['auto', 'sqrt'],

#                }

#

#grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid = grid_values_rf,cv=tscv,scoring = rmsle_scorer)

#grid_rf.fit(X_train, y_train)

#save_pickle(grid_rf.best_estimator_)

#pd.DataFrame(grid_rf.cv_results_).to_csv('RF_cs_results.csv')

################################################################################

########################BaggringRegressor#######################################

#grid_values_br = {'n_estimators':[10,100,500,1000]}

#

#grid_br = GridSearchCV(BaggingRegressor(random_state=42) , param_grid = grid_values_br,cv=tscv,scoring = rmsle_scorer)

#grid_br.fit(X_train, y_train)

#

##################save_pickle(grid_br.best_estimator_)#####skipped due very long size on disk

#pd.DataFrame(grid_br.cv_results_).to_csv('BR_cv_results.csv')

################################################################################

################################################################################
#!cp 'XGBRegressor.model' 'drive/My Drive/XGBRegressor.model'

#!cp 'XGB_cv_results.csv' 'drive/My Drive/XGB_cv_results.csv'



#!cp 'RandomForestRegressor.model' 'drive/My Drive/RandomForestRegressor.model'

#!cp 'RF_cs_results.csv' 'drive/My Drive/RF_cs_results.csv'
def load_pickle(url):

    """

    Funtion to load existing trained model, previously saved by pickle.

    Inputs

    -------

    - url: DataFrame generated by test_algorithms() function

    

    Returns

    -------

    - sklearn trained model

    

    """

    try:

        loaded_model = pickle.load(urlopen(url))

        print('Model loaded.')

        return loaded_model

    except:

        print('Unable to load the model from url.')

        return 0
xgb_model_url = 'https://www.dropbox.com/s/zihko63rehvmpb6/XGBRegressor.model?dl=1'

rf_model_url = 'https://www.dropbox.com/s/9o62bwl4d8zmqx8/RandomForestRegressor.model?dl=1'





xgb_results_url = 'https://www.dropbox.com/s/tb7125oq2g5fqt4/XGB_cv_results.csv?dl=1'

rf_results_url = 'https://www.dropbox.com/s/dluxmpewpu3fen7/RF_cs_results.csv?dl=1'





br_results_url = 'https://www.dropbox.com/s/hkeselt1tbnsgx0/BR_cv_results.csv?dl=1'
xgb_tuned = load_pickle(xgb_model_url)

assert_is_not_none(xgb_tuned)

rf_tuned = load_pickle(rf_model_url)

assert_is_not_none(rf_tuned)
xgb_cv = pd.read_csv(xgb_results_url)

rf_cv = pd.read_csv(rf_results_url)

br_cv = pd.read_csv(br_results_url)



assert_is_not_none(xgb_cv)

assert_is_not_none(rf_cv)

assert_is_not_none(br_cv)

xgb_cv[xgb_cv['rank_test_score']<4]
rf_cv[rf_cv['rank_test_score']<4]
br_cv[br_cv['rank_test_score']<4]
def convert_df(df,name):

    """

    Function to convert the cv_result DataFrame for plotting.

    

    Inputs

    -------

    - df: DataFrame generated by GridSearchCV

    - name: name of the algorithm tuned with GridSearchCV

    

    Returns

    -------

    - converted DataFrame suitable for barplot

    """

    assert_is_not_none(df)

    assert_is_instance(df, pd.DataFrame)

    test_set = set(['rank_test_score','split0_test_score','split1_test_score','split2_test_score',

                   'split3_test_score','split4_test_score'])

    assert_true(test_set.issubset(set(df.columns)))



    

    t = df[df['rank_test_score']==1][['split0_test_score','split1_test_score','split2_test_score','split3_test_score','split4_test_score']]

    t = -t.T

    t=t.reset_index()

    t.columns=['algorithm','RMSLE_score']

    t.algorithm = name

    return t

clean = convert_df(xgb_cv,'XGBoost')

clean = clean.append(convert_df(rf_cv,'RandomForest'))

clean = clean.append(convert_df(br_cv,'BaggingRegressor'))

clean
plot_performance(clean, x='algorithm',y='RMSLE_score')
def plot_importance(model):

    """

    Fuction to plot variable importance from machine learning model.

    Note: the model requires to have "model.feature_importances_".

    

    Inputs

    -------

    - model: a model with a object ".feature_importances_"



    Returns

    -------

    - feature importance plot

    



    """

    assert_is_not_none(model.feature_importances_)



    feature_importance = model.feature_importances_

    # make importances relative to max importance

    feature_importance = 100.0 * (feature_importance / feature_importance.max())

    sorted_idx = np.argsort(feature_importance)

    pos = np.arange(sorted_idx.shape[0]) + .5

    plt.barh(pos, feature_importance[sorted_idx], align='center')

    plt.yticks(pos, X_train.columns[sorted_idx])

    plt.xlabel('Relative Importance')

    plt.title(f'Variable Importance {model.__class__.__name__}')

    plt.show()



plot_importance(xgb_tuned)

plot_importance(rf_tuned)
def plot_scores(model,X_train,y_train):

    """

    Functon to plot scores RMSE, MAE, R^2, RMSLE from model.

    

    Inputs

    -------

    - model: a trained model

    - X_train: X_train or X_test DataFrame/Series

    - y_train: y_train or y_test DataFrame/Series

    

    

    """



    assert(y_train.shape[0] == X_train.shape[0])

    

    print("Model parameters ", model.__class__.__name__)

    prediction_train = np.expm1((model.predict(X_train).clip(min=0)))

    y_train = np.expm1(y_train)

    print("Root Mean Squared Error: " + str(np.sqrt(mean_squared_error(y_train, prediction_train))))

    print("Mean Absolute Error: " + str(mean_absolute_error(y_train, prediction_train)))

    print("R^2 Coefficient of Determination: " + str(r2_score(y_train, prediction_train)))

    print('RMSLE:', np.sqrt(mean_squared_log_error(y_train, prediction_train)))

    
plot_scores(xgb_tuned,X_test,y_test)
plot_scores(rf_tuned,X_test,y_test)
algo_xgb = xgb_tuned.predict(X_test)

algo_rf = rf_tuned.predict(X_test)

mean =  (np.expm1(algo_xgb) + np.expm1(algo_rf))/2

mean_log = np.log1p(mean)

print("RMSLE of combined XGBoost and Random Forrest = ",rmsle(y_test,mean_log))


def residual_plot(y_test, y_predicted,main_title):

    """

    Function to plot the residual distribution calculated from y_test and y_predicted.

    

    Inputs

    -------

    - y_test: the log transformed target values

    - y_predicted: raw predictions from the algorithm

    - main_title: main title of the plot

    

    Returns

    -------

    - a plot containing 3 subplots: Scatter plot, Histogram and Q-Q plot

    

    

    """

    res =  np.expm1(y_test) - np.expm1(y_predicted)

    fig, axs = plt.subplots(1,3, figsize=(25, 5))

    ##D’Agostino’s K^2 Test

    stat, p = stats.normaltest(res)

    print("Performing a D’Agostino’s K^2 Test for a normal distribution of residuals.")

    print("H0 = The distribution is Gaussian.P value to reject H0 is p= 0.05 = 5%")

    print('Statistics=%.3f, p=%.5f' % (stat, p))

    # interpretation of the test

    alpha = 0.05

    if p > alpha:

        print("failed to reject H0 (H0 = The distribution is Gaussian).Can't accept any hypothesis. ")

    else:

        print('The distribution does not look Gaussian (H0 is rejected)')

    #End of D’Agostino’s K^2 Test

    #Begin of plotting

    plt.suptitle(f'{main_title} ; mean_residual =  {res.mean():.1f}, std= {res.std():.1f},  skew =  {res.skew():.4f} , kurtosis= {res.kurtosis():.2f}'  )

    axs[0].title.set_text('Scatter Plot y_test vs residuals(true-predicted)')

    axs[0].scatter(np.expm1(y_test),res)

    axs[1].title.set_text('Histogram of residuals')

    sns.distplot(res, fit=norm, ax = axs[1]);

    axs[2].title.set_text('Q-Q plot')

    sm.qqplot(res, stats.t, distargs=(4,),line = 's', ax=axs[2])

    plt.show()



residual_plot(y_test, xgb_tuned.predict(X_test) ,"Residual plots of XGBoost predictions")

residual_plot(y_test, rf_tuned.predict(X_test) ,"Residual plots of RandomForrest predictions")



print('---------Model evaluation on train set---------------')

plot_scores(xgb_tuned,X_train,y_train)

print('\n--------Model evaluation on test set---------------')

plot_scores(xgb_tuned,X_test,y_test)



print('\n---------Model evaluation on train set---------------')

plot_scores(rf_tuned,X_train,y_train)

print('\n--------Model evaluation on test set---------------')

plot_scores(rf_tuned,X_test,y_test)
target_rmsle = np.log(1.25)

target_rmsle