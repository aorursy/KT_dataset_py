# @title Set everything up.



!pip install pandas numpy xgboost python-dateutil kaggle seaborn



%matplotlib inline



import pandas as pd

from pandas import Grouper



import matplotlib.pyplot as plt

from matplotlib import dates



import numpy as np



from datetime import date

from dateutil.relativedelta import relativedelta

import datetime



import seaborn as sns



import xgboost as xgb



import lightgbm as lgb



from sklearn import linear_model

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import TimeSeriesSplit

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.metrics import mean_squared_error



from tqdm import tqdm_notebook as tqdm



import missingno as msno



from copy import deepcopy
# @title Load the data

df_train = pd.read_csv("../input/electricity/train_electricity.csv")

df_test = pd.read_csv("../input/electricity/test_electricity.csv")

n_train, n_test = len(df_train), len(df_test)



print("Training dataset has", n_train, "entries.")

print("Testing dataset has", n_test, "entries.")



print(f"\n\t{'Column':20s} | {'Type':8s} | {'Min':12s} | {'Max':12s}\n")

for col_name in df_train.columns:

    col = df_train[col_name]

    print(f"\t{col_name:20s} | {str(col.dtype):8s} | {col.min():12.1f} | {col.max():12.1f}")
# @title Adding datetime features (as in the baseline example)



def add_datetime_features(df):

    features = ["Year", "Month", "Week", "Day", "Dayofyear", "Dayofweek",

                "Hour", "Minute"]#, "Is_year_end", "Is_year_start", "Is_month_end", "Is_month_start"]

    one_hot_features = []#["Month", "Dayofweek"]



    datetime = pd.to_datetime(df.Date * (10 ** 9))



    df['Datetime'] = datetime  # We won't use this for training, but we'll remove it later



    for feature in features:

        new_column = getattr(datetime.dt, feature.lower())

        if feature in one_hot_features:

            df = pd.concat([df, pd.get_dummies(new_column, prefix=feature)], axis=1)

        df[feature] = new_column

    return df



df_train = add_datetime_features(df_train)

df_test = add_datetime_features(df_test)



df_train = df_train.sort_values(by=["Datetime"], ascending=True)

df_test = df_test.sort_values(by=["Datetime"], ascending=True)
# @title Exploring distances between samples

def plot_distances(df, df_desc):

  dates = df["Date"].values

  intervals = dates[1:] - dates[:-1]

  intervals = intervals.astype('timedelta64[m]').astype(int)

  plt.plot(intervals, c ="r")

  plt.title("Distances between samples (minutes) - {}".format(df_desc))

  plt.xlabel("# of measurement")

  plt.ylabel("minutes")

  plt.show()

  

plot_distances(df_train, "training set")

plot_distances(df_test, "testing set")
# @title Dropping Outliers from training set

df_train.plot('Datetime', 'Consumption_MW', title='Consumption MW', figsize=(25, 5))

print("Before dropping")

df_train.drop([85008,85009,146999,159698,161540,161541],inplace=True)

df_train.plot('Datetime', 'Consumption_MW', title='Consumption MW', figsize=(25, 5))

print("After dropping")
# @title



from pandas import Grouper

from collections import defaultdict



markers=['x', 'o', 's', 'v', '*', '>']



groups = df_train[["Consumption_MW", "Datetime"]].groupby(Grouper(key='Datetime', freq="M"))

rs = defaultdict(lambda: [])

theta = [i*2*np.pi/12 for i in range(12)]

ax = plt.subplot(111, projection='polar')

ax.set_title("Average electricity consumption across months for different years", va='bottom')



for name, group in groups:

  rs[name.year].append(np.mean(group[["Consumption_MW"]].values.flatten()) - 5000)



leg = []

  

for i, (year, r) in enumerate(rs.items()):

  ax.plot(theta[:len(r)] + [0], r + [r[0]], marker=markers[i%len(markers)])

  leg.append(year)

    

ax.set_yticks([0, 1000, 2000])

ax.set_yticklabels(["5000 MW", "6000 MW", "7000 MW"])

ax.set_xticks(theta)

months = ["January", "February", "March", "April", "May", "June",

          "July", "August", "September", "October", "November", "December"]

ax.set_xticklabels(months)

ax.grid(True)

ax.legend(leg, loc="center")



plt.gcf().set_size_inches(15, 15)

plt.show()
# @title



from pandas import Grouper

from collections import defaultdict

from copy import deepcopy



df_temp = deepcopy(df_train[["Consumption_MW", "Datetime"]])

df_temp["Datetime"] = df_temp["Datetime"].apply(lambda x: datetime.datetime(2004, x.month, x.day))

groups = df_temp.groupby(Grouper(key='Datetime', freq="D"))

rs = defaultdict(lambda: [])

theta = [i*2*np.pi/31 for i in range(31)]

ax = plt.subplot(111, projection='polar')

ax.set_title("Average electricity consumption across hours for different months", va='bottom')



for name, group in groups:

  rs[name.month].append(np.mean(group[["Consumption_MW"]].values.flatten()) - 5000)



leg = []

  

for i, (month, r) in enumerate(rs.items()):

  ax.plot(theta[:len(r)], r, marker=markers[i%len(markers)])

  leg.append(months[month-1])

    

ax.set_yticks([0, 1000, 2000])

ax.set_yticklabels(["5000 MW", "6000 MW", "7000 MW"])

ax.set_xticks(theta)

ax.set_xticklabels(range(1, 33))

ax.grid(True)

ax.legend(leg, loc="best")



plt.gcf().set_size_inches(15, 15)

plt.show()
#@title



from pandas import Grouper

from collections import defaultdict

from copy import deepcopy



days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]



df_temp = deepcopy(df_train[["Consumption_MW", "Datetime"]])

df_temp["Datetime"] = df_temp["Datetime"].apply(lambda x: datetime.datetime(2004, 1, 1+days.index(x.strftime("%A")), x.hour))

groups = df_temp.groupby(Grouper(key='Datetime', freq="h"))

rs = defaultdict(lambda: [])

theta = [i*2*np.pi/24 for i in range(24)]

ax = plt.subplot(111, projection='polar')

ax.set_title("Average electricity consumption across days for different months", va='bottom')



for i, (name, group) in enumerate(groups):

  rs[days[name.day-1]].append(np.mean(group[["Consumption_MW"]].values.flatten()) - 5000)



leg = []

  

for i, (day, r) in enumerate(rs.items()):

  ax.plot(theta[:len(r)] + [0], r + [r[0]], marker=markers[i%len(markers)])

  leg.append(day)

    

ax.set_yticks([0, 1000, 2000])

ax.set_yticklabels(["5000 MW", "6000 MW", "7000 MW"])

ax.set_xticks(theta)

ax.set_xticklabels(range(0, 24))

ax.grid(True)

ax.legend(leg, loc="best")



plt.gcf().set_size_inches(15, 15)

plt.show()
# @title Exploring and transforming features

def plot_features(df, cols, df_desc):

  prop_cycle = plt.rcParams['axes.prop_cycle']

  colors = prop_cycle.by_key()['color']

  c = len(cols)

  fig, axes = plt.subplots(nrows=c, ncols=1)

  fig.set_size_inches(20, 10)

  fig.suptitle("Features - {}".format(df_desc))

  for i, col in enumerate(cols):

    if col != cols[-1]:

        axes[i].set_xticks([])

        axes[i].axes.get_xaxis().set_visible(False)

    df.plot("Datetime", col, ax=axes[i], c=colors[i], legend=False)

    axes[i].set_title(col, loc='left')

  plt.show()



cols = ['Coal_MW', 'Gas_MW', 'Hidroelectric_MW', 'Nuclear_MW',

        'Wind_MW', 'Solar_MW', 'Biomass_MW', 'Production_MW']

plot_features(df_train, cols, "Training dataset")
# @title

plot_features(df_test, cols, "Testing dataset")
# @title

'''

def wiggle_plot(df, cols, df_desc):

  plt.stackplot(range(len(df)), df[cols].T, baseline='wiggle')

  plt.title("Features {}".format(df_desc))

  plt.gcf().set_size_inches(20, 10)

  plt.legend(cols)

  plt.show()

  

wiggle_plot(df_train, cols, "Training dataset")

wiggle_plot(df_test, cols, "Testing dataset")

'''
# @title 

def stacked_bar(df, cols, l, df_desc):

  leg = []

  last_means = np.zeros(l)

  xticks = []

  xticklabels = []



  for c in cols:

    if c == "Production_MW":

      continue

    groups = df[[c, "Datetime"]].groupby(Grouper(key='Datetime', freq="M"))



    m, s = [], []

    for i, (name, group) in enumerate(groups):

      m.append(np.mean(group[[c]].values.flatten()))

      s.append(np.std(group[[c]].values.flatten()))

      if c == "Gas_MW":

        if (name.month == 1):

          xticklabels.append(name.year)

          xticks.append(i)



    plt.bar(range(l), m, 0.5, bottom=last_means)

    last_means += m



    leg.append(c)





  plt.ylabel('Production_MW')

  plt.title('Production over dates - {}'.format(df_desc))

  plt.gca().set_xticks(xticks)

  plt.gca().set_xticklabels(xticklabels)

  plt.gcf().set_size_inches(25, 10)

  plt.legend(leg)

  plt.show()

  

stacked_bar(df_train, cols, 97, "Training dataset")
#@title

stacked_bar(df_test, cols, 13, "Testing dataset")
df_train = df_train.drop(df_train.index[[0,220000]])
# @title Adding cyclical datetime features



def transform_time_features(df):

  df['Year'] = df['Datetime'].apply(lambda x: x.year - 2009)

  df['Month_sin'] = df['Datetime'].apply(lambda x: np.sin(2*np.pi*x.month / 12.0))

  df['Month_cos'] = df['Datetime'].apply(lambda x: np.cos(2*np.pi*x.month / 12.0))

  df['Day_sin'] = df['Datetime'].apply(lambda x: np.sin(2*np.pi*x.day /

                      (date(x.year-1 if x.month == 1 else x.year, 12 if x.month == 1 else x.month - 1, 1) - date(x.year, x.month, 1)).days))

  df['Day_cos'] = df['Datetime'].apply(lambda x: np.cos(2*np.pi*x.day /

                      (date(x.year-1 if x.month == 1 else x.year, 12 if x.month == 1 else x.month - 1, 1) - date(x.year, x.month, 1)).days))

  df['Hour_sin'] = df['Datetime'].apply(lambda x: np.sin(2*np.pi*(x.hour*3600 + x.minute*60 + x.second) / (24*60*60)))

  df['Hour_cos'] = df['Datetime'].apply(lambda x: np.cos(2*np.pi*(x.hour*3600 + x.minute*60 + x.second) / (24*60*60)))



transform_time_features(df_train)

transform_time_features(df_test)

  

def radial_plot(df, sin_col, cos_col, df_desc, n=5000):

  np.arctan2(df[sin_col], df[cos_col])

  r = range(n)

  plt.scatter(r*df[sin_col][:n], r*df[cos_col][:n], alpha=0.75, s=1, c=r)

  plt.gcf().set_size_inches(5, 5)

  plt.title("Transformed cyclical features {}, {}, ({})".format(sin_col, cos_col, df_desc))

  plt.xticks([])

  plt.yticks([])

  plt.show()

  

radial_plot(df_train, 'Hour_sin', 'Hour_cos', 'Training set')

radial_plot(df_train, 'Day_sin', 'Day_cos', 'Training set', n=10000)

radial_plot(df_train, 'Month_sin', 'Month_cos', 'Training set', n=100000)
# @title Adding public holiday/weekends information



def freeday_from_date(date):

    dt_object = datetime.datetime.fromtimestamp(date)

    

    #list of holidays in Romania

    if str(dt_object.date()) in ["2018-01-01","2018-01-02","2018-01-24","2018-04-06","2018-04-08","2018-04-09","2018-05-01","2018-05-27","2018-05-28","2018-06-01","2018-08-15","2018-11-30","2018-12-01","2018-12-25","2018-12-26","2017-01-01","2017-01-02","2017-01-24","2017-04-16","2017-04-17","2017-05-01","2017-06-01","2017-06-02","2017-06-04","2017-06-05","2017-08-15","2017-11-30","2017-12-01","2017-12-25","2017-12-26","2016-01-01","2016-01-02","2016-01-24","2016-05-01","2016-05-02","2016-05-28","2016-06-19","2016-06-20","2016-08-15","2016-11-30","2016-12-01","2016-12-02","2016-12-25","2016-12-26","2015-01-01","2015-01-02","2015-01-24","2015-04-12","2015-04-13","2015-05-01","2015-05-31","2015-06-01","2015-08-15","2015-11-30","2015-12-01","2015-12-25","2015-12-26","2014-01-01","2014-01-02","2014-04-20","2014-04-21","2014-05-01","2014-06-08","2014-06-09","2014-08-15","2014-11-30","2014-12-01","2014-12-25","2014-12-26","2013-01-01","2013-01-02","2013-05-01","2013-05-02","2013-05-03","2013-05-05","2013-05-06","2013-06-23","2013-05-24","2013-08-15","2013-11-30","2013-12-01","2013-12-25","2013-12-26","2012-01-01","2012-01-02","2012-04-15","2012-04-16","2012-05-01","2012-06-03","2012-06-04","2012-08-15","2012-11-30","2012-12-01","2012-12-25","2012-12-26","2011-01-01","2011-01-02","2011-04-24","2011-04-25","2011-05-01","2011-06-12","2011-06-13","2011-08-15","2011-12-01","2011-12-25","2011-12-26","2010-01-01","2010-01-02","2010-04-04","2010-04-05","2010-05-01","2010-05-23","2010-05-24","2010-08-15","2010-12-01","2010-12-25","2010-12-26"]:

      return 1

    else:

      day = dt_object.weekday()    

      if day in [5,6]:

        return 1

      else:

        return 0



df_train["Holiday"]= df_train.Date.apply(freeday_from_date)

df_test["Holiday"]= df_test.Date.apply(freeday_from_date)
# @title Adding temperature features

def add_date_YMD(date):

    dt_object = datetime.datetime.fromtimestamp(date)

    return str(dt_object.date())



df_train["Date_YMD"]=df_train["Date"].apply(add_date_YMD)

df_test["Date_YMD"]=df_test["Date"].apply(add_date_YMD)





df_temperature = pd.read_csv("../input/temperature-romania/date_temperature.csv")

df_temperature.rename(columns={'Date': 'Date_YMD'}, inplace=True)



df_train = pd.merge(left=df_train,right=df_temperature, how='left', on="Date_YMD")

df_test = pd.merge(left=df_test,right=df_temperature, how='left', on="Date_YMD")

df_train.drop(columns=["Date_YMD"], inplace=True)

df_test.drop(columns=["Date_YMD"], inplace=True)





print("Missing values")

msno.matrix(df=df_train, figsize=(20,14), color=(0.5,0,0))

msno.matrix(df=df_test, figsize=(20,14), color=(0.5,0,0))
# @title

# mean Romanian temperature

df_train = df_train.replace(np.nan, 33, regex=True)

df_test = df_test.replace(np.nan, 33, regex=True)



#compare temp and consumption

plt.plot(df_train['Consumption_MW'])

plt.plot(df_train['Temp']*125)

plt.gcf().set_size_inches(25, 10)

plt.legend(['Consumption_MW', 'Temperature'])
#@title Correlation map

corrmat = df_train.corr()

plt.subplots(figsize=(20,13))

sns.heatmap(corrmat, vmax=0.9, square=True)
'''

# @title Setting up helper functions

def plot_results(Y_train, Y_pred, title):

  plt.figure(figsize=(20,5))

  plt.plot(Y_train)

  plt.plot(Y_pred)

  plt.title(title)

  plt.show()



def time_series_cv(X_train, Y_train, model, k=4):

  tscv = TimeSeriesSplit(n_splits=k)

  scores = []

  for i, (train_index, test_index) in enumerate(tscv.split(X_train)):

    X1, X2 = X_train[train_index, :], X_train[test_index, :]

    Y1, Y2 = Y_train[train_index], Y_train[test_index]

    m = deepcopy(model)

    m.fit(X1, Y1)

    Y_pred = m.predict(X2)

    scores.append(np.sqrt(mean_squared_error(Y2, Y_pred)))

    plot_results(Y2, Y_pred, "Fold #{}, rmse={}".format(i+1, scores[-1]))

  return scores



X_train = df_train.drop(["Date", "Datetime", "Consumption_MW"], axis=1).values

Y_train = df_train[["Consumption_MW"]].values

X_test = df_test.drop(["Date", "Datetime"], axis=1).values

'''
'''

# @title Stacking Averaged Models



# source: https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, base_models, meta_model, n_folds=5):

        self.base_models = base_models

        self.meta_model = meta_model

        self.n_folds = n_folds

   

    # We again fit the data on clones of the original models

    def fit(self, X, y):

        self.base_models_ = [list() for x in self.base_models]

        self.meta_model_ = clone(self.meta_model)

        kfold = TimeSeriesSplit(n_splits=self.n_folds)

        

        # Train cloned base models then create out-of-fold predictions

        # that are needed to train the cloned meta-model

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):

            for train_index, holdout_index in kfold.split(X, y):

                instance = clone(model)

                self.base_models_[i].append(instance)

                instance.fit(X[train_index], y[train_index])

                y_pred = instance.predict(X[holdout_index])

                out_of_fold_predictions[holdout_index, i] = y_pred

                

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature

        self.meta_model_.fit(out_of_fold_predictions, y)

        return self

   

    #Do the predictions of all base models on the test data and use the averaged predictions as 

    #meta-features for the final prediction which is done by the meta-model

    def predict(self, X):

        meta_features = np.column_stack([

            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)

            for base_models in self.base_models_ ])

        return self.meta_model_.predict(meta_features)

'''
'''

lasso = make_pipeline(RobustScaler(), Lasso())

lasso_scores=time_series_cv(X_train, Y_train, lasso)

'''
'''

ENet = make_pipeline(RobustScaler(), ElasticNet())

enet_scores=time_series_cv(X_train, Y_train, ENet)

'''
'''

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

GBoost_scores=time_series_cv(X_train, Y_train.flatten(), GBoost)

'''
'''

model_xgb = xgb.XGBRegressor(n_estimators=1000, max_depth=8, learning_rate=0.1, min_child_weight=10,

                         subsample=0.75, colsample_bylevel=1, reg_lambda=10, gamma=5,

                         n_jobs=-1, silent=1)

xgb_scores=time_series_cv(X_train, Y_train, model_xgb)

'''
'''

print("Lasso scores: {}".format(lasso_scores))

print("ENet scores: {}".format(enet_scores))

print("GBoost scores: {}".format(GBoost_scores))

print("xgb scores: {}".format(xgb_scores))

'''
'''

# stacked_averaged_models = StackingAveragedModels(base_models = (model_xgb, GBoost, model_xgb),

#                                  meta_model = lasso)

stacked_averaged_models = StackingAveragedModels(base_models = (lasso, ENet, ENet),

                                 meta_model = lasso)

stacked_scores=time_series_cv(X_train, Y_train, stacked_averaged_models)

'''
'''

N_ESTIMATORS = [400, 600]

MAX_DEPTH = [7, 8]

LEARNING_RATE = [0.01, 0.1]

RANDOM_STATE = [1, 42]



Y_pred = np.zeros(X_test.shape[0])

nmodels = 0



from itertools import product



for n_estimators, max_depth, learning_rate, random_state in product(N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE, RANDOM_STATE):

  print(n_estimators, max_depth, learning_rate, random_state)

  bst = xgb.XGBRegressor(n_estimators=n_estimators,

                         max_depth=max_depth,

                         learning_rate=learning_rate,

                         min_child_weight=10,

                         subsample=0.75,

                         colsample_bylevel=1,

                         reg_lambda=10,

                         gamma=5,

                         n_jobs=-1,

                         random_state=random_state

                        )

  bst = bst.fit(X_train, Y_train, eval_metric='rmse')

  Y_pred += bst.predict(X_test)

  nmodels += 1



Y_pred /= nmodels

'''
'''

df_test["Consumption_MW"] = Y_pred

df_test[["Date", "Consumption_MW"]].to_csv("submission.csv", index=False)

'''