import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

from pandas import Timestamp

import xgboost

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.feature_selection import SelectKBest, chi2, f_regression, mutual_info_regression, VarianceThreshold, RFE

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, Ridge, Lasso, LogisticRegression

from mlxtend.feature_selection import SequentialFeatureSelector

#Dataset: https://www.kaggle.com/hmavrodiev/london-bike-sharing-dataset
data_df=pd.read_csv('..//input/london-bike-sharing-dataset/london_merged.csv')



#convert timestamp type to datetime

data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
data_df.head(10)
data_df.info()
data_df.describe()


#add column representing the hour of the timestamp

data_df.loc[:, 'hour'] = data_df['timestamp'].dt.hour



#add column representing the month of the timestamp

data_df.loc[:, 'month'] = data_df['timestamp'].dt.month



#add column representing the day of the week

data_df['day_of_week'] = data_df['timestamp'].dt.dayofweek



#add column indicating if it is work day or not

data_df['work_day'] = [row['is_holiday'] == 0 and row['is_weekend'] == 0 for i, row in data_df.iterrows()]

data_df.describe()
time_diffs = np.diff(data_df['timestamp'])

time_diffs = time_diffs / np.timedelta64(1, 'h')



print("Unique values: ", np.unique(time_diffs))



Counter(time_diffs)
display(min(data_df['timestamp']))

display(max(data_df['timestamp']))
n_days = 7

plt.figure(figsize=(10,10))

plt.plot(data_df['timestamp'][:n_days * 24],data_df['t1'][:n_days * 24], label="t1")

plt.plot(data_df['timestamp'][:n_days * 24],data_df['t2'][:n_days * 24], label="t2")



plt.legend()

plt.show()



print('From: {}\nTo: {}'.format(data_df['timestamp'][0], data_df['timestamp'][n_days * 24 - 1]))


fig, ax = plt.subplots(2, 1, figsize=(20,15))



ax[0].plot(data_df['timestamp'],data_df['t1'], label="t1")

ax[1].plot(data_df['timestamp'],data_df['t2'], label="t2")



ax[0].title.set_text('T1 Feature')

ax[1].title.set_text('T2 Feature')



ax[0].set_xlabel('Timestamp')

ax[0].set_ylabel(u'C°')



ax[1].set_xlabel('Timestamp')

ax[1].set_ylabel(u'C°')



plt.show()
#Analysis of the correlation of the temperatue in each year
data_2015 = data_df.loc[(data_df['timestamp'] > Timestamp('2015-01-01 0:0:0')) & (data_df['timestamp'] <= Timestamp('2015-12-31 23:59:59'))]

data_2016 = data_df.loc[(data_df['timestamp'] > Timestamp('2016-01-01 0:0:0')) & (data_df['timestamp'] <= Timestamp('2016-12-31 23:59:59'))]





fig = plt.figure(figsize=(20, 15))

grid = plt.GridSpec(3, 2)





top_ax = fig.add_subplot(grid[0, 0:])

top_ax.plot(data_2015['timestamp'], data_2015['t1'], label="t1 2015")

top_ax.plot(data_2016['timestamp'], data_2016['t1'], label="t1 2016")

top_ax.set_ylabel(u'C°')



top_ax.legend()





mid_ax = fig.add_subplot(grid[1, 0:], sharex = top_ax)

plt.setp(mid_ax.get_xticklabels(), visible=False)

mid_ax.plot(data_2015['timestamp'], data_2015['t2'], label="t2 2015")

mid_ax.plot(data_2016['timestamp'], data_2016['t2'], label="t2 2016")

mid_ax.invert_yaxis()

mid_ax.set_ylabel(u'C°')



mid_ax.legend()



left_ax = plt.subplot(grid[2, 0]);

left_ax.plot([i for i in range(0, len(data_2015))], data_2015['t2'], label="t2")

left_ax.plot([i for i in range(0, len(data_2016))], data_2016['t2'], label="t1")



right_ax = plt.subplot(grid[2, 1]);

right_ax.plot([i for i in range(0, len(data_2015))], data_2015['t2'], label="t2")

right_ax.plot([i for i in range(0, len(data_2016))], data_2016['t2'], label="t1")



top_ax.title.set_text('T1 and T2 over time')

left_ax.title.set_text('T1 Feature')

right_ax.title.set_text('T2 Feature')



left_ax.set_xlabel('Timestamp')

left_ax.set_ylabel(u'C°')

left_ax.legend()



right_ax.set_xlabel('Timestamp')

right_ax.set_ylabel(u'C°')

right_ax.legend()







display('Correlation between T1 and T2 over the years', data_df[['t1', 't2']].corr())



plt.show()
print(data_df['weather_code'].unique())



data_df.loc[data_df['weather_code'] == 26, 'weather_code'] = 5

data_df.loc[data_df['weather_code'] == 10, 'weather_code'] = 6

data_df.loc[data_df['weather_code'] == 94, 'weather_code'] = 8



def plot_sensors_hist(df, s_columns):



    fig = plt.figure(figsize=(14, 7))

    fig.subplots_adjust(wspace=0.4)



    

    for i, s in enumerate(s_columns):



        ax = plt.subplot(2, 3, i+1)

        

        vals = df[s]



        vals.hist()

        

        ax.set_ylabel(s)

        ax.set_xlabel('Time')



    plt.show()



        

plot_sensors_hist(data_df, ['t1', 't2', 'hum', 'wind_speed', 'weather_code', 'season'])









def plot_bar_cnt(df, s_columns, custom_bins):

    

    fig = plt.figure(figsize=(25, 10))

    fig.subplots_adjust(wspace=0.2)



    t_df = df.copy()

    for i, s in enumerate(s_columns):



        ax = plt.subplot(2, 3, i+1)



        custom_bins[i] = np.round(custom_bins[i], 1)

        

        t_df['bins'] = pd.cut(x=t_df[s], bins=custom_bins[i])

        

        t_df.groupby('bins').mean()['cnt'].plot.bar(rot=0)

        

        ax.set_ylabel('Mean of Nº of bike shares')

        ax.set_xlabel(s)



    plt.show()



        

custom_bins = [np.linspace(min(data_df['t1']), max(data_df['t1']), num=5),

        np.linspace(min(data_df['t2']), max(data_df['t2']), num=5),

        np.linspace(min(data_df['hum']), max(data_df['hum']), num=5),

        np.linspace(min(data_df['wind_speed']), max(data_df['wind_speed']), num=5),

        np.append(-1, np.linspace(min(data_df['weather_code']), max(data_df['weather_code']), num=len(data_df['weather_code'].unique()))),

        np.append(-1, np.linspace(min(data_df['season']), max(data_df['season']), num=len(data_df['season'].unique()))),

       ]



plot_bar_cnt(data_df, ['t1', 't2', 'hum', 'wind_speed', 'weather_code', 'season'], custom_bins)



def plot_hourly_cnt(df, label):

    df.groupby(df['hour']).mean()['cnt'].plot(label=label)

    

    

    

fig, ax = plt.subplots(figsize=(15,10))



plot_hourly_cnt(data_df.loc[(data_df['is_holiday'] == 0) & (data_df['is_weekend'] == 0), :], 'Work Days')

plot_hourly_cnt(data_df.loc[data_df['is_holiday'] == 1, :], 'Weekend')

plot_hourly_cnt(data_df.loc[data_df['is_weekend'] == 1, :], 'Holiday')



ax.set_ylabel('Nº of bike shares')

ax.title.set_text('Mean of the number of bikes shares along the day')

ax.set_xlabel('Hour of the day (h)')



plt.legend()

plt.show()

def plot_cnt(df, label, ax):

    #df.groupby(df['timestamp'].dt.month).mean()['cnt'].plot.bar(label=label, rot=0)



    df.boxplot(column='cnt', by='hour', ax=ax[0])

    df.boxplot(column='cnt', by='month', ax=ax[1])

    ax[0].title.set_text('Mean of the number of bikes shares hourly - ' + label)

    ax[1].title.set_text('Mean of the number of bikes shares monthly - ' + label)

    ax[0].set_ylabel('Nº of bike shares')

    ax[1].set_ylabel('Nº of bike shares')

    

    



fig, ax = plt.subplots(2, 3, figsize=(30, 15))



plot_cnt(data_df.loc[(data_df['is_holiday'] == 0) & (data_df['is_weekend'] == 0), :], 'Work Days', ax[:, 0])

plot_cnt(data_df.loc[data_df['is_holiday'] == 1, :], 'Weekend', ax[:, 1])

plot_cnt(data_df.loc[data_df['is_weekend'] == 1, :], 'Holiday', ax[:, 2])



plt.show()
#Plot the data features, in groups of 2
sns.pairplot(data_df[['cnt', 't1', 't2', 'hum', 'wind_speed', 'weather_code', 'is_holiday', 'is_weekend', 'season', 'hour', 'month']])

plt.show()


data_df[['cnt', 't1', 't2', 'hum', 'wind_speed', 'weather_code', 'is_holiday', 'is_weekend', 'season']].corr()['cnt']





plt.figure(figsize=(15,15))



sns.set(font_scale=1.1)



ax = sns.heatmap(

    data_df[['cnt', 't1', 't2', 'hum', 'wind_speed', 'weather_code', 'is_holiday', 'is_weekend', 'season']].corr(), 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(240, 240, l=35, n=20),

    square=True,

    annot=True, annot_kws={"size": 15}

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='center'

);



           

plt.show()

data_df.isnull().sum().sum()
X_train, X_test, y_train, y_test = train_test_split(data_df[data_df.columns.difference(['timestamp', 'cnt'])], data_df['cnt'], test_size = 0.2, random_state = 0)

y_true = data_df['cnt']
data_df.columns
#1. K best

#Regression: f_regression, mutual_info_regression, chi2 (only positive number)



selector = SelectKBest(f_regression, k=4)

selector = SelectKBest(mutual_info_regression, k=4)

X_train_new = selector.fit_transform(X_train, y_train)



relevant_features_f1 = X_train.columns[selector.get_support(indices=True)].values

print("Feature Selected: ", X_train.columns[selector.get_support(indices=True)].values)

#2. Variance Threshold



#Features with a training-set variance lower than this threshold will be removed. O means removing only constant features



thresholder = VarianceThreshold(threshold=15)

X_train_new_2 = thresholder.fit_transform(X_train)



relevant_features_f2 = X_train.columns[selector.get_support(indices=True)].values

print("Feature Selected: ", X_train.columns[thresholder.get_support(indices=True)].values)

#3. Correlation



entire_X_train = X_train.copy()

entire_X_train['cnt'] = y_train



cor = entire_X_train[['cnt', 't1', 't2', 'hum', 'wind_speed', 'weather_code', 'is_holiday', 'is_weekend', 'season']].corr()

cor_target = abs(cor["cnt"])



relevant_features_f3 = cor_target[cor_target>0.3]

print("Feature Selected: ", relevant_features_f3.index.values)
#1. Forward: Adding feature with better results, in each iteration. The Random Forest model will be used



feature_selector = SequentialFeatureSelector(LinearRegression(),

           k_features=4,

           forward=True,

           verbose=1,

           scoring='neg_mean_squared_error',

           cv=3)



mdl = feature_selector.fit(X_train, y_train)



relevant_features_w1 = X_train.columns[list(mdl.k_feature_idx_)]

print("Feature Selected: ", relevant_features_w1.values)

#2. Backward: Removing the feature with worse results, in each iteration. The Random Forest model will be used



feature_selector = SequentialFeatureSelector(LinearRegression(n_jobs=-1),

           k_features=5,

           forward=False,

           verbose=1,

           scoring='neg_mean_squared_error',

           cv=3)



mdl = feature_selector.fit(X_train, y_train)



relevant_features_w2 = X_train.columns[list(mdl.k_feature_idx_)]

print("Feature Selected: ", relevant_features_w2.values)
#3. RFE:



model = LinearRegression()

rfe = RFE(model, 5)

X_rfe = rfe.fit_transform(X_train, y_train)  



model.fit(X_rfe,y_train)



relevant_features_w3 = X_train.columns[rfe.support_]

#print(rfe.ranking_)

print("Feature Selected: ", relevant_features_w3.values)

#1. Lasso



#1.1 With cross validation to determine the best alpha



reg = LassoCV(cv = 3)

reg.fit(X_train, y_train)



coef = pd.Series(reg.coef_, index = X_train.columns)

relevant_features_l1 = [i for i in coef.index if coef[i] == 0.0]



print("Feature Selected: ", [i for i in coef.index if coef[i] == 0.0])







#1.2.1 Simple Lasso (without cv)



reg = Lasso(alpha = 7)

reg.fit(X_train, y_train)



coef = pd.Series(reg.coef_, index = X_train.columns)

relevant_features_l1



print("Feature Selected: ", [i for i in coef.index if coef[i] == 0.0])







#1.2.2 Lasso with Gradient Descent: only for classification



#2. Tree based: RF or XGBoost do not work for Regression

X_train = X_train[relevant_features_w1]

X_test = X_test[relevant_features_w1]
X_train.info()
def convert_to_categorical(df, column_names, y_true):

    df = pd.concat([X_train, X_test], sort=False)

    

    for col in column_names:

        if col in X_train.columns:

            df[col] = df[col].astype('category')

            df = pd.concat([df, pd.get_dummies(df[col], prefix=col, drop_first=True)], axis=1, sort=False)

            df.drop([col], axis=1, inplace=True)

    

    return train_test_split(df, y_true, test_size = 0.2, random_state = 0)





#cat_columns = ['weather_code']

#X_train, X_test, y_train, y_test = convert_to_categorical(data_df, cat_columns, y_true)

continuous_columns = ['t1', 'hum']



sc = StandardScaler()

X_train[continuous_columns] = sc.fit_transform(X_train[continuous_columns])

X_test[continuous_columns] = sc.transform(X_test[continuous_columns])
X_train.head(10)
#XGBOOST



parameters = {

    'max_depth': [1, 5],

    'n_estimators': [400, 800],

    'learning_rate': [0.01, 0.05]

}

#'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 800





#Cross Validation

grid_search = GridSearchCV(

    estimator=xgboost.XGBRegressor(objective ='reg:squarederror'),

    param_grid=parameters,

    n_jobs = 10,

    cv = 10

)





grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

best_model = grid_search.best_estimator_







#sgb = xgboost.XGBRegressor(objective ='reg:squarederror', learning_rate= 0.01, max_dept= 1, n_estimators= 800)

#best_model = sgb.fit(X_train, y_train)
#Results



results = best_model.predict(X_test)



print("MSE: ", np.round(mean_squared_error(y_test, results), 2))

print("MAE: ", np.round(mean_absolute_error(y_test, results), 2))
