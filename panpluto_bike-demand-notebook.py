import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from sklearn.metrics import mean_squared_log_error
from matplotlib.ticker import FuncFormatter 
from sklearn.preprocessing import OneHotEncoder


diro = 'C:\\Users\\Lenovo\\PycharmProjects\\Kaggle\\Project6_BikeDemand\\data\\'
diro = '../input/'
train = pd.read_csv(diro + 'train.csv', parse_dates=True)
test = pd.read_csv(diro + 'test.csv')
samsub = pd.read_csv(diro + 'sampleSubmission.csv')

print('Import completed')
train.head()
test.head()
train.datetime = pd.to_datetime(train.datetime)
test.datetime = pd.to_datetime(test.datetime)
train.dtypes
print('Number of rows \n\ttrain:{}\n\ttest:{}\nNumber of columns\n\ttrain:{}\n\ttest:{}'
      .format(train.shape[0], test.shape[0], train.shape[1], test.shape[1]))
nans = pd.concat([train.isnull().sum(), test.isnull().sum()], axis=1, sort=True)
nans.columns=['Train', 'Test']
print('Amount of Null values:\n')
print(nans)
print('\nColumns which are not in Test data: {}'.format(list(nans[nans['Test'] != 0].index)))
#train.loc[:, common].describe().loc[['mean', 'min', 'max'], :]
common = list(set(train.columns).intersection(set(test.columns)))
train = train.loc[: , common + ['count']]
test = test.loc[: , common]
# convert datetime to month and day; month; hour
for table in [train, test]:
    ## create new fields
    table['year'] = pd.DatetimeIndex(table.datetime).year
    table['month'] = pd.DatetimeIndex(table.datetime).month
    table['day'] = pd.DatetimeIndex(table.datetime).day
    table['hour'] = pd.DatetimeIndex(table.datetime).hour
    table['dow'] = pd.DatetimeIndex(table.datetime).dayofweek
    table['year-month'] = train.datetime.dt.to_period('M')

train.drop('datetime', axis=1, inplace=True)
# Prepare data
count = train.groupby(['year-month'])['count'].sum().values
temp = train.groupby(['year-month'])['atemp'].mean().values
year_month = train.groupby(['year-month'])['count'].sum().index
#time = time.astype('O')

fig, ax1 = plt.subplots()
train.groupby(['year-month'])['count'].sum().plot(ax=ax1, color='k')
ax1.set_xlabel('Year - Month')
#Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('Total of rentals', color='k')
ax1.tick_params('y', color='k')
ax1.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
ax1.set_title('Total Rentals/Temperature per Year and Month')

ax2 = ax1.twinx()
train.groupby(['year-month'])['atemp'].mean().plot(ax=ax2, color='b')
ax2.set_ylabel('temperature', color='b')
ax2.tick_params('y', colors='b')

fig.tight_layout()
fig, ax = plt.subplots()
train.groupby(['hour'])['count'].sum().plot(color='k')
ax.set_ylabel('Total of rentals', color='k')
ax.tick_params('y', colors='k')
ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
ax.set_title('Total rentals per hour')
def weather_descr(nr):
    if weather == 1:
         return 'Clear, Few clouds, \nPartly cloudy, Partly cloudy'
    elif weather == 2:
	    return 'Mist + Cloudy, Mist + Broken clouds, \nMist + Few clouds, Mist '
    elif weather == 3:
	    return 'Light Snow, Light Rain + \nThunderstorm + Scattered clouds, \nLight Rain + Scattered clouds'
    else:
        return 'Heavy Rain + Ice Pallets + \nThunderstorm + Mist, Snow + Fog'
# First we want to see how many hours in each day have been assigned to different weather conditions
data = train.groupby(['weather'])['count'].count()
x = data.index
height = data.values

fig, ax = plt.subplots()
sns.barplot(x, height)
fig, ax = plt.subplots(1, 2, sharey=True)   

x = [1, 2, 3, 4]

for wd in [1, 0]:
    data = train.query('workingday == @wd')
    if wd == 0:
        height = np.concatenate((data.groupby(['weather'])['count'].sum().values, np.array([0])))
    else:
        height = data.groupby(['weather'])['count'].sum().values
    ax[wd].bar(x, height, width=0.8)
    title = lambda x: 'Working Days' if x == 1 else 'Week End'
    ax[wd].set_title(title(wd))
    ax[wd].set_xlabel('Weather condition', color='k')
ax[0].set_ylabel('Total of rentals', color='k')
ax[0].get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
data = train.groupby(['dow'])['count'].sum().values

fig, ax = plt.subplots()
sns.barplot(days, data)
ax.set_ylabel('Total of rentals', color='k')
ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
for tick in ax.get_xticklabels():
    tick.set_rotation(45)
fig, ax = plt.subplots(2, 2,figsize=(8, 4), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.5)
weather = 1

for i in range(2):
    for j in range(2):
        
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        data = train.query('weather == @weather').groupby(['dow'])['count'].sum()
        
        ax[i, j].bar(days, data.values)
        ax[i, j].set_title(weather_descr(weather))
        ax[i, 0].set_ylabel('Total of rentals', color='k')
        ax[0, 0].get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
        
        for tick in ax[1, j].get_xticklabels():
            tick.set_rotation(45)
        weather += 1
#  1 = spring, 2 = summer, 3 = fall, 4 = winter 

fig, ax = plt.subplots(figsize=(10, 6))
for season in train.season.unique():
    sns.kdeplot(train.query('season == @season')['atemp'], legend=False)
ax.annotate('Spring', xy=(12, 0.07), xycoords='data',bbox=dict(boxstyle='round', fc='none', ec='grey'), 
            xytext=(40, 40), textcoords='offset points', ha='center', arrowprops=dict(arrowstyle='->'))

ax.annotate('Summer', xy=(26, 0.08), xycoords='data',bbox=dict(boxstyle='round', fc='none', ec='grey'), 
            xytext=(40, 40), textcoords='offset points', ha='center', arrowprops=dict(arrowstyle='->'))

ax.annotate('Fall', xy=(35, 0.08), xycoords='data',bbox=dict(boxstyle='round', fc='none', ec='grey'), 
            xytext=(40, 40), textcoords='offset points', ha='center', arrowprops=dict(arrowstyle='->'))

ax.annotate('Winter', xy=(20, 0.065), xycoords='data',bbox=dict(boxstyle='round', fc='none', ec='grey'), 
            xytext=(-20, 20), textcoords='offset points', ha='center', arrowprops=dict(arrowstyle='->'))
for pair1, pair2 in [('atemp', 'windspeed'), ('atemp', 'humidity'), ('windspeed', 'humidity')]:
    data = train.loc[:, [pair1, pair2]]
    sns.jointplot(pair1, pair2, data, kind='hex')
# Set style of charts Allstyles:  print(plt.style.available)
plt.style.use('seaborn-ticks')

# Function to ammend plot
def format_plot(ax, title, x_label, y_label, ticks='Y', x_labels=None, y_labels=None):
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    if ticks == 'Y':
        ax.set_xticks(np.arange(20))
        ax.set_xticklabels(x_labels)

        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        ax.set_yticks(np.arange(20))
        ax.set_yticklabels(y_labels)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
# Prepare function for data    
def prepare_data(dim1, dim2, season_y_n='N', ticks='N', season=None):
    if season_y_n != 'N':
        if season == 'Spring':
            data = train.query('season == 1').loc[:, [dim1, dim2, 'count']].copy()
        elif season == 'Summer':
            data = train.query('season == 2').loc[:, [dim1, dim2, 'count']].copy()
        elif season == 'Fall':
            data = train.query('season == 3').loc[:, [dim1, dim2, 'count']].copy()
        elif season == 'Winter':
            data = train.query('season == 4').loc[:, [dim1, dim2, 'count']].copy()
        else:
            data = train.loc[:, [dim1, dim2, 'count']].copy()
    else:
        data = train.loc[:, [dim1, dim2, 'count']].copy()
    
    dim1_cat = pd.cut(data[dim1], 20).cat.categories
    data[dim1] = pd.cut(data[dim1], 20, labels=np.arange(20))
    dim2_cat = pd.cut(data[dim2], 20).cat.categories
    data[dim2] = pd.cut(data[dim2], 20, labels=np.arange(20))
    
    data = data.groupby([dim1, dim2])['count'].sum().reset_index()
    x = data[dim1]
    y = data[dim2]
    colors = data['count']
    
    if ticks == 'Y':
        return x, y, colors, dim1_cat, dim2_cat
    else: 
        return x, y, colors

# Create plots for each pair in meteorical 
for dim1, dim2, color in [('atemp', 'windspeed', 'Greens'), ('atemp', 'humidity', 'Reds'), ('windspeed', 'humidity', 'Blues')]:
    # set up ploot grid
    fig = plt.figure(figsize=(16, 8))
    grid = plt.GridSpec(2, 4, wspace=0.4, hspace=0.4)

    # create grid of plots
    main_ax = fig.add_subplot(grid[:2, :2])
    spring_ax = fig.add_subplot(grid[0, 2])
    summer_ax = fig.add_subplot(grid[0, 3])
    fall_ax = fig.add_subplot(grid[1, 2])
    winter_ax = fig.add_subplot(grid[1, 3])

    # First Chart
    x, y, colors, atemp_cat, wind_cat = prepare_data(dim1, dim2, ticks='Y')
    main_ax.scatter(x, y, c=colors, alpha=0.6, cmap=color, s=300)
    format_plot(main_ax, 'All Year', dim1, dim2, 'Y', atemp_cat, wind_cat)
    # Seasons charts
    for ax, season in [(spring_ax, 'Spring'), (summer_ax, 'Summer'), (fall_ax, 'Fall'), (winter_ax, 'Winter')]:
        x, y, colors = prepare_data(dim1, dim2, season_y_n='Y', season=season)
        ax.scatter(x, y, c=colors, alpha=0.6, cmap=color, s=80)
        format_plot(ax, season, dim1, dim2, 'N')
# Select columns
columns = ['year','month', 'hour', 'dow', 'weather', 'atemp', 'humidity', 'windspeed', 'day']
train = train.loc[:, columns + ['count']]
test = test.loc[:, columns]
# Convert weather and DOW to 0/1
enc = OneHotEncoder()
tables = [train, test]
for i, table in enumerate(tables):
    weather = pd.DataFrame(enc.fit_transform(table.loc[:, ['weather']]).toarray(), columns=['Weather' + str(k) for k in range(1, 5)])
    dow = pd.DataFrame(enc.fit_transform(table.loc[:, ['dow']]).toarray(), columns=['dow' + str(k) for k in range(7)]) 
    table.drop(['weather', 'dow'], axis=1, inplace=True)
    tables[i] = pd.concat((table, dow), axis=1)
train, test = tables
print('encoding complete')
# split the train set into train_test and train_train subsets for validation of my base prediction. 
# the split needs to be made like: first 15 days of the month for train and rest 5 into test to 
# make laboratory conditions of real test set. 

train_train = train.query('day <= 15')
train_test = train.query('day > 15')

## build "prediction table":
y_pred = pd.DataFrame(round(train_train.groupby(['year', 'month', 'hour'])['count'].mean()))
y_pred.columns = ['count_pred']

## cutt train_test to show only valid coulumns
train_test = train_test.loc[:, ['year', 'month', 'hour', 'count']]

## join train_test with y_pred
prediction = pd.merge(train_test, y_pred, left_on=['year', 'month', 'hour'], right_index=True)

## check accuracy
train_error = mean_squared_log_error(prediction['count'], prediction['count_pred'])

# reproduce it on Test set
X_train = pd.DataFrame(round(train.groupby(['year', 'month', 'hour'])['count'].mean()))
Y = test.loc[:, ['datetime', 'year', 'month', 'hour']]
y_pred = pd.merge(Y, X_train, left_on=['year', 'month', 'hour'], right_index=True)
y_pred.drop(['year', 'month', 'hour'], axis=1, inplace=True)
y_pred['count'] = y_pred['count'].apply(lambda x: int(x))
y_pred = y_pred.sort_values(by='datetime')
#y_pred.to_csv(diro + 'Base prediction.csv', index=False)

kaggle_error = 70.87
print('Train score: {}\nKaggle score: {}'.format(round(train_error*100, 2), kaggle_error))
from sklearn.model_selection import cross_validate, ShuffleSplit, KFold

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge

from xgboost import XGBRegressor



y = train['count']
X = train.drop(['day', 'count'], axis=1)

ML = [LinearRegression(),
      LogisticRegression(),
      SVC(),
      RandomForestRegressor(),
     
     ElasticNet(),
     Lasso(),
     BayesianRidge(),
     LassoLarsIC(),
     GradientBoostingRegressor(),
     KernelRidge(),
     
     XGBRegressor()]

ml_results = pd.DataFrame(columns=['model', 'test_score','train_score', 'time', 'parameters'])
cv_split = ShuffleSplit(n_splits=2, test_size=.3, train_size=.7, random_state=8)
cv_split = KFold(n_splits=3, shuffle=True, random_state=0)


for i, algo in enumerate(ML):
    ml_results.loc[i, 'model'] = algo.__class__.__name__
    cv_results = cross_validate(algo, X, y, cv=cv_split, scoring="neg_mean_squared_error")
    ml_results.loc[i, 'test_score'] = np.sqrt(-cv_results['test_score'].mean())
    ml_results.loc[i, 'train_score'] = np.sqrt(-cv_results['train_score'].mean())
    ml_results.loc[i, 'time'] = cv_results['fit_time'].mean()
    ml_results.loc[i, 'parameters'] = str(algo.get_params())
    
ml_results
# find the best set of parameters
from sklearn.model_selection import GridSearchCV

param = {'n_estimators': [10, 25 , 50, 100, 250]
}

clf = GridSearchCV(RandomForestRegressor(), param, cv=cv_split)

cv_results = clf.fit(X, y)
cv_results.best_estimator_, cv_results.best_score_


ML = [RandomForestRegressor(n_estimators=500)]

dt = pd.read_csv(diro + 'test.csv', usecols=['datetime'])

# export results into csv
for algo in ML:
    prediction = algo.fit(X, y).predict(test.drop('day', axis=1))
    prediction = np.rint(prediction)
    prediction = pd.concat((dt, pd.Series(prediction)), axis=1)
    prediction.columns = ['datetime', 'count']
    prediction.to_csv(str(algo.__class__.__name__), index=False)



