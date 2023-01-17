# Data Handling
import pandas as pd
import numpy as np
import math
import scipy.stats as sps
#from scipy import stats, integrate
from time import time


# sklearn and models
from sklearn import preprocessing, ensemble, metrics, feature_selection, model_selection, pipeline
import xgboost as xgb

#plotting and display
from IPython.display import display
from matplotlib import pyplot
# create date parser
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

# create data type converters
dtype_map_weather = dict(Station = 'str')
dtype_map_test_train = dict(Block = 'str', Street = 'str')

# read data into PANDAS DataFrames with date parsing
test = pd.read_csv('../input/test.csv', parse_dates=['Date'], date_parser=dateparse, dtype= dtype_map_test_train)
train = pd.read_csv('../input/train.csv', parse_dates=['Date'], date_parser=dateparse, dtype= dtype_map_test_train)
weather = pd.read_csv('../input/weather.csv', parse_dates=['Date'], date_parser=dateparse, dtype= dtype_map_weather)
sample_sub = pd.read_csv('../input/sampleSubmission.csv')
print('Train')
display(train.info())

print('Test')
display(test.info())
print('Weather')
display(weather.info())
# weather
weather_exclude = ['Dewpoint', 'WetBulb', 'CodeSum', 'Depth', 'Water1', 'SnowFall', 'StnPressure',
                 'SeaLevel', 'ResultSpeed', 'ResultDir', 'AvgSpeed','DewPoint']
weather_cols = [col for col in weather.columns if col not in weather_exclude]
weather = weather[weather_cols]


# train
train_exclude = ['Address', 'AddressNumberAndStreet', 'AddressAccuracy', 'NumMosquitos']
train_cols = [col for col in train.columns if col not in train_exclude]
train = train[train_cols]

# test
test_exclude = ['Address', 'AddressNumberAndStreet', 'AddressAccuracy', 'Id']
test_cols = [col for col in test.columns if col not in test_exclude]
test = test[test_cols]
weather.info()
print('Weather')
display(weather.head())

print('Train')
display(train.head())
# what species have been detected (note that according to the CDC each
# of these species can carry WNV)
set(train.Species)
# does this correspond to the test set
set(test.Species)
# it looks like there is another category
train.groupby('Species').sum().WnvPresent
# strip whitespace
weather.PrecipTotal = weather.PrecipTotal.str.strip()
miss_weather = ['M', '-']
trace_weather = ['T']
cols_not_date = [col for col in weather.columns if col != 'Date']
weather[cols_not_date].apply(pd.value_counts, axis=1)[miss_weather + trace_weather].fillna(0).sum()
# Both stations
check = weather[cols_not_date].apply(pd.value_counts, axis=0).fillna(0)
check.loc[['M', '-', 'T']]
# Station 1
check_stat1 = weather[cols_not_date][weather.Station == '1'].apply(pd.value_counts, axis=0).fillna(0)
check_stat1.loc[['M', '-', 'T']]
# Station 2
check_stat2 = weather[cols_not_date][weather.Station == '2'].apply(pd.value_counts, axis=0).fillna(0)
check_stat2.loc[['M', '-', 'T']]
# Both stations
check.loc[['M', '-', 'T']]/(len(weather)) * 100
# Station 1
check_stat1.loc[['M', '-', 'T']]/(len(weather)) * 100
# Station 2()
check_stat2.loc[['M', '-', 'T']]/(len(weather)) * 100
weather = weather.replace('M', np.NaN)
weather = weather.replace('-', np.NaN)
weather = weather.replace('T', 0.005) # very small amounts of rain can impact mosquito hatches
weather.Tmax = weather.Tmax.fillna(method = 'ffill')
weather.Tmin = weather.Tmin.fillna(method = 'ffill')
weather.Depart = weather.Depart.fillna(method = 'ffill')
weather.Heat = weather.Heat.fillna(method = 'ffill')
weather.Cool = weather.Cool.fillna(method = 'ffill')
weather.PrecipTotal = weather.PrecipTotal.fillna(method = 'ffill')
# convert datatpypes

to_numeric = ['Tmax','Tmin','Tavg', 'Depart', 'Heat', 'Cool', 'PrecipTotal']

for col in to_numeric:
    weather[col]= pd.to_numeric(weather[col])
weather.Sunrise = weather.Sunrise.fillna(method = 'ffill')
weather.Sunset = weather.Sunset.fillna(method = 'ffill')
# sunset has entries where instead of incrementing to the next hour after xx59 it incremented to xx60
# This causes an exception, let's take a look
counter = 0
tracker = []
for index, val in enumerate(weather.Sunset):
    try:
        pd.to_datetime(val, format = '%H%M').time()
    except:
        counter += 1
        tracker.append((index, val, val[2:], counter))

print(tracker[-1])

# there are 48 exceptions
# let's deal with this by decrmenting by 1 for each invalid instance
weather.Sunset = weather.Sunset.replace('\+?60', '59', regex = True)
# time conversion lambda function
time_func = lambda x: pd.Timestamp(pd.to_datetime(x, format = '%H%M'))
weather.Sunrise = weather.Sunrise.apply(time_func)
weather.Sunset = weather.Sunset.apply(time_func)
# what is the range of values for sunrise and sunset (in hours)
minutes= (weather.Sunset - weather.Sunrise).astype('timedelta64[m]')
hours = minutes/60
set(np.round(hours.values))
#create a DayLength column with minute level precsion
weather['DayLength_MPrec'] = (weather.Sunset - weather.Sunrise).astype('timedelta64[m]')/60
#create a DayLength column with rounded to the nearest hour
weather['DayLength_NearH'] = np.round(((weather.Sunset - weather.Sunrise).astype('timedelta64[m]')/60).values)
# length of night with minute level precision
weather['NightLength_MPrec']= 24.0 - weather.DayLength_MPrec
# lenght of night rounded to nearest hour
weather['NightLength_NearH']= 24.0 - weather.DayLength_NearH
# function to calculate sunset and sunrise times in hours
hours_RiseSet_func = lambda x: x.minute/60.0 + float(x.hour)
# sunrise in hours
weather['Sunrise_hours'] = weather.Sunrise.apply(hours_RiseSet_func)
# sunset in hours
weather['Sunset_hours'] = weather.Sunset.apply(hours_RiseSet_func)
mean_func = lambda x: x.mean()

blend_cols = ['Tmax', 'Tmin', 'Depart' ,'Heat', 'Cool', 'PrecipTotal']
blended_cols= ['blended_' + col for col in blend_cols]
station_1 = weather[blend_cols][weather.Station == '1']
station_2 = weather[blend_cols][weather.Station == '2']
station_blend = pd.DataFrame((station_1.values + station_2.values)/2, columns= blended_cols)
extract_2 = weather[weather.Station == '2'].reset_index(drop = True)
extract_2.head()
extract_1 = weather[weather.Station == '1'].reset_index(drop = True)
extract_1.head()
joined_1 = extract_1.join(station_blend)
joined_2 = extract_2.join(station_blend)
weather_blend = pd.concat([joined_1, joined_2])
weather_blend.info()
month_func = lambda x: x.month
day_func= lambda x: x.day
day_of_year_func = lambda x: x.dayofyear
week_of_year_func = lambda x: x.week

# train
train['month'] = train.Date.apply(month_func)
train['day'] = train.Date.apply(day_func)
train['day_of_year'] = train.Date.apply(day_of_year_func)
train['week'] = train.Date.apply(week_of_year_func)

# test
test['month'] = test.Date.apply(month_func)
test['day'] = test.Date.apply(day_func)
test['day_of_year'] = test.Date.apply(day_of_year_func)
test['week'] = test.Date.apply(week_of_year_func)
train.describe()
test.describe()
# remove sunrise and sunset since we have extracted critical information into other fields
weather_blend = weather_blend.drop(['Sunrise', 'Sunset'], axis= 1)
train = train.merge(weather_blend, on='Date')
test = test.merge(weather_blend, on='Date')
weather_blend.ix[:,:12].describe()
weather_blend.ix[:,12:].describe()
train.describe()
# columns to write
cols_to_write = [col for col in train.columns if col != 'Date'] # exclude 'Date'
# split the data into two dataframes by station

train_station_1= train[train.Station == '1']
train_station_2= train[train.Station == '2']

test_station_1= test[test.Station == '1']
test_station_2= test[test.Station == '2']
# export to JSON for external use
#train_station_1.to_json('train_station_1.json')
#train_station_2.to_json('train_station_2.json')
#train.to_json('train.json')

# epxort to csv for external use
#train_station_1.to_csv('train_station_1.csv')
#train_station_2.to_csv('train_station_2.csv')
train.to_csv('train.csv')
# set up a merge for stations 1 and 2
# keep unique cols from station 2
keep_cols = ['Date', u'Tmax', u'Tmin', u'Tavg',u'PrecipTotal']
train_station_2 = train_station_2[keep_cols]
test_station_2 = test_station_2[keep_cols]

# rename cols with prefix
prefix_s2 = 'stat_2_'
rename_cols_s2 = [prefix_s2 + col for col in train_station_2.columns]
train_station_2.columns = rename_cols_s2
test_station_2.columns = rename_cols_s2
# drop cols from station 1 that won't be used in model
drop_cols = ['Heat', 'Cool', 'Depart', 'NightLength_MPrec', 'NightLength_NearH',
            'blended_Depart', 'blended_Heat', 'blended_Cool']

train_station_1 = train_station_1.drop(drop_cols, axis= 1)
test_station_1 = test_station_1.drop(drop_cols, axis= 1)   
# raname uniqe station 1 columns
prefix_s1 = 'stat_1_'
rename_cols_s1 = [prefix_s1 + col for col in keep_cols]
cols_to_rename= [col for col in train_station_1.columns if col in keep_cols]

# setup name mapping
s1_name_map = dict(zip(cols_to_rename, rename_cols_s1))

train_station_1 = train_station_1.rename(columns= s1_name_map)
test_station_1 = test_station_1.rename(columns= s1_name_map)
# concat (outer join)
train_station_1 =  train_station_1.reset_index(drop= True)
train_station_2 = train_station_2.reset_index(drop = True)
train_merge = pd.concat([train_station_1, train_station_2], axis= 1)

test_station_1 =  test_station_1.reset_index(drop= True)
test_station_2 = test_station_2.reset_index(drop = True)
test_merge = pd.concat([test_station_1, test_station_2], axis= 1)
train_merge.columns
test_merge.columns
# get label
labels = train_merge.pop('WnvPresent').values
# remove dates
train_merge = train_merge.drop(['stat_1_Date', 'stat_2_Date'], axis = 1)

test_merge = test_merge.drop(['stat_1_Date', 'stat_2_Date' ], axis = 1)
# add lat and long integer columns

train_merge['Lat_int'] = train_merge.Latitude.astype(int)
train_merge['Long_int'] = train_merge.Longitude.astype(int)

test_merge['Lat_int'] = test_merge.Latitude.astype(int)
test_merge['Long_int'] = test_merge.Longitude.astype(int)
# Create dummies from the categorical species, block, trap, and streetname
train_merge = pd.get_dummies(train_merge, columns= ['Species'])
train_merge = pd.get_dummies(train_merge, columns= ['Block'])
train_merge = pd.get_dummies(train_merge, columns= ['Street'])
train_merge = pd.get_dummies(train_merge, columns= ['Trap'])

test_merge = pd.get_dummies(test_merge, columns= ['Species'])
test_merge = pd.get_dummies(test_merge, columns= ['Block'])
test_merge = pd.get_dummies(test_merge, columns= ['Street'])
test_merge = pd.get_dummies(test_merge, columns= ['Trap'])
#train_merge= train_merge.drop(['Street', 'Trap', 'Station'], axis= 1)
#test_merge= test_merge.drop(['Street', 'Trap', 'Station'], axis= 1)

train_merge= train_merge.drop('Station', axis= 1)
test_merge= test_merge.drop('Station', axis= 1)
#drops= ['Block', 'Street', 'Trap', 'Latitude', 'Longitude']

#train_merge= train_merge.drop(drops, axis= 1)
#test_merge= test_merge.drop(drops, axis= 1)
len(train_merge.columns)
len(test_merge.columns)
unique_test_cols = [col for col in test_merge.columns if col not in train_merge.columns]
test_merge= test_merge.drop(unique_test_cols, axis= 1)
# epxort to csv for external use
#train_merge.to_csv('train_merge.csv')
#train_merge.to_csv('test_merge.csv')
clf = ensemble.RandomForestClassifier(n_estimators=1000, min_samples_split= 2, random_state= 42)
clf.fit(train_merge, labels)
# create predictions and submission file
predictions_randfor = clf.predict_proba(test_merge)[:,1]
# fit model no training data
xgbc = xgb.XGBClassifier(seed= 42)
xgbc.fit(train_merge, labels)
# feature importance
#print(xgb.feature_importances_)

# plot feature importance
fig, ax = pyplot.subplots(figsize=(10, 15))
xgb.plot_importance(xgbc, ax=ax)
#pyplot.show()
xgbc.get_fscore()
# feature importance
xgbc.get_fscore()
#print(xgbc.feature_importances_)
def calc_roc_auc(y, predict_probs):
    
    """
    Function accepts labels (matrix y) and predicted probabilities
    Function calculates fpr (false positive rate), tpr (true postivies rate), thresholds and auc (area under
    the roc curve)
    Function returns auc
    """
    fpr, tpr, thresholds = metrics.roc_curve(y, predict_probs)
    roc_auc = metrics.auc(fpr, tpr)
    
    return roc_auc
train_split, val_split, label_train_split, label_val_split = model_selection.train_test_split(train_merge, 
                                      labels, test_size = 0.33, random_state = 42, stratify= labels)
def select_features_by_importance_threshold(model, X_train, y_train, selection_model, X_test, y_test,
                                           minimum = False):

    # Fit model using each importance as a threshold
    if minimum:
        thresholds= np.unique(model.feature_importances_[model.feature_importances_ > minimum])
        # include 0 for all features
        thresholds = np.insert(thresholds, 0, 0.)
    else:
        thresholds= np.unique(model.feature_importances_)
        
    
    print(thresholds)
    for thresh in thresholds:
	    # select features using threshold
        selection = feature_selection.SelectFromModel(model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(X_train)
	    # train model
        selection_model = selection_model
        selection_model.fit(select_X_train, y_train)
	    # eval model
        select_X_test = selection.transform(X_test)
        y_pred = selection_model.predict_proba(select_X_test)[:,1]
        predictions = y_pred
        #predictions = [round(value) for value in y_pred]
        auc = calc_roc_auc(y_test, predictions)
        print("Thresh=%.3f, n=%d, AUC: %.2f%%" % (thresh, select_X_train.shape[1], auc))
train_merge.shape
# Set a minimum threshold of 0.023
sfm = feature_selection.SelectFromModel(xgbc, threshold=0.023, prefit= True)
sfm_train= sfm.transform(train_merge)
n_features = sfm_train.shape[1]
print(n_features)
# initialize and fit model
xgb_clf= xgb.XGBClassifier(seed= 42)
xgb_clf.fit(sfm_train, labels)
sfm_test = sfm.transform(test_merge)
predictions_xgb = xgb_clf.predict_proba(sfm_test)[:,1]
# plot single tree
xgb.plot_tree(xgb_clf, rankdir= 'LR')
pyplot.show()
X_train= train_split
X_test= val_split
y_train= label_train_split
y_test= label_val_split
model= xgb.XGBClassifier(seed= 42)

eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, eval_metric="auc", eval_set=eval_set, verbose=True)
results = model.evals_result()
print(results)
model.fit(X_train, y_train, eval_metric=["auc", "logloss", "error"], eval_set=eval_set)
# retrieve performance metrics
results = model.evals_result()
epochs = len(results['validation_0']['auc'])
x_axis = range(0, epochs)

# plot auc
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['auc'], label='Train')
ax.plot(x_axis, results['validation_1']['auc'], label='Test')
ax.legend()
pyplot.ylabel('AUC')
pyplot.title('XGBoost AUC by Epoch')
pyplot.show()

# plot logloss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
pyplot.ylabel('Logloss')
pyplot.title('XGBoost Logloss by Epoch')
pyplot.show()

# plot error
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
pyplot.ylabel('Error')
pyplot.title('XGBoost Error by Epoch')
pyplot.show()
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, eval_metric=["auc"], eval_set=eval_set, early_stopping_rounds=10)
results = model.evals_result()
print(results)
# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
#n_estimators_dist= np.random.randint(1, 500)# number of trees, could use a discrete list or np.random.exponential(scale=0.1, size= 100)
#colsample_bytree_dist= np.random.uniform(0.2,0.6) # should be 0.3 - 0.5
#max_depth_dist = np.random.randint(2, 12) # typical values 3 - 10
#learning_rate_dist= np.random.uniform(0.01, 0.3) # default 0.3, typical values 0.01 - 0.2

#learning_rate_dist= scipy.stats.expon(scale=100)
#learning_rate_dist= 10. ** np.arange(-3, -2)
n_estimators_dist= sps.randint(1, 300)
learning_rate_dist = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
#cv = model_selection.StratifiedShuffleSplit(n_splits = 10, random_state = 42)  

param_dist = dict(learning_rate= learning_rate_dist, n_estimators= n_estimators_dist) 

# run randomized search
n_iter_search = 20
random_search = model_selection.RandomizedSearchCV(model, param_distributions=param_dist,
                                   n_iter=n_iter_search, scoring= 'roc_auc')

start = time()
random_search.fit(X_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)
sample_sub['WnvPresent'] = predictions_xgb
sample_sub.to_csv('sub_xgb.csv', index=False)

#sample_sub['WnvPresent'] = predictions_randfor
#sample_sub.to_csv('sub_randfor.csv', index=False)