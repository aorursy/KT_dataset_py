# Imports 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap


import seaborn as sns

#maschine learning libraries
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 

from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVC

import datetime
import time
from time import strftime, gmtime
df_flights = pd.read_csv('../input/flights.csv')
df_flights.head()
df_flights.loc[:,('YEAR','MONTH','DAY')].dtypes
df_flights.count()
df_flights.head(15)
df_flights.dtypes
# converting input time value to datetime.
def conv_time(time_val):
    if pd.isnull(time_val):
        return np.nan
    else:
            # replace 24:00 o'clock with 00:00 o'clock:
        if time_val == 2400: time_val = 0
            # creating a 4 digit value out of input value:
        time_val = "{0:04d}".format(int(time_val))
            # creating a time datatype out of input value: 
        time_formatted = datetime.time(int(time_val[0:2]), int(time_val[2:4]))
    return time_formatted
### # convert ARRIVAL_TIME to datetime time format and write it back into df field ARRIVAL_TIME:
df_flights['ARRIVAL_TIME'] = df_flights['ARRIVAL_TIME'].apply(conv_time)
df_flights['DEPARTURE_TIME'] = df_flights['DEPARTURE_TIME'].apply(conv_time)
df_flights['SCHEDULED_DEPARTURE'] = df_flights['SCHEDULED_DEPARTURE'].apply(conv_time)
df_flights['WHEELS_OFF'] = df_flights['WHEELS_OFF'].apply(conv_time)
df_flights['WHEELS_ON'] = df_flights['WHEELS_ON'].apply(conv_time)
df_flights['SCHEDULED_ARRIVAL'] = df_flights['SCHEDULED_ARRIVAL'].apply(conv_time)
df_flights[['YEAR','MONTH','DAY','SCHEDULED_DEPARTURE','DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT',
       'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME','WHEELS_ON', 'TAXI_IN', 'SCHEDULED_ARRIVAL'
            ,'ARRIVAL_TIME','ARRIVAL_DELAY','AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY']].dtypes
#-------------------------------------------------------------
# null value analysing function.
# gives some infos on columns types and number of null values:
def nullAnalysis(df):
    tab_info=pd.DataFrame(df.dtypes).T.rename(index={0:'column type'})

    tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'null values (nb)'}))
    tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100)
                         .T.rename(index={0:'null values (%)'}))
    return tab_info
nullAnalysis(df_flights)
# show selected columns where AIRLINE_DELAY isnot null
df_flights.loc[df_flights['AIRLINE_DELAY'].notnull(), ['AIRLINE_DELAY','AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY','WEATHER_DELAY']].head()
df_flights['AIRLINE_DELAY'] = df_flights['AIRLINE_DELAY'].fillna(0)
df_flights['AIR_SYSTEM_DELAY'] = df_flights['AIR_SYSTEM_DELAY'].fillna(0)
df_flights['SECURITY_DELAY'] = df_flights['SECURITY_DELAY'].fillna(0)
df_flights['LATE_AIRCRAFT_DELAY'] = df_flights['LATE_AIRCRAFT_DELAY'].fillna(0)
df_flights['WEATHER_DELAY'] = df_flights['WEATHER_DELAY'].fillna(0)
nullAnalysis(df_flights)
df_flights.loc[df_flights['CANCELLATION_REASON'].notnull(),['CANCELLATION_REASON']].head(15)
# group by CANCELLATION_REASON to see the ration
df_flights['CANCELLATION_REASON'].value_counts()
# -------------------------------------
# converting categoric value to numeric
df_flights.loc[df_flights['CANCELLATION_REASON'] == 'A', 'CANCELLATION_REASON'] = 1
df_flights.loc[df_flights['CANCELLATION_REASON'] == 'B', 'CANCELLATION_REASON'] = 2
df_flights.loc[df_flights['CANCELLATION_REASON'] == 'C', 'CANCELLATION_REASON'] = 3
df_flights.loc[df_flights['CANCELLATION_REASON'] == 'D', 'CANCELLATION_REASON'] = 4

# -----------------------------------
# converting NaN data to numeric zero
df_flights['CANCELLATION_REASON'] = df_flights['CANCELLATION_REASON'].fillna(0)
# check null values
nullAnalysis(df_flights)
# drop the last 1% of missing data rows.
df_flights = df_flights.dropna(axis=0)

df_times = df_flights[
[
    'SCHEDULED_DEPARTURE',
    'DEPARTURE_TIME',
    'DEPARTURE_DELAY',
    'TAXI_OUT',
    'WHEELS_OFF',
    'SCHEDULED_TIME',
    'ELAPSED_TIME',
    'AIR_TIME',
    'DISTANCE',
    'WHEELS_ON',
    'TAXI_IN',
    'SCHEDULED_ARRIVAL',
    'ARRIVAL_TIME',
    'ARRIVAL_DELAY',
    'DIVERTED',
    'CANCELLED',
    'CANCELLATION_REASON',
    'AIR_SYSTEM_DELAY',
    'SECURITY_DELAY',
    'AIRLINE_DELAY',
    'LATE_AIRCRAFT_DELAY',
    'WEATHER_DELAY'
]]
pd.set_option('float_format', '{:f}'.format)

df_times.describe()
df_airlines = pd.read_csv('../input/airlines.csv')
df_airlines
df_flights['AIRLINE'].value_counts()
# joining airlines
df_flights = df_flights.merge(df_airlines, left_on='AIRLINE', right_on='IATA_CODE', how='inner')
# dropping old column and rename new one
df_flights = df_flights.drop(['AIRLINE_x','IATA_CODE'], axis=1)
df_flights = df_flights.rename(columns={"AIRLINE_y":"AIRLINE"})
sns.set(style="whitegrid")

# initialize the figure
fig_dim = (16,14)
f, ax = plt.subplots(figsize=fig_dim)
sns.despine(bottom=True, left=True)

# Show each observation with a scatterplot
sns.stripplot(x="ARRIVAL_DELAY", y="AIRLINE",
              data=df_flights, dodge=True, jitter=True
            )
# Group by airline and sum up / count the values
df_flights_grouped_sum = df_flights.groupby('AIRLINE', as_index= False)['ARRIVAL_DELAY'].agg('sum').rename(columns={"ARRIVAL_DELAY":"ARRIVAL_DELAY_SUM"})
df_flights_grouped_cnt = df_flights.groupby('AIRLINE', as_index= False)['ARRIVAL_DELAY'].agg('count').rename(columns={"ARRIVAL_DELAY":"ARRIVAL_DELAY_CNT"})

# Merge the two groups together
df_flights_grouped_delay = df_flights_grouped_sum.merge(df_flights_grouped_cnt, left_on='AIRLINE', right_on='AIRLINE', how='inner')
# Calculate the average delay per airline
df_flights_grouped_delay.loc[:,'AVG_DELAY_AIRLINE'] = df_flights_grouped_delay['ARRIVAL_DELAY_SUM'] / df_flights_grouped_delay['ARRIVAL_DELAY_CNT']

df_flights_grouped_delay.sort_values('ARRIVAL_DELAY_SUM', ascending=False)
# Dataframe correlation
del_corr = df_flights.corr()

# Draw the figure
f, ax = plt.subplots(figsize=(11, 9))

# Draw the heatmap
sns.heatmap(del_corr)
# Only using data from January
df_flights_jan = df_flights.loc[(df_flights.loc[:,'YEAR'] == 2015 ) & (df_flights.loc[:,'MONTH'] == 1 )]
df_flights_jan.head()
# Marking the delayed flights
df_flights_jan['DELAYED'] = df_flights_jan.loc[:,'ARRIVAL_DELAY'].values > 0
# Label definition
y = df_flights_jan.DELAYED

# Choosing the predictors
feature_list_s = [
    'LATE_AIRCRAFT_DELAY'
    ,'AIRLINE_DELAY'
    ,'AIR_SYSTEM_DELAY'
    ,'WEATHER_DELAY'
    ,'ELAPSED_TIME']

# New dataframe based on a small feature list
X_small = df_flights_jan[feature_list_s]
# RandomForestClassifier with 10 trees and fitted on the small feature set 
clf = RandomForestClassifier(n_estimators = 10, random_state=32) 
clf.fit(X_small, y)

# Extracting feature importance for each feature
i=0
df_feature_small = pd.DataFrame(columns=['FEATURE','IMPORTANCE'])
for val in (clf.feature_importances_):
    df_feature_small.loc[i] = [feature_list_s[i],val]
    i = i + 1
    

df_feature_small.sort_values('IMPORTANCE', ascending=False)
# choosing the predictors
feature_list = [
    'YEAR'
    ,'MONTH'
    ,'DAY'
    ,'AIRLINE'
    ,'LATE_AIRCRAFT_DELAY'
    ,'AIRLINE_DELAY'
    ,'AIR_SYSTEM_DELAY'
    ,'WEATHER_DELAY'
    ,'ELAPSED_TIME'
    ,'DEPARTURE_DELAY'
    ,'SCHEDULED_TIME'
    ,'AIR_TIME'
    ,'DISTANCE'
    ,'TAXI_IN'
    ,'TAXI_OUT'
    ,'DAY_OF_WEEK'
    ,'SECURITY_DELAY'
]

X = df_flights_jan[feature_list]
# Label encoding of AIRLINE and write this back to df
from sklearn.preprocessing import LabelEncoder
labelenc = LabelEncoder()

# Converting "category" airline to integer values
X.iloc[:,feature_list.index('AIRLINE')] = labelenc.fit_transform(X.iloc[:,feature_list.index('AIRLINE')])
# Convert my encoded categories back
labelenc.inverse_transform(X.iloc[:, feature_list.index('AIRLINE')])
# Fit the new features and the label (based on feature_list)
clf = RandomForestClassifier(n_estimators=10, random_state=32) 
clf.fit(X, y)
i=0
df_feature_selection = pd.DataFrame(columns=['FEATURE','IMPORTANCE'])
for val in (clf.feature_importances_):
    df_feature_selection.loc[i] = [feature_list[i],val]
    i = i + 1
    

df_feature_selection.sort_values('IMPORTANCE', ascending=False)
# RandomForest with 100 trees
forest_model = RandomForestRegressor(n_estimators = 100, random_state=42)
y = df_flights_jan.ARRIVAL_DELAY
y = np.array(y)
X = np.array(X)
# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.35, random_state = 42)
print('Training Features Shape:', train_X.shape)
print('Training Labels Shape:', train_y.shape)
print('Testing Features Shape:', val_X.shape)
print('Testing Labels Shape:', val_y.shape)
# Average arrival delay for our dataset
baseline_preds = df_flights_jan['ARRIVAL_DELAY'].agg('sum') / df_flights_jan['ARRIVAL_DELAY'].agg('count') 

# Baseline error by average arrival delay 
baseline_errors = abs(baseline_preds - val_y)
print('Average baseline error: ', round(np.mean(baseline_errors),2))
# Fit the model
forest_model.fit(train_X, train_y)
# Predict the target based on testdata 
flightdelay_pred= forest_model.predict(val_X)
#Calculate the absolute errors
errors = abs(flightdelay_pred - val_y)
print('Mean Absolute Error: ', round(np.mean(errors),3), 'minutes.')
# Determine the feature importance of our model
i=0
df_model_features = pd.DataFrame(columns=['FEATURE','IMPORTANCE'])
for val in (forest_model.feature_importances_):
    df_model_features.loc[i] = [feature_list[i],val]
    i = i + 1
    
# Print the determined feature importance
df_model_features.sort_values('IMPORTANCE', ascending=False)
from statistics import *

# Calculate the solpe and intercept
def best_fit_slope_and_intercept(xs,ys):
    m = ( ((mean(xs) * mean(ys)) - mean(xs*ys)) /
          ((mean(xs) * mean(xs)) - mean(xs*xs)) )
    b = mean(ys) - m*mean(xs)
    return m, b

# Calculate the regression line
def regression_line(m, feature, b):
        regression_line = [(m*x) + b for x in feature]
        return regression_line

# Draw six grid scatter plot and calculate all necessary functions
def draw_sixgrid_scatterplot(feature1, feature2, feature3, feature4, feature5, feature6, target):
    fig = plt.figure(1, figsize=(16,15))
    gs=gridspec.GridSpec(3,3)
    
    # Axis for the grid
    ax1=fig.add_subplot(gs[0,0])
    ax2=fig.add_subplot(gs[0,1])
    ax3=fig.add_subplot(gs[0,2])
    ax4=fig.add_subplot(gs[1,0])
    ax5=fig.add_subplot(gs[1,1])
    ax6=fig.add_subplot(gs[1,2])
    
    # Drawing dots based on feature and target
    ax1.scatter(feature1, target, color = 'g')
    ax2.scatter(feature2, target, color = 'c')
    ax3.scatter(feature3, target, color = 'y')
    ax4.scatter(feature4, target, color = 'k')
    ax5.scatter(feature5, target, color = 'grey')
    ax6.scatter(feature6, target, color = 'm')
    
    # Get best fit for slope and intercept
    m1,b1 = best_fit_slope_and_intercept(feature1, target)
    m2,b2 = best_fit_slope_and_intercept(feature2, target)
    m3,b3 = best_fit_slope_and_intercept(feature3, target)
    m4,b4 = best_fit_slope_and_intercept(feature4, target)
    m5,b5 = best_fit_slope_and_intercept(feature5, target)
    m6,b6 = best_fit_slope_and_intercept(feature6, target)

    # Build regression lines
    regression_line1 = regression_line(m1, feature1, b1)
    regression_line2 = regression_line(m2, feature2, b2)
    regression_line3 = regression_line(m3, feature3, b3)
    regression_line4 = regression_line(m4, feature4, b4)
    regression_line5 = regression_line(m5, feature5, b5)
    regression_line6 = regression_line(m6, feature6, b6)
            
    # Plotting regression lines
    ax1.plot(feature1,regression_line1)
    ax2.plot(feature2,regression_line2)
    ax3.plot(feature3,regression_line3)
    ax4.plot(feature4,regression_line4)
    ax5.plot(feature5,regression_line5)
    ax6.plot(feature6,regression_line6)
    
    # Naming the axis
    ax1.set_xlabel(feature1.name)
    ax1.set_ylabel(target.name)
    ax2.set_xlabel(feature2.name)    
    ax2.set_ylabel(target.name)
    ax3.set_xlabel(feature3.name)
    ax3.set_ylabel(target.name)
    ax4.set_xlabel(feature4.name)
    ax4.set_ylabel(target.name)
    ax5.set_xlabel(feature5.name)
    ax5.set_ylabel(target.name)
    ax6.set_xlabel(feature6.name)
    ax6.set_ylabel(target.name)
    
    # Give the labels space
    plt.tight_layout()
    plt.show()
        
# Determine the squared error
def squared_error_reg(ys_orig, ys_line):
    return sum((ys_line-ys_orig)**2)

# Calculating r-squared
def coefficient_of_determination(ys_orig, ys_line):
    y_mean:line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error_reg(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)
# Draw the grid scatters
draw_sixgrid_scatterplot(df_flights_jan['DEPARTURE_DELAY'], df_flights_jan['AIR_SYSTEM_DELAY'], df_flights_jan['SCHEDULED_TIME'],
                         df_flights_jan['ELAPSED_TIME'], df_flights_jan['TAXI_OUT'], df_flights_jan['AIRLINE_DELAY'], df_flights_jan['ARRIVAL_DELAY'])


# The original forest model
model = forest_model

# Extract single tree
estimator = model.estimators_[1]

from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(estimator, out_file='tree.dot', 
                max_depth=6,
                rotate = True,
                feature_names = feature_list,
               # class_names = ,
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=3000'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')
i=0
df_model_s_features = pd.DataFrame(columns=['FEATURE','IMPORTANCE'])
for val in (forest_model.feature_importances_):
    df_model_s_features.loc[i] = [feature_list[i],val]
    i = i + 1
    

df_model_s_features.sort_values('IMPORTANCE', ascending=False)
df_feature_selection.sort_values('IMPORTANCE', ascending=False).head(6) 
# Count of DEPARTURE_DELAYs that are not zero and could influence our prediction.
print("DEPARTURE_DELAY count: ")
print(df_flights_jan[df_flights_jan['DEPARTURE_DELAY'] != 0]['DEPARTURE_DELAY'].count())
print("-------------------------------")
print("All datarow count:")
print((df_flights_jan)['DEPARTURE_DELAY'].count())
print("-------------------------------")
print("-------------------------------")
print("Percentag of DEPARTURE_DELAY that is not zero:")
print(df_flights_jan[df_flights_jan['DEPARTURE_DELAY'] != 0]['DEPARTURE_DELAY'].count() / df_flights_jan['DEPARTURE_DELAY'].count())
print("----------------- TRAINING ------------------------")
print("r-squared score: ",forest_model.score(train_X, train_y))
print("------------------- TEST --------------------------")
print("r-squared score: ", forest_model.score(val_X, val_y))
df_flights_feb = df_flights.loc[(df_flights.loc[:,'YEAR'] == 2015 ) & (df_flights.loc[:,'MONTH'] == 2 )]

# We only need them as test sets, no split in train and test(val) needed
X2 = df_flights_feb[feature_list]
y2 = df_flights_feb.ARRIVAL_DELAY
# Converting "category" airline to integer values
X2.iloc[:,feature_list.index('AIRLINE')] = labelenc.fit_transform(X2.iloc[:,feature_list.index('AIRLINE')])
# Filling the features and the target again
X2 = np.array(X2)
y2 = np.array(y2)

# Predict the new data based on the old model (forest_model)
flightdelay_pred_feb = forest_model.predict(X2)

#Calculate the absolute errors
errors_feb = abs(flightdelay_pred_feb - y2)
# Mean Absolute Error im comparison
print('Mean Absolute Error January: ', round(np.mean(errors),3), 'minutes.')
print('---------------------------------------------------------------')
print('Mean Absolute Error February: ', round(np.mean(errors_feb),3), 'minutes.')
print("r-squared score January: ",forest_model.score(val_X, val_y))
print("------------------- TEST --------------------------")
print("r-squared score February: ", forest_model.score(X2, y2))
# Searching for a flight that fits our needs
df_flights_feb[(df_flights_feb.loc[:,'DEPARTURE_DELAY'] < 0) & (df_flights_feb.loc[:,'ARRIVAL_DELAY'] > 60)].head(10)
# Look into the flight with indexnumber 19777
df_flights_feb.loc[19777]
# Setting up a new dataframe for February and converting the AIRLINE feature again
X3 = df_flights_feb.loc[:,feature_list]
X3.iloc[:,feature_list.index('AIRLINE')] = labelenc.fit_transform(X3.iloc[:,feature_list.index('AIRLINE')])

# Retrieving the flight with index 19777 (delayed flight without departure delay).
X3 = X3.loc[19777]
# Setting the target for our flight index 19777
y3 = df_flights_feb.loc[19777]['ARRIVAL_DELAY']

# Converting to array for the model use
X3 = np.array(X3)
y3 = np.array(y3)
# Printing the important stuff
flight_pred_s = forest_model.predict([X3])
print("Predicted Delay of the Flight (Minutes): ", flight_pred_s)
print("-------------------------------------------------")
print("Original Delay of the Flight (Minutes):  ", y3)
print("_________________________________________________")
print("_________________________________________________")
print("Difference (Minutes)                   : ", flight_pred_s - y3)


