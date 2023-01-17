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
# Numerical libraries

import numpy as np   



# Import Linear Regression machine learning library

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso



from sklearn.metrics import r2_score



# to handle data in form of rows and columns 

import pandas as pd    



# importing ploting libraries

import matplotlib.pyplot as plt   



import statsmodels.formula.api as sm



#importing seaborn for statistical plots

import seaborn as sns



import datetime

import time

from time import strftime, gmtime



import statsmodels.formula.api as smf

#maschine learning libraries

from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix 



from sklearn.metrics import mean_absolute_error

from sklearn.svm import SVC

from random import sample
df_flights=pd.read_csv("/kaggle/input/flight-delays/flights.csv")
df_flights.head()
df_flights.info()
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
df_flights['ARRIVAL_TIME'] = df_flights['ARRIVAL_TIME'].apply(conv_time)

df_flights['DEPARTURE_TIME'] = df_flights['DEPARTURE_TIME'].apply(conv_time)

df_flights['SCHEDULED_DEPARTURE'] = df_flights['SCHEDULED_DEPARTURE'].apply(conv_time)

df_flights['WHEELS_OFF'] = df_flights['WHEELS_OFF'].apply(conv_time)

df_flights['WHEELS_ON'] = df_flights['WHEELS_ON'].apply(conv_time)

df_flights['SCHEDULED_ARRIVAL'] = df_flights['SCHEDULED_ARRIVAL'].apply(conv_time)
df_flights.isnull().sum()
df_flights['AIRLINE_DELAY'] = df_flights['AIRLINE_DELAY'].fillna(0)

df_flights['AIR_SYSTEM_DELAY'] = df_flights['AIR_SYSTEM_DELAY'].fillna(0)

df_flights['SECURITY_DELAY'] = df_flights['SECURITY_DELAY'].fillna(0)

df_flights['LATE_AIRCRAFT_DELAY'] = df_flights['LATE_AIRCRAFT_DELAY'].fillna(0)

df_flights['WEATHER_DELAY'] = df_flights['WEATHER_DELAY'].fillna(0)
df_flights.isnull().sum()
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
df_flights.isnull().sum()
# drop the last 1% of missing data rows.

df_flights = df_flights.dropna(axis=0)
df_flights.isnull().sum()
df_airlines = pd.read_csv('/kaggle/input/flightdelay/airlines.csv')

df_airlines
# joining airlines

df_flights = df_flights.merge(df_airlines, left_on='AIRLINE', right_on='IATA_CODE', how='inner')
# dropping old column and rename new one

df_flights = df_flights.drop(['AIRLINE_x','IATA_CODE'], axis=1)

df_flights = df_flights.rename(columns={"AIRLINE_y":"AIRLINE"})
fig_dim = (14,18)

f, ax = plt.subplots(figsize=fig_dim)

quality=df_flights["AIRLINE"].unique()

size=df_flights["AIRLINE"].value_counts()



plt.pie(size,labels=quality,autopct='%1.0f%%')

plt.show()
sns.set(style="whitegrid")



# initialize the figure

fig_dim = (10,12)

f, ax = plt.subplots(figsize=fig_dim)

sns.despine(bottom=True, left=True)



# Show each observation with a scatterplot

sns.stripplot(x="ARRIVAL_DELAY", y="AIRLINE",

              data=df_flights, dodge=True, jitter=True

            )

plt.show()
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

f, ax = plt.subplots(figsize=(14, 12))



# Draw the heatmap

sns.heatmap(del_corr,annot=True,cmap='inferno')

plt.show()
# Marking the delayed flights

df_flights['DELAYED'] = df_flights.loc[:,'ARRIVAL_DELAY'].values > 0
figsize=plt.subplots(figsize=(10,12))

sns.countplot(x='DELAYED',hue='AIRLINE',data=df_flights)

plt.show()
# Label definition

y = df_flights.DELAYED



# Choosing the predictors

feature_list_s = [

    'LATE_AIRCRAFT_DELAY'

    ,'AIRLINE_DELAY'

    ,'AIR_SYSTEM_DELAY'

    ,'WEATHER_DELAY'

    ,'ELAPSED_TIME']



# New dataframe based on a small feature list

X_small = df_flights[feature_list_s]
# RandomForestClassifier with 10 trees and fitted on the small feature set 

clf = RandomForestClassifier(n_estimators = 10, random_state=32) 

clf.fit(X_small, y)

importances=clf.feature_importances_

importances=pd.DataFrame([X_small.columns,importances]).transpose()

importances.columns=[['Variables','Importance']]

importances
# choosing the predictors

feature_list = [

    'YEAR'

    ,'MONTH'

    ,'DAY'

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

# Any number can be used in place of '0'. 

import random

random.seed(0)

    

df_flights_1=df_flights.sample(n=50000)

X = df_flights_1[feature_list]

X.info()
y = df_flights_1.DELAYED
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import scale

X_train=scale(X_train)

X_test=scale(X_test)
model=LinearRegression()

model=model.fit(X_train,y_train)

slope=model.coef_

coef=model.intercept_

print(slope.flatten())

print(coef)
y_pred=model.predict(X_train)
r2_score(y_train,y_pred)
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

sfs = SFS(lr, k_features='best', forward=True, floating=False, 

          scoring='neg_mean_squared_error', cv=10)

model = sfs.fit(X_train, y_train)



fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')



plt.title('Sequential Forward Selection (w. StdErr)')

plt.grid()

plt.show()
print('Selected features:', sfs.k_feature_idx_)
lr = LinearRegression()

sfs2 = SFS(lr, k_features='best', forward=False, floating=False, 

          scoring='neg_mean_squared_error', cv=10)

model = sfs2.fit(X_train, y_train)



fig = plot_sfs(sfs2.get_metric_dict(), kind='std_err')



plt.title('Backward Selection (w. StdErr)')

plt.grid()

plt.show()
print('Selected features:', sfs2.k_feature_idx_)
from sklearn import ensemble,gaussian_process,linear_model,naive_bayes,neighbors,svm,tree
MLA = [

    #Ensemble Methods

    ensemble.AdaBoostRegressor(),

    ensemble.BaggingRegressor(),

    ensemble.ExtraTreesRegressor(),

    ensemble.GradientBoostingRegressor(),

    ensemble.RandomForestRegressor(),

    #Nearest Neighbor

    neighbors.KNeighborsRegressor(),

    #Trees    

    tree.DecisionTreeRegressor(),

    tree.ExtraTreeRegressor()

    ]
from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_curve, roc_auc_score,precision_score,recall_score,auc
MLA_columns = []

MLA_compare = pd.DataFrame(columns = MLA_columns)

results=[]



row_index = 0

for alg in MLA:

    

    cv_results = cross_val_score(alg, X_train, y_train, cv=10)

    results.append(cv_results)

    predicted = alg.fit(X_train, y_train).predict(X_test)

    fp, tp, th = roc_curve(y_test, predicted)

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index,'MLA Name'] = MLA_name

    MLA_compare.loc[row_index, 'MLA Train Accuracy'] = round(alg.score(X_train, y_train), 4)

    MLA_compare.loc[row_index, 'MLA Test Accuracy'] = round(alg.score(X_test, y_test), 4)

    MLA_compare.loc[row_index, 'MLA AUC'] = auc(fp, tp)

    

    

    row_index+=1

    

MLA_compare.sort_values(by = ['MLA Test Accuracy'], ascending = False, inplace = True)    

MLA_compare
plt.subplots(figsize=(15,6))

sns.lineplot(x="MLA Name", y="MLA Train Accuracy",data=MLA_compare,palette='hot',label='Train Accuracy')

sns.lineplot(x="MLA Name", y="MLA Test Accuracy",data=MLA_compare,palette='hot',label='Test Accuracy')

plt.xticks(rotation=90)

plt.title('MLA Accuracy Comparison')

plt.legend()

plt.show()
plt.subplots(figsize=(15,6))

sns.lineplot(x="MLA Name", y="MLA AUC",data=MLA_compare,palette='hot',label='Accuracy')



plt.xticks(rotation=90)

plt.title('MLA Accuracy Comparison')

plt.legend()

plt.show()
#sns.boxplot(MLA_compare["MLA AUC"])# boxplot algorithm comparison

fig = plt.figure(figsize=(10,10))

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results,labels=MLA_compare['MLA Name'])

plt.xticks(rotation=45)

plt.show()
# RandomForest with 100 trees

forest_model = RandomForestRegressor(n_estimators = 100, random_state=42)
y = df_flights_1.ARRIVAL_DELAY

y = np.array(y)
X = np.array(X)
# split data into training and validation data, for both predictors and target

# The split is based on a random number generator. Supplying a numeric value to

# the random_state argument guarantees we get the same split every time we

# run this script.



train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.30, random_state = 42)
#The Shape of Train- and Testdata

print('Training Features Shape:', train_X.shape)

print('Training Labels Shape:', train_y.shape)

print('Testing Features Shape:', val_X.shape)

print('Testing Labels Shape:', val_y.shape)
# Average arrival delay for our dataset

baseline_preds = df_flights['ARRIVAL_DELAY'].agg('sum') / df_flights['ARRIVAL_DELAY'].agg('count') 



# Baseline error by average arrival delay 

baseline_errors = abs(baseline_preds - val_y)

print('Average baseline error: ', round(np.mean(baseline_errors),2))
# Fit the model

forest_model.fit(train_X, train_y)
# Predict the target based on testdata 

flightdelay_pred= forest_model.predict(val_X)
#Calculate the absolute errors

errors_random1 = abs(flightdelay_pred - val_y)
print('Mean Absolute Error: ', round(np.mean(errors_random1),3), 'minutes.')
X=pd.DataFrame(X)
importances=forest_model.feature_importances_

importances=pd.DataFrame([X.columns,importances]).transpose()

importances.columns=[['Variables','Importance']]

importances
# Count of DEPARTURE_DELAYs that are not zero and could influence our prediction.

print("DEPARTURE_DELAY count: ")

print(df_flights_1[df_flights_1['DEPARTURE_DELAY'] != 0]['DEPARTURE_DELAY'].count())

print("-------------------------------")

print("All datarow count:")

print((df_flights_1)['DEPARTURE_DELAY'].count())

print("-------------------------------")

print("-------------------------------")

print("Percentag of DEPARTURE_DELAY that is not zero:")

print(df_flights_1[df_flights_1['DEPARTURE_DELAY'] != 0]['DEPARTURE_DELAY'].count() / df_flights_1['DEPARTURE_DELAY'].count())
print("----------------- TRAINING ------------------------")

print("r-squared score: ",forest_model.score(train_X, train_y))

print("------------------- TEST --------------------------")

print("r-squared score: ", forest_model.score(val_X, val_y))
random.seed(1)

df_flights__2=df_flights.sample(n=50000)

X2 = df_flights__2[feature_list]

y2 = df_flights__2.ARRIVAL_DELAY
# Predict the new data based on the old model (forest_model)

flightdelay_pred_ = forest_model.predict(X2)



#Calculate the absolute errors

errors_random_2 = abs(flightdelay_pred_ - y2)

# Mean Absolute Error im comparison

print('Mean Absolute Error Random Sample 1: ', round(np.mean(errors_random1),3), 'minutes.')

print('---------------------------------------------------------------')

print('Mean Absolute Error Random Sample 1: ', round(np.mean(errors_random_2),3), 'minutes.')
print("r-squared score Random Sample 1: ",forest_model.score(val_X, val_y))

print("------------------- TEST --------------------------")

print("r-squared score Random Sample 2: ", forest_model.score(X2, y2))
# Searching for a flight that fits our needs

a=df_flights__2[(df_flights__2.loc[:,'DEPARTURE_DELAY'] < 0) & (df_flights__2.loc[:,'ARRIVAL_DELAY'] > 60)].head(10)

a
# Look into the flight with Arrival Delay but no Departure Delay

a.iloc[1]
# Retrieving the flight with index 3221210 (delayed flight without departure delay).

X3 = a.loc[:,feature_list]

X3 = X3.iloc[0]

# Setting the target for our flight index 3221210

y3 =a.iloc[0]['ARRIVAL_DELAY']

print(y3)

X3

# Printing the important stuff

flight_pred_s = forest_model.predict([X3])

print("Predicted Delay of the Flight (Minutes): ", flight_pred_s)

print("-------------------------------------------------")

print("Original Delay of the Flight (Minutes):  ", y3)

print("_________________________________________________")

print("_________________________________________________")

print("Difference (Minutes)                   : ",  y3-flight_pred_s)