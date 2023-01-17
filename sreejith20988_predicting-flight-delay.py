# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import RandomForestClassifier



import sklearn.tree as tree

#import pydotplus

from sklearn.externals.six import StringIO 

from IPython.display import Image





from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import auc

from sklearn.metrics import plot_roc_curve

from scipy import interp



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#flights = pd.read_csv('../input/flight-delays/flights.csv')

#flights_sub = flights.sample(n = 10000, random_state = 123)

#flights_sub.info()

#flights_sub.to_csv('flights_sample.csv')
file = '../working/flights_sample.csv'

flights_sub = pd.read_csv(file)

print(flights_sub.shape)
flights_sub.head()
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, sharey = False, figsize=(12,3))

sns.distplot(flights_sub['DEPARTURE_DELAY'], kde = True, bins = 5, ax = ax1)

sns.boxplot(data = flights_sub, x = 'DEPARTURE_DELAY', ax = ax2)
fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (18,3))

sns.distplot(flights_sub.query('DEPARTURE_DELAY > 30 & DEPARTURE_DELAY < 300')['DEPARTURE_DELAY'], kde = True, bins = 5, ax = ax1)

sns.stripplot(data = flights_sub.query('DEPARTURE_DELAY > 30 & DEPARTURE_DELAY < 300'), x = 'DEPARTURE_DELAY', jitter = True, ax = ax2)

sns.stripplot(data = flights_sub.query('DEPARTURE_DELAY > 30 & DEPARTURE_DELAY < 300'), x = 'DEPARTURE_DELAY', y ='AIRLINE', jitter = True, ax = ax3)
print(flights_sub['DEPARTURE_DELAY'].isna().sum())

print(flights_sub[flights_sub['DEPARTURE_DELAY'].isna()]['CANCELLED'].sum())

print(flights_sub[flights_sub['DEPARTURE_DELAY'].isna()].groupby(['CANCELLATION_REASON'])['CANCELLATION_REASON'].count())

flights_sub['DELAYED'] = flights_sub['DEPARTURE_DELAY']>30

flights_sub = flights_sub[flights_sub['DEPARTURE_DELAY']<300]

sns.catplot(data = flights_sub, kind = 'count', y = 'AIRLINE', hue = 'DELAYED', aspect = 0.8, ax = ax1)
pd.concat([round((flights_sub[(flights_sub['DEPARTURE_DELAY']>30)]['AIRLINE'].value_counts() / flights_sub['AIRLINE'].value_counts())*100,0).sort_values(ascending = False) ,

          flights_sub['AIRLINE'].value_counts()], axis = 1, sort = False)

airport_codes = pd.read_csv('../input/airport-codes/Airport_codes.csv', dtype ={'DOT Code': str,'Code':str})

airport_codes.head()

flights_sub['ORIGIN_AIRPORT'] = flights_sub['ORIGIN_AIRPORT'].astype(str)

flights_sub.reset_index(inplace=True, drop=True)





airports_fixed = pd.merge(flights_sub, airport_codes, left_on='ORIGIN_AIRPORT', right_on = 'DOT Code')

flights_sub.drop(flights_sub[flights_sub['ORIGIN_AIRPORT'].str.len()>3].index, inplace = True)

#airports_fixed['ORIGIN_AIRPORT'] = airports_fixed['DOT Code']

#flights_sub = pd.concat([airports_fixed, flights_sub], axis = 0, sort = False)

fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (15,3))

sns.countplot(data = flights_sub.query('DELAYED == True'), y = 'ORIGIN_AIRPORT', ax = ax2)

sns.countplot(data = flights_sub, y = 'ORIGIN_AIRPORT', ax = ax1)

sns.distplot(flights_sub['ORIGIN_AIRPORT'].value_counts(), ax = ax3)
bin_pct_97 = flights_sub['ORIGIN_AIRPORT'].value_counts().quantile(0.97)

bin_pct_90 = flights_sub['ORIGIN_AIRPORT'].value_counts().quantile(0.90)

bin_pct_75 = flights_sub['ORIGIN_AIRPORT'].value_counts().quantile(0.75)



airports = flights_sub['ORIGIN_AIRPORT'].value_counts()

airports_index = flights_sub[flights_sub['ORIGIN_AIRPORT'].isin(airports[airports > bin_pct_97].index)].index

flights_sub.loc[airports_index,'AIRPORT_TYPE'] = 'Heavy'



airports_index = flights_sub[flights_sub['ORIGIN_AIRPORT'].isin(airports[(airports > bin_pct_90)&(airports <= bin_pct_97)].index)].index

flights_sub.loc[airports_index,'AIRPORT_TYPE'] = 'Medium'



airports_index = flights_sub[flights_sub['ORIGIN_AIRPORT'].isin(airports[(airports > bin_pct_75)&(airports <= bin_pct_90)].index)].index

flights_sub.loc[airports_index,'AIRPORT_TYPE'] = 'Light'



airports_index = flights_sub[flights_sub['ORIGIN_AIRPORT'].isin(airports[airports <= bin_pct_75].index)].index

flights_sub.loc[airports_index,'AIRPORT_TYPE'] = 'Very Light'
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows = 1, ncols = 4, figsize = (20,5), sharey = False)

sns.countplot(data = flights_sub[flights_sub['AIRPORT_TYPE']=='Very Light'], y = 'ORIGIN_AIRPORT', ax = ax4)

sns.countplot(data = flights_sub[flights_sub['AIRPORT_TYPE']=='Light'], y = 'ORIGIN_AIRPORT', ax = ax3)

sns.countplot(data = flights_sub[flights_sub['AIRPORT_TYPE']=='Medium'], y = 'ORIGIN_AIRPORT', ax = ax2)

sns.countplot(data = flights_sub[flights_sub['AIRPORT_TYPE']=='Heavy'], y = 'ORIGIN_AIRPORT', ax = ax1)
flights_sub['Date'] = pd.to_datetime(flights_sub[['YEAR', 'MONTH', 'DAY']])

flights_sub['MONTH'] = flights_sub['Date'].dt.month
flights_sub['WEATHER_DELAY'] = flights_sub['WEATHER_DELAY'].fillna(0)

flights_sub['WEATHER_DELAY'].values

flights_sub[flights_sub['DELAYED']==True]['WEATHER_DELAY'].values

plt.hist(flights_sub[flights_sub['DELAYED']==True]['WEATHER_DELAY'].values, log = True)

plt.hist(flights_sub[flights_sub['DELAYED']==False]['WEATHER_DELAY'].values, log = True)



plt.clf()

plt.hist(flights_sub[flights_sub['DELAYED']==True]['SECURITY_DELAY'].values, log = False, bins = 3)

plt.hist(flights_sub[flights_sub['DELAYED']==False]['SECURITY_DELAY'].values, log = False, bins = 3)



plt.clf()

plt.hist(flights_sub['AIR_SYSTEM_DELAY'].values, log = False, bins = 3)

plt.hist(flights_sub[flights_sub['DELAYED']==True]['AIR_SYSTEM_DELAY'].values, log = False, bins = 3)



plt.clf()

plt.hist(flights_sub['AIRLINE_DELAY'].values, log = False, bins = 50)

plt.hist(flights_sub[flights_sub['DELAYED']==True]['AIRLINE_DELAY'].values, log = False, bins = 50)



plt.clf()

plt.hist(flights_sub['LATE_AIRCRAFT_DELAY'].values, log = False, bins = 50)

plt.hist(flights_sub[flights_sub['DELAYED']==True]['LATE_AIRCRAFT_DELAY'].values, log = False, bins = 50)

predictors = ['MONTH', 'AIRLINE','AIRPORT_TYPE', 'SECURITY_DELAY', 'WEATHER_DELAY', 'AIR_SYSTEM_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY']

features = ['MONTH','AIRLINE', 'AIRPORT_TYPE', 'SECURITY_DELAY', 'WEATHER_DELAY', 'AIR_SYSTEM_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY','DELAYED']

flights_model = flights_sub[features]

flights_model = flights_model.fillna(0)

random_state = 40



y = flights_model['DELAYED'].values

X = flights_model.drop('DELAYED', axis = 1).values



labelencoder_X = LabelEncoder()

X[:,1] =labelencoder_X.fit_transform(X[:,1])

X[:,2] =labelencoder_X.fit_transform(X[:,2])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = random_state)



dt = DecisionTreeClassifier(criterion = 'gini',max_depth = 4, random_state = random_state)

dt.fit(X_train, y_train)



y_pred = dt.predict(X_test)



accuracy_gini = accuracy_score(y_test, y_pred)

print(accuracy_gini)
dt = DecisionTreeClassifier(criterion = 'entropy',max_depth = 4, random_state = random_state)

dt.fit(X_train, y_train)



y_pred = dt.predict(X_test)



accuracy_entropy = accuracy_score(y_test, y_pred)

print(accuracy_entropy)


#cv = StratifiedKFold(n_splits=6)

cv = 50

clf = DecisionTreeClassifier(criterion = 'entropy',max_depth = 50, min_samples_leaf = 0.05, random_state = random_state)

clf.fit(X_train, y_train)

clf.predict(X_test)



scores = cross_val_score(clf, X_train, y_train, cv = cv, n_jobs = -1)

scores.mean()
cf_matrix = confusion_matrix(y_test, y_pred)

cf_matrix_per = cf_matrix/np.sum(cf_matrix)



print(y_test.shape)

print(cf_matrix)

sns.heatmap(cf_matrix_per, annot=True, fmt='.2%', cmap='Blues')

# dot_data = StringIO()

# tree.export_graphviz(clf, 

# out_file=dot_data, 

# class_names=['0','1'], # the target names.

# feature_names=predictors, # the feature names.

# filled=True, # Whether to fill in the boxes with colours.

# rounded=True, # Whether to round the corners of the boxes.

# special_characters=True)

# graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 

# Image(graph.create_png())
#bc = BaggingClassifier(base_estimator = clf, n_estimators = 100, n_jobs = -1)

#bc.fit(X_train, y_train)

#y_pred = bc.predict(X_test)



#accuracy = accuracy_score(y_test, y_pred)

#print(round(accuracy,2))
#bc = BaggingClassifier(base_estimator = clf, n_estimators = 100, n_jobs = -1, oob_score = True)

#bc.fit(X_train, y_train)

#y_pred = bc.predict(X_test)



#oob_accuracy = bc.oob_score_

#print(oob_accuracy)
rf = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', max_depth = 50, min_samples_leaf = 0.02, n_jobs = -1)

rf.fit(X_train, y_train)



y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(accuracy)
cf_matrix = confusion_matrix(y_test, y_pred)

cf_matrix_per = cf_matrix/np.sum(cf_matrix)



print(y_test.shape)

print(cf_matrix)

sns.heatmap(cf_matrix_per, annot=True, fmt='.2%', cmap='Blues')

importances_rf = pd.Series(rf.feature_importances_, index = predictors)

sorted_importances_rf = importances_rf.sort_values()

sorted_importances_rf.plot(kind = 'barh', color = 'm')

plt.show()