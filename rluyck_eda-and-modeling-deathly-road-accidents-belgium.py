import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns

import os, re, sys

import time
df_accidents = pd.read_csv('/kaggle/input/road-accidents-belgium/road_accidents_2005_2018.csv')
df_accidents.shape
df_accidents.columns
df_accidents = df_accidents.drop(['Unnamed: 0','MS_ACCT_WITH_MORY_INJ',

       'MS_ACCT_WITH_SERLY_INJ', 'MS_ACCT_WITH_SLY_INJ','MS_ACCT'],axis=1)
cols = [c for c in df_accidents.columns if not c.endswith('FR')]

cols

df_accidents=df_accidents[cols]
df_accidents.columns
df_accidents.shape
df_accidents.head(5)
type(df_accidents['DT_DAY'].iloc[1])
df_accidents['date'] = pd.to_datetime(df_accidents['DT_DAY'])

df_accidents = df_accidents.drop('DT_DAY',axis=1)
type(df_accidents['date'].iloc[1])
df_accidents['day'] = df_accidents['date'].apply(lambda date: date.day)

df_accidents['month'] = df_accidents['date'].apply(lambda date: date.month)

df_accidents['year'] = df_accidents['date'].apply(lambda date: date.year)

df_accidents['quarter'] = df_accidents['date'].apply(lambda date: date.quarter)
df_accidents.index = pd.DatetimeIndex(df_accidents['date'])
plt.figure(figsize=(15,6))

plt.title('Distribution of accidents per day', fontsize=16)

plt.tick_params(labelsize=14)

sns.distplot(df_accidents.resample('D').size(), bins=60);
accidents_daily = pd.DataFrame(df_accidents.resample('D').size())

accidents_daily['MEAN'] = df_accidents.resample('D').size().mean()

accidents_daily['STD'] = df_accidents.resample('D').size().std()

UCL = accidents_daily['MEAN'] + 3 * accidents_daily['STD']

LCL = accidents_daily['MEAN'] - 3 * accidents_daily['STD']



plt.figure(figsize=(15,6))

df_accidents.resample('D').size().plot(label='Accidents per day')

UCL.plot(color='red', ls='--', linewidth=1.5, label='UCL')

LCL.plot(color='red', ls='--', linewidth=1.5, label='LCL')

accidents_daily['MEAN'].plot(color='red', linewidth=2, label='Average')

plt.title('Total accidents per day', fontsize=16)

plt.xlabel('Day')

plt.ylabel('Number of accidents')

plt.tick_params(labelsize=14)

plt.legend(prop={'size':16})
plt.figure(figsize=(15,6))

df_accidents.resample('M').size().plot(label='Total per month')

df_accidents.resample('M').size().rolling(window=12).mean().plot(color='red', linewidth=5, label='12-months Moving Average')



plt.title('Accidents per month', fontsize=16)

plt.xlabel('')

plt.legend(prop={'size':16})

plt.tick_params(labelsize=16)



plt.figure(figsize=(15,6))

plt.title('Distribution of accidents per month', fontsize=16)

plt.tick_params(labelsize=14)

sns.distplot(df_accidents.resample('M').size(), bins=60);
accidents_daily = pd.DataFrame(df_accidents.resample('M').size())

accidents_daily['MEAN'] = df_accidents.resample('M').size().mean()

accidents_daily['STD'] = df_accidents.resample('M').size().std()

UCL = accidents_daily['MEAN'] + 3 * accidents_daily['STD']

LCL = accidents_daily['MEAN'] - 3 * accidents_daily['STD']



plt.figure(figsize=(15,6))

df_accidents.resample('M').size().plot(label='Accidents per month')

UCL.plot(color='red', ls='--', linewidth=1.5, label='UCL')

LCL.plot(color='red', ls='--', linewidth=1.5, label='LCL')

accidents_daily['MEAN'].plot(color='red', linewidth=2, label='Average')

plt.title('Total accidents per month', fontsize=16)

plt.xlabel('Month')

plt.ylabel('Number of accidents')

plt.tick_params(labelsize=14)

plt.legend(prop={'size':16})
accidents_daily = pd.DataFrame(df_accidents.resample('Y').size())

accidents_daily['MEAN'] = df_accidents.resample('Y').size().mean()

accidents_daily['STD'] = df_accidents.resample('Y').size().std()

UCL = accidents_daily['MEAN'] + 3 * accidents_daily['STD']

LCL = accidents_daily['MEAN'] - 3 * accidents_daily['STD']



plt.figure(figsize=(15,6))

df_accidents.resample('Y').size().plot(label='Accidents per year')

UCL.plot(color='red', ls='--', linewidth=1.5, label='UCL')

LCL.plot(color='red', ls='--', linewidth=1.5, label='LCL')

accidents_daily['MEAN'].plot(color='red', linewidth=2, label='Average')

plt.title('Total accidents per year', fontsize=16)

plt.xlabel('Year')

plt.ylabel('Number of accidents')

plt.tick_params(labelsize=14)

plt.legend(prop={'size':16})
df_accidents.columns
### Remove duplicate text attributes
df_accidents = df_accidents.drop(['TX_DAY_OF_WEEK_DESCR_NL','TX_BUILD_UP_AREA_DESCR_NL',

                                 'TX_COLL_TYPE_DESCR_NL','TX_LIGHT_COND_DESCR_NL',

                                 'TX_ROAD_TYPE_DESCR_NL','TX_MUNTY_DESCR_NL',

                                 'TX_ADM_DSTR_DESCR_NL','TX_PROV_DESCR_NL',

                                 'TX_RGN_DESCR_NL'],axis=1)
df_accidents.columns
df_accidents.isnull().sum()
(df_accidents.isnull().sum())/len(df_accidents)
attributes_missing = [att for att in df_accidents.columns if df_accidents[att].isnull().sum()>0 ]
attributes_missing
for att in attributes_missing:

    

    print(df_accidents[att].value_counts())
attributes_missing
#replace nan's 



for att in attributes_missing:

    

    if att == 'CD_BUILD_UP_AREA':

        df_accidents[att].fillna(1.0, inplace = True) 

    elif att == 'CD_COLL_TYPE':

        df_accidents[att].fillna(4.0, inplace = True) 

    elif att == 'CD_LIGHT_COND':

        df_accidents[att].fillna(1.0, inplace = True) 

    elif att == 'CD_ROAD_TYPE':

        df_accidents[att].fillna(2.0, inplace = True) 

    else:

        df_accidents[att].fillna(10000.0, inplace = True)
df_accidents.isnull().sum()
plt.figure(figsize=(15,8))

sns.heatmap(df_accidents.corr().round(2),annot=True)
df_accidents.columns
# Create a pivot table by crossing the day number by the month and calculate the average number of accidents for each crossing

accidents_pivot_table = df_accidents.pivot_table(values='date', index='day', columns='month', aggfunc=len)

accidents_pivot_table_date_count = df_accidents.pivot_table(values='date', index='day', columns='month', aggfunc=lambda x: len(x.unique()))

accidents_average = accidents_pivot_table/accidents_pivot_table_date_count

accidents_average.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']



# Using seaborn heatmap

plt.figure(figsize=(7,9))

plt.title('Average accidents per day and month', fontsize=14)

sns.heatmap(accidents_average.round(), cmap='coolwarm', linecolor='grey',linewidths=0.1, cbar=False, annot=True, fmt=".0f")
df_accidents = df_accidents.drop('date',axis=1)
for att in df_accidents.columns:

    plt.figure() #this creates a new figure on which your plot will appear

    sns.countplot(df_accidents[att])
df_accidents.shape
df_accidents = df_accidents[df_accidents['MS_ACCT_WITH_DEAD_30_DAYS']!=2]
df_accidents = df_accidents[df_accidents['MS_ACCT_WITH_DEAD']!=2]
df_accidents.shape
df_accidents.columns
df_accidents = df_accidents.rename(index=str, columns={'CD_BUILD_UP_AREA':'where',

                                                      'CD_COLL_TYPE':'how',

                                                      'CD_ROAD_TYPE':'typeofroad',

                                                      'CD_MUNTY_REFNIS':'refnismun',

                                                      'CD_DSTR_REFNIS':'refnisdist',

                                                      'CD_PROV_REFNIS':'refnisprov',

                                                      'CD_RGN_REFNIS':'refnisgew',

                                                      'CD_LIGHT_COND':'illumination',

                                                      'MS_ACCT_WITH_DEAD':'dead',

                                                      'MS_ACCT_WITH_DEAD_30_DAYS':'deadafter30d'})
df_accidents.head()
print(df_accidents['refnismun'].nunique())

print(df_accidents['refnisdist'].nunique())

print(df_accidents['refnisprov'].nunique())

print(df_accidents['refnisgew'].nunique())
df_accidents = df_accidents.drop(['refnisdist','refnisprov','refnisgew'],axis=1)
df_accidents.head()
df_accidents = df_accidents.reset_index()
df_accidents = df_accidents.drop('date',axis=1)
df_accidents.head()
plt.figure(figsize=(10,6))

sns.heatmap(df_accidents.corr().round(2),annot=True)
df_accidents = df_accidents.drop(['deadafter30d','quarter'],axis=1)
df_accidents.head()
df_accidents.columns
df_accidents.columns = ['hour', 'day_of_week', 'where', 'how', 'illumination',

       'roadtype', 'refnismun', 'death', 'day_of_month', 'month', 'year']
df_accidents.head()
df_accidents['year'] = df_accidents['year']-2000
df_accidents.head()
df_accidents = df_accidents.astype(int)
df_accidents.head()
df_accidents['refnismun'].unique()
plt.figure(figsize=(10,6))

sns.heatmap(df_accidents.corr().round(2),annot=True)
# from sklearn.feature_selection import RFE #recursive feature elimination (RFE)

# from sklearn.svm import SVR # Support Vector Regression

# X = df_accidents.drop('death',axis=1)

# X = X[:7000]

# y = df_accidents['death']

# y = y[:7000]

# print(X.shape, y.shape)
# estimator = SVR(kernel="linear")

# selector = RFE(estimator, n_features_to_select=5, step=1)
# selector = selector.fit(X, y)

# selector.support_
#  selector.ranking_
df_accidents = df_accidents.drop('year',axis=1)
df_accidents.head()
plt.figure(figsize=(10,6))

sns.heatmap(df_accidents.corr().round(2),annot=True)
df = df_accidents
sns.countplot(df['death'])
df['death'].value_counts()/len(df)
X = df.drop('death', axis=1).values

y = df['death'].values
from imblearn.over_sampling import RandomOverSampler



# define oversampling strategy



oversample = RandomOverSampler(sampling_strategy='minority')



# fit and apply the transform



X_over, y_over = oversample.fit_resample(X, y)



from collections import Counter
# summarize class distribution



print(Counter(y))



# summarize class distribution



print(Counter(y_over))
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

sns.countplot(df['death'], ax=ax1).set_title('Before')

sns.countplot((y_over), ax=ax2).set_title('After')
X = X_over

y = y_over



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=101)
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

scaler.fit(X_train)



X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
logmodel_over = LogisticRegression()

logmodel_over.fit(X_train, y_train)

rfc_over = RandomForestClassifier(n_estimators=100, verbose=100)

rfc_over.fit(X_train,y_train)
#Predictions and Evaluations



from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import plot_confusion_matrix
#Model performance on train dataset



training_predictions = logmodel_over.predict(X_train)



print(classification_report(y_train, training_predictions))

print(confusion_matrix(y_train, training_predictions))



plot_confusion_matrix(logmodel_over, X_train, y_train)
plot_confusion_matrix(logmodel_over, X_train, y_train,normalize='true')
#Model performance on test dataset



test_predictions = logmodel_over.predict(X_test)



print(classification_report(y_test, test_predictions))

print(confusion_matrix(y_test, test_predictions))



plot_confusion_matrix(logmodel_over, X_test, y_test)
plot_confusion_matrix(logmodel_over, X_test, y_test,normalize='true')
#Model performance on train dataset



training_predictions = rfc_over.predict(X_train)



print(classification_report(y_train, training_predictions))

print(confusion_matrix(y_train, training_predictions))



plot_confusion_matrix(rfc_over, X_train, y_train)
#Model performance on test dataset



test_predictions = rfc_over.predict(X_test)



print(classification_report(y_test, test_predictions))

print(confusion_matrix(y_test, test_predictions))



plot_confusion_matrix(rfc_over, X_test, y_test)
X_test = df_accidents.drop('death', axis=1).values

y_test = df_accidents['death'].values
predictions_log_reg = logmodel_over.predict(X_test)

predictions_rf = rfc_over.predict(X_test)
print(classification_report(y_test, predictions_log_reg))



print(confusion_matrix(y_test, predictions_log_reg))



plot_confusion_matrix(logmodel_over, X_test, y_test)

print(classification_report(y_test, predictions_rf))



print(confusion_matrix(y_test, predictions_rf))



plot_confusion_matrix(rfc_over, X_test, y_test)
X = df.drop('death', axis=1).values

y = df['death'].values
from imblearn.under_sampling import RandomUnderSampler



# define undersampling strategy



oversample = RandomUnderSampler(sampling_strategy='majority')



# fit and apply the transform



X_over, y_over = oversample.fit_resample(X, y)



from collections import Counter
# summarize class distribution



print(Counter(y))



# summarize class distribution



print(Counter(y_over))
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

sns.countplot(df['death'], ax=ax1).set_title('Before')

sns.countplot((y_over), ax=ax2).set_title('After')
X = X_over

y = y_over



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=101)
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

scaler.fit(X_train)



X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
logmodel_under = LogisticRegression()

logmodel_under.fit(X_train, y_train)



rfc_under = RandomForestClassifier(n_estimators=100,verbose=100)

rfc_under.fit(X_train,y_train)
#Model performance on train dataset



training_predictions = logmodel_under.predict(X_train)



print(classification_report(y_train, training_predictions))

print(confusion_matrix(y_train, training_predictions))



plot_confusion_matrix(logmodel_under, X_train, y_train)
#Model performance on test dataset



test_predictions = logmodel_under.predict(X_test)



print(classification_report(y_test, test_predictions))

print(confusion_matrix(y_test, test_predictions))



plot_confusion_matrix(logmodel_under, X_test, y_test)
#Model performance on train dataset



training_predictions = rfc_under.predict(X_train)



print(classification_report(y_train, training_predictions))

print(confusion_matrix(y_train, training_predictions))



plot_confusion_matrix(rfc_under, X_train, y_train)
#Model performance on test dataset



test_predictions = rfc_under.predict(X_test)



print(classification_report(y_test, test_predictions))

print(confusion_matrix(y_test, test_predictions))



plot_confusion_matrix(rfc_under, X_test, y_test)
X_test = df_accidents.drop('death', axis=1).values

y_test = df_accidents['death'].values
predictions_log_reg = logmodel_under.predict(X_test)

predictions_rf = rfc_under.predict(X_test)
print(classification_report(y_test, predictions_log_reg))
print(confusion_matrix(y_test, predictions_log_reg))
plot_confusion_matrix(logmodel_under, X_test, y_test)
print(classification_report(y_test, predictions_rf))



print(confusion_matrix(y_test, predictions_rf))


plot_confusion_matrix(rfc_under, X_test, y_test)