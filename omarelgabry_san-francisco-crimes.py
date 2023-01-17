%matplotlib inline
# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

# csv
import csv
import gzip

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
import xgboost as xgb
# get crimes & test csv files as a DataFrame
crimes_df  = pd.read_csv('../input/train.csv')
test_df    = pd.read_csv('../input/test.csv')

# preview the data
crimes_df.head()
crimes_df.info()
print("----------------------------")
test_df.info()
# drop unnecessary columns, these columns won't be useful in analysis and prediction
crimes_df = crimes_df.drop(['Descript','Resolution','Address', 'X', 'Y'], axis=1)
test_df   = test_df.drop(['Address', 'X', 'Y'], axis=1)
# Category

# Plot
crimes_df["Category"].value_counts().plot(kind='barh', figsize=(15,10))
# Dates

# Convert Date to Year, Month, and Week
crimes_df['Year']  = crimes_df['Dates'].apply(lambda x: int(str(x)[:4]))
crimes_df['Month'] = crimes_df['Dates'].apply(lambda x: int(str(x)[5:7]))

test_df['Year']  = test_df['Dates'].apply(lambda x: int(str(x)[:4]))
test_df['Month'] = test_df['Dates'].apply(lambda x: int(str(x)[5:7]))
# .... continue with Dates

# Assign Dates column to Dates(Year-Month) instead of (Year-Month-Day)
# this column will be useful in analysis and visualization
crimes_df['Dates'] = crimes_df['Dates'].apply(lambda x: (str(x)[:7]))
test_df['Dates']   = test_df['Dates'].apply(lambda x: (str(x)[:7]))

# count number of crimes by Dates(Year-Month)
count_crimes  = crimes_df.groupby('Dates').size()
# crimes_pct_change  = crimes_df.groupby('Dates').size().pct_change()

fig, (axis1) = plt.subplots(1,1,sharex=True,figsize=(25,10))

# plot number of crimes over time(year-month)
ax1 = count_crimes.plot(legend=True,ax=axis1,marker='o')
ax1.set_xticks(range(len(count_crimes)))
xlabels = ax1.set_xticklabels(count_crimes.index.tolist(), rotation=80)
# year
# Which year has higher number of crimes

# Plot
fig, (axis1, axis2) = plt.subplots(2,1,figsize=(15,8))

sns.countplot(x=crimes_df["Year"], ax=axis1)

# most frequent crimes in a year
crimes_df["Category"][crimes_df["Year"] == 2003].value_counts().plot(kind='barh', ax=axis2, ylim=(0,10))
# month
# Which month has higher number of crimes

# Plot
fig, (axis1) = plt.subplots(1,1,figsize=(15,5))

sns.countplot(x=crimes_df["Month"], order=[1,2,3,4,5,6,7,8,9,10,11,12], ax=axis1)
# Create dummy varibales for DayOfWeek
day_dummies_crimes  = pd.get_dummies(crimes_df['DayOfWeek'])
day_dummies_crimes.drop(['Sunday'], axis=1, inplace=True)

day_dummies_test  = pd.get_dummies(test_df['DayOfWeek'])
day_dummies_test.drop(['Sunday'], axis=1, inplace=True)

crimes_df   = crimes_df.join(day_dummies_crimes)
test_df     = test_df.join(day_dummies_test)

# Plot
# Which DayOfWeek has higher number of crimes
fig, (axis1) = plt.subplots(1,1,figsize=(15,5))
sns.countplot(x=crimes_df["DayOfWeek"], ax=axis1)

crimes_df.drop(['DayOfWeek'], axis=1,inplace=True)
test_df.drop(['DayOfWeek'], axis=1,inplace=True)
# Create dummy varibales for PdDistrict
day_dummies_crimes  = pd.get_dummies(crimes_df['PdDistrict'])
day_dummies_crimes.drop(['RICHMOND'], axis=1, inplace=True)

day_dummies_test  = pd.get_dummies(test_df['PdDistrict'])
day_dummies_test.drop(['RICHMOND'], axis=1, inplace=True)

crimes_df   = crimes_df.join(day_dummies_crimes)
test_df     = test_df.join(day_dummies_test)

# Plot
# Which PdDistrict has higher number of crimes
fig, (axis1, axis2) = plt.subplots(2,1,figsize=(15,8))
sns.countplot(x=crimes_df["PdDistrict"], ax=axis1)

# most frequent crimes in a PdDistrict
crimes_df["Category"][crimes_df["PdDistrict"] == 'NORTHERN'].value_counts().plot(kind='barh', ax=axis2, ylim=(0,10))

crimes_df.drop(['PdDistrict'], axis=1,inplace=True)
test_df.drop(['PdDistrict'], axis=1,inplace=True)
# drop Dates
crimes_df.drop(['Dates'], axis=1,inplace=True)
test_df.drop(['Dates'], axis=1,inplace=True)

# drop Year and Month
# crimes_df.drop(["Year", "Month"], axis=1, inplace=True)
# test_df.drop(["Year", "Month"], axis=1, inplace=True)
ordered_crimes   = crimes_df["Category"].value_counts().index.order()
map_values_dic   = Series(range(0,len(ordered_crimes)), index=ordered_crimes).to_dict()

crimes_df['Category'] = crimes_df['Category'].map(map_values_dic)
# define training and testing sets

X_train = crimes_df.drop(["Category"],axis=1)
Y_train = crimes_df["Category"].copy()
X_test  = test_df.drop("Id",axis=1).copy()
# Xgboost 

params = {"objective": "multi:softprob", "num_class": len(ordered_crimes)}

T_train_xgb = xgb.DMatrix(X_train, Y_train)
X_test_xgb  = xgb.DMatrix(X_test)

gbm = xgb.train(params, T_train_xgb, 20)
Y_pred = gbm.predict(X_test_xgb)
# Create submission

i = 0

# Create column name based on ordered crime names(see sample_submission.csv)
with gzip.open('crimes.csv.gz', 'wt') as outf:
    fo = csv.writer(outf, lineterminator='\n')
    fo.writerow(['Id'] + list(ordered_crimes))
    
    for i, pred in enumerate(Y_pred):
        fo.writerow([i] + list(pred))