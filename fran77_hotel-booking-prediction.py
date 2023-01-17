# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')
data.head()
data.columns
plt.figure(figsize=(10,6))

g = sns.countplot(data.hotel)
plt.figure(figsize=(10,6))

g = sns.countplot(data.is_canceled)
plt.figure(figsize=(15,10))

g = sns.kdeplot(data.lead_time)
data.lead_time.describe()
plt.figure(figsize=(10,6))

g = sns.countplot(data.arrival_date_year)
# Order by month

months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

data['arrival_date_month'] = pd.Categorical(data['arrival_date_month'], categories=months, ordered=True)
plt.figure(figsize=(15,6))

g = sns.countplot(data.sort_values(by='arrival_date_month').arrival_date_month)
# Peak in the summer : July and August
plt.figure(figsize=(15,6))

g = sns.countplot(data.arrival_date_week_number)
# Peak between weeks 28 and 34, and week 53 for Winter holidays
plt.figure(figsize=(15,6))

g = sns.countplot(data.arrival_date_day_of_month)
plt.figure(figsize=(15,6))

g = sns.countplot(data.stays_in_weekend_nights)
plt.figure(figsize=(15,6))

g = sns.countplot(data.stays_in_week_nights)
# Less number of nights after 5, which corresponds to 1 week
data['total_days'] = data['stays_in_weekend_nights'] + data['stays_in_week_nights']
plt.figure(figsize=(15,6))

g = sns.countplot(data.total_days)
plt.figure(figsize=(15,6))

g = sns.countplot(data.adults)
# > 10 : groups
plt.figure(figsize=(15,6))

g = sns.countplot(data.children)
# A lot of couples
plt.figure(figsize=(15,6))

g = sns.barplot(x="adults", y="children", data=data, dodge=False)
# Exploring children without parents

len(data[(data.adults == 0) & (data.children > 1)])
data[(data.adults == 0) & (data.children > 1)].hotel.value_counts()
# Only City Hotels
data[(data.adults == 0) & (data.children > 1)].arrival_date_month.value_counts()
# Especially on summer and for the winter holidays
data[(data.adults == 0) & (data.children > 1)].total_days.value_counts()
# Stay 3 or 4 days
data[(data.adults == 0) & (data.children > 1)].total_of_special_requests.value_counts()
# Not a lot of special requests
plt.figure(figsize=(15,6))

g = sns.countplot(data.babies)
plt.figure(figsize=(15,6))

g = sns.countplot(data.meal)
data.meal.value_counts() / len(data) * 100
# Majority of Bed & Breakfast (77.3%)

# 12.1 % of Half board (breakfast and one other meal – usually dinner)

# 8.9% of No Meal

# 0.6% of Full board (breakfast, lunch and dinner)
plt.figure(figsize=(15,8))

g = sns.countplot(x='country',data=data, order = data['country'].value_counts().iloc[:10].index)
data.country.value_counts()[:10] / len(data) * 100
# Majority from Portugal, we guess the hotels are in Portugal
plt.figure(figsize=(15,6))

g = sns.countplot(data.market_segment)
data.market_segment.value_counts() / len(data) * 100
# Majority of Online Travel Agent
plt.figure(figsize=(15,6))

g = sns.countplot(data.distribution_channel)
data.distribution_channel.value_counts().nlargest(5) / len(data) * 100
# 82% of Travel Agent / Tour Operators
plt.figure(figsize=(15,6))

g = sns.countplot(data.is_repeated_guest)
plt.figure(figsize=(15,6))

g = sns.countplot(data.previous_cancellations)
data.previous_bookings_not_canceled.value_counts().nlargest(10) / len(data) * 100
plt.figure(figsize=(15,6))

g = sns.countplot(data.reserved_room_type)
plt.figure(figsize=(15,6))

g = sns.countplot(data.assigned_room_type)
plt.figure(figsize=(15,6))

g = sns.countplot(data.booking_changes)
plt.figure(figsize=(15,6))

g = sns.countplot(data.deposit_type)
# The majority did not make a deposit
data.agent.value_counts().nlargest(5) / len(data) * 100
data.company.value_counts().nlargest(5) / len(data) * 100
data.days_in_waiting_list.value_counts().nlargest(5) / len(data) * 100
# In 97% of the time, the booking is confirmed the same day
plt.figure(figsize=(15,6))

g = sns.countplot(data.customer_type)
# Majority of Transient (when the booking is not part of a group or contract, and is not associated to other transient booking)
plt.figure(figsize=(15,10))

g = sns.kdeplot(data.adr)
data.adr.describe()
plt.figure(figsize=(15,6))

g = sns.countplot(data.required_car_parking_spaces)
plt.figure(figsize=(15,6))

g = sns.countplot(data.total_of_special_requests)
plt.figure(figsize=(15,6))

g = sns.countplot(data.reservation_status)
data.head()
# lead_time is right skew, we will normalize it



# Normalize features columns

# Models performe better when values are close to normally distributed

from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()
data['lead_time'] = scaler.fit_transform(data['lead_time'].values.reshape(-1, 1))

data['adr'] = scaler.fit_transform(data['adr'].values.reshape(-1, 1))
# Convert to categorical values

data['arrival_date_month'] = data.arrival_date_month.astype('category').cat.codes

data['meal'] = data.meal.astype('category').cat.codes

data['country'] = data.country.astype('category').cat.codes

data['market_segment'] = data.market_segment.astype('category').cat.codes

data['distribution_channel'] = data.distribution_channel.astype('category').cat.codes

data['reserved_room_type'] = data.reserved_room_type.astype('category').cat.codes

data['assigned_room_type'] = data.assigned_room_type.astype('category').cat.codes

data['deposit_type'] = data.deposit_type.astype('category').cat.codes

data['customer_type'] = data.customer_type.astype('category').cat.codes

data['reservation_status'] = data.reservation_status.astype('category').cat.codes

data['hotel'] = data.hotel.astype('category').cat.codes
# Fill NA to 0

data.isnull().sum(axis = 0)
data['children'] = data.children.fillna(0) # replace the 4 nan with 0

data['agent'] = data.agent.fillna(0)

data['company'] = data.company.fillna(0)
data.head()
# Remove columns not important

data = data.drop(["arrival_date_year", "reservation_status_date"], axis=1)
# Get columns with at least 0.2 correlation

data_corr = data.corr()['is_canceled']

cols = data_corr[abs(data_corr) > 0.1].index.tolist()

data = data[cols]
# plot the heatmap

data_corr = data.corr()

plt.figure(figsize=(10,8))

sns.heatmap(data_corr, 

        xticklabels=data_corr.columns,

        yticklabels=data_corr.columns, cmap=sns.diverging_palette(220, 20, n=200))
data.corr()['is_canceled'].sort_values(ascending=False)
# Too much correlation

data = data.drop('reservation_status', 1)
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn import tree

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV
X = data.drop("is_canceled", axis=1)

Y = data["is_canceled"]
# Split 20% test, 80% train



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)
# Logistic Regression



log = LogisticRegression(max_iter=1000)

log.fit(X_train, Y_train)

Y_pred_log = log.predict(X_test)

acc_log = accuracy_score(Y_pred_log, Y_test)

acc_log
t = tree.DecisionTreeClassifier()



# search the best params

grid = {'min_samples_split': [5, 10, 20, 50, 100]},



clf_tree = GridSearchCV(t, grid, cv=10)

clf_tree.fit(X_train, Y_train)



Y_pred_tree = clf_tree.predict(X_test)



# get the accuracy score

acc_tree = accuracy_score(Y_pred_tree, Y_test)

print(acc_tree)
clf_tree.best_params_
rf = RandomForestClassifier()



# search the best params

grid = {'n_estimators':[100,200], 'max_depth': [2,5,10]}



clf_rf = GridSearchCV(rf, grid, cv=10)

clf_rf.fit(X_train, Y_train)



Y_pred_rf = clf_rf.predict(X_test)

# get the accuracy score

acc_rf = accuracy_score(Y_pred_rf, Y_test)

print(acc_rf)
clf_rf.best_params_
# The best model is Decision Tree 