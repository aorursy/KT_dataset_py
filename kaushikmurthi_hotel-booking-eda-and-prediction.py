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



# Ignore warnings

import warnings  

warnings.filterwarnings('ignore')
data = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')
data.head()
data.info()
data.describe().T
import matplotlib.pyplot as plt

import seaborn as sns
perc_missing_data = pd.DataFrame([data.isnull().sum(),data.isnull().sum()*100.0/data.shape[0]]).T

perc_missing_data.columns = ['No. of Missing Data', '% Missing Data']

perc_missing_data
data['children'].value_counts()
data['children'].fillna(0,inplace=True)
perc_country_data = pd.DataFrame([data['country'].value_counts(),data['country'].value_counts()*100/data.shape[0]]).T

perc_country_data.columns = ['Count', '% Distribution']

perc_country_data
data['country'].fillna('PRT',inplace=True)
data.drop(['agent','company'],axis=1,inplace=True)
perc_missing_data = pd.DataFrame([data.isnull().sum(),data.isnull().sum()*100.0/data.shape[0]]).T

perc_missing_data.columns = ['No. of Missing Data', '% Missing Data']

perc_missing_data
plt.style.use('fivethirtyeight')
plt.figure(figsize=(14,6))

sns.countplot(x='hotel',data=data,hue='is_canceled',palette='pastel')

plt.show()
plt.figure(figsize=(14,6))

sns.countplot(x='deposit_type',data=data,hue='is_canceled',palette='pastel')

plt.show()
data['arrival_date'] = data['arrival_date_year'].astype(str) + '-' + data['arrival_date_month'] + '-' + data['arrival_date_day_of_month'].astype(str)

data['arrival_date'] = data['arrival_date'].apply(pd.to_datetime)

data['reservation_status_date'] = data['reservation_status_date'].apply(pd.to_datetime)
cancelled_data = data[data['reservation_status'] == 'Canceled']

cancelled_data['canc_to_arrival_days'] = cancelled_data['arrival_date'] - cancelled_data['reservation_status_date']

cancelled_data['canc_to_arrival_days'] = cancelled_data['canc_to_arrival_days'].dt.days
plt.figure(figsize=(14,6))

sns.distplot(cancelled_data['canc_to_arrival_days'])

plt.show()
print('Percentage of cancellations that are within a week of arrival: ', 

      (cancelled_data[cancelled_data['canc_to_arrival_days']<=7]['canc_to_arrival_days'].count()*100/cancelled_data['canc_to_arrival_days'].count()).round(2), '%')
month_sorted = ['January','February','March','April','May','June','July','August','September','October','November','December']

plt.figure(figsize=(14,6))

sns.countplot(data['arrival_date_month'], palette='pastel', order = month_sorted)

plt.xticks(rotation = 90)

plt.show()
perc_monthly_canc = pd.DataFrame(data[data['is_canceled'] == 1]['arrival_date_month'].value_counts() * 100 / data['arrival_date_month'].value_counts())

perc_monthly_canc.reset_index()

plt.figure(figsize=(14,6))

sns.barplot(x=perc_monthly_canc.index,y='arrival_date_month',data=perc_monthly_canc, order=month_sorted, palette='pastel')

plt.xticks(rotation = 90)

plt.ylabel('% cancellation per month')

plt.show()
plt.figure(figsize=(8,8))

explode = [0.005] * len(cancelled_data['market_segment'].unique())

colors = ['royalblue','orange','y','darkgreen','gray','purple','red','lightblue']

plt.pie(cancelled_data['market_segment'].value_counts(),

       autopct = '%.1f%%',

       explode = explode,

       colors = colors)

plt.legend(cancelled_data['market_segment'].unique(), bbox_to_anchor=(-0.1, 1.),

           fontsize=14)

plt.title('Market Segment vs Cancelled Bookings')

plt.tight_layout()

plt.show()
plt.figure(figsize=(10,8))

data.corr()['is_canceled'].sort_values()[:-1].plot(kind='bar')

plt.show()
plt.figure(figsize=(16,12))

plt.subplot(221)

sns.countplot(data['meal'], hue=data['is_canceled'])

plt.xlabel('Meal Type')

plt.subplot(222)

sns.countplot(data['customer_type'], hue=data['is_canceled'])

plt.xlabel('Customer Type')

plt.subplot(223)

sns.countplot(data['reserved_room_type'], hue=data['is_canceled'])

plt.xlabel('Reserved Room Type')

plt.subplot(224)

sns.countplot(data['reservation_status'], hue=data['is_canceled'])

plt.xlabel('Reservation Status')

plt.show()
data = data.drop(['meal','country','reserved_room_type','assigned_room_type','deposit_type','reservation_status','reservation_status_date','arrival_date'], axis=1)

data = pd.concat([data, 

                 pd.get_dummies(data['hotel'], drop_first=True), 

                 pd.get_dummies(data['arrival_date_month'], drop_first=True), 

                 pd.get_dummies(data['market_segment'], drop_first=True),

                 pd.get_dummies(data['distribution_channel'], drop_first=True),

                 pd.get_dummies(data['customer_type'], drop_first=True)

                 ], axis=1)

data = data.drop(['hotel','arrival_date_month','market_segment','distribution_channel','customer_type'], axis=1)
data.info()
plt.figure(figsize=(16,8))

data.corr()['is_canceled'].sort_values()[:-1].plot(kind='bar')

plt.show()
X = data.iloc[:, 1:].values

y = data.iloc[:, 0].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
# Empty dictionary of model accuracy results

model_accuracy_results = {}



# Function for calculating accuracy from confusion matrix

from sklearn.metrics import confusion_matrix

def model_accuracy(y_test, y_pred):

    cm = confusion_matrix(y_test, y_pred)

    accuracy = ((cm[0,0] + cm [1,1]) * 100 / len(y_test)).round(2)

    return accuracy
# Baseline model

(unique, counts) = np.unique(y_train, return_counts=True)

if counts[0]  > counts[1]:

    idx = 0

else:

    idx = 1



# Applying baseline results to y_pred

if idx == 0:

    y_pred = np.zeros(y_test.shape)

else:

    y_pred = np.ones(y_test.shape)



# Computing accuracy

model_accuracy_results['Baseline'] = model_accuracy(y_test, y_pred)
# Fit and train

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0, max_iter=250)

classifier.fit(X_train, y_train)



# Predict

y_pred = classifier.predict(X_test)



# Computing accuracy

model_accuracy_results['LogisticRegression'] = model_accuracy(y_test, y_pred)
# Fit and train

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 10)

classifier.fit(X_train,y_train)



# Predict

y_pred = classifier.predict(X_test)



# Computing accuracy

model_accuracy_results['KNearestNeighbors'] = model_accuracy(y_test, y_pred)
# Fit and train

from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state=0)

classifier.fit(X_train,y_train)



# Predict

y_pred = classifier.predict(X_test)



# Computing accuracy

model_accuracy_results['SVM'] = model_accuracy(y_test, y_pred)
# Fit and train

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)

classifier.fit(X_train,y_train)



# Predict

y_pred = classifier.predict(X_test)



# Computing accuracy

model_accuracy_results['RandomForest'] = model_accuracy(y_test, y_pred)
df_model_accuracies = pd.DataFrame(list(model_accuracy_results.values()), index=model_accuracy_results.keys(), columns=['Accuracy'])

df_model_accuracies

# Grid Search

from sklearn.model_selection import GridSearchCV

parameters = [{'n_estimators': [10,25,50,100,500] , 'criterion': ['entropy', 'gini']}]

randomforestclassifier = RandomForestClassifier()

grid_search = GridSearchCV(estimator = randomforestclassifier,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           n_jobs = -1)

grid_search.fit(X_train, y_train)
print('Best Score: ', grid_search.best_score_.round(2))

print('Best Parameters: ', grid_search.best_params_)
# Fit and train

optimized_classifier = RandomForestClassifier(n_estimators=500, criterion='entropy', random_state=0)

optimized_classifier.fit(X_train,y_train)



# Predict

y_pred = optimized_classifier.predict(X_test)



# Computing accuracy

model_accuracy_results['OptimizedRandomForest'] = model_accuracy(y_test, y_pred)

df_model_accuracies = pd.DataFrame(list(model_accuracy_results.values()), index=model_accuracy_results.keys(), columns=['Accuracy'])

df_model_accuracies
orf_cm = confusion_matrix(y_test, optimized_classifier.predict(X_test))



names = ['True Neg','False Pos','False Neg','True Pos'] # list of descriptions for each group

values = [value for value in orf_cm.flatten()] # list of values for each group

percentages = [str(perc.round(2))+'%' for perc in orf_cm.flatten()*100/np.sum(orf_cm)] # list of percentages for each group

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(names,values,percentages)] # zip them into list of strings as labels

labels = np.asarray(labels).reshape(2,2)

sns.heatmap(orf_cm, annot=labels, fmt='', cmap='binary')