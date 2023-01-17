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
import pandas as pd
import numpy as np
df= pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')
df.head(4)
df.describe()
df.head(2)
df.drop(['arrival_date_week_number','meal','agent','company','adr','reservation_status','reservation_status_date'],axis=1,inplace=True)

df.drop(['country'],axis=1,inplace=True)
df.head(2)
df.isnull().sum(axis = 0)
df.shape

(df[df['children']!= 0]['children'].sum())/(df[df['children']!= 0]['children'].count())
df[df['children'].isnull()]['children']
#replacing null values with 1.
df['children'] = df.children.fillna(1)
df[df['children'].isnull()]['children']
#No more null values
df.isnull().sum(axis = 0)
df[df['children']!= 0]['children'].sum()
#we have to reset the index after dropping null values
df = df.reset_index(drop=True)
df.shape
#No more null records
df.isnull().sum(axis = 0)
df.head(2)
df['arrival_date_month'].unique()
month_dict={'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6,'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12 }
month_dict
a=df['arrival_date_month'].map(month_dict)
df['arrival_date_month']=a
df.head(3)
#drop_first=True will avoid dummy variable trap
cols_to_transform = [ 'hotel','market_segment', 'distribution_channel', 'reserved_room_type','assigned_room_type','deposit_type','customer_type', ]
df_with_dummies = pd.get_dummies(df,columns = cols_to_transform,drop_first=True)
df_with_dummies.head(3)
from sklearn.model_selection import train_test_split

X = df_with_dummies[df_with_dummies.loc[:, df_with_dummies.columns != 'is_canceled'].columns]
y = df_with_dummies['is_canceled']
X
y

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,test_size=0.2, random_state=0)
X_train.shape
#here the values should between 0 and 1. We will use MinMax scaler

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train.values)
X_test_scaled = scaler.fit_transform(X_test.values)
#y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1,1))
#y_test_scaled = scaler.fit_transform(y_test.values.reshape(-1,1))
X_train=X_train_scaled
X_test=X_test_scaled
y_train.shape
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)

# Model performance
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train, y_train, cv=10)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


from sklearn import ensemble
clf = ensemble.RandomForestClassifier()
print(clf.get_params())
#Finding the best params using grid search cv
from sklearn.model_selection import GridSearchCV


params = {
    'bootstrap': [False],
    'max_depth': [5, 10],
    'max_features': ['auto'],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [2, 4],
    'n_estimators': [100, 250]}

gsv = GridSearchCV(clf, params, cv=2, n_jobs=-1, scoring='accuracy')
gsv.fit(X_train, y_train)

gsv.best_estimator_.feature_importances_
print(gsv.best_params_)

print(gsv.best_estimator_)
from sklearn.metrics import accuracy_score
def evaluate(mdl, X, y):
    predictions = mdl.predict(X)
    errors = abs(y - predictions)
    acc_score = accuracy_score(y, predictions)
    print('Model Performance')
    print('Average Error: {:0.3f}.'.format(np.mean(errors)))
    print('Accuracy = {:0.3f}%.'.format(acc_score))    
    return acc_score
#We were able to achieve an accuracy of 81.3% using RF classifier.
gsv_accuracy = evaluate(gsv.best_estimator_, X_test, y_test)
print(gsv_accuracy)
from sklearn.svm import SVC
from sklearn import metrics
svc=SVC() #Default hyperparameters
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))
#Cross validation on linear kernel

from sklearn.cross_validation import cross_val_score
svc=SVC(kernel='linear')
scores = cross_val_score(svc, X_train, y_train, cv=5, scoring='accuracy') #cv is cross validation
print(scores)
#Cross validation on rbf kernel
svc=SVC(kernel='rbf')
scores = cross_val_score(svc, X_train, y_train, cv=5, scoring='accuracy') #cv is cross validation
print(scores)
##Cross validation on poly kernel

svc=SVC(kernel='poly')
scores = cross_val_score(svc, X_train, y_train, cv=5, scoring='accuracy') #cv is cross validation
print(scores)

#Implementing Naive Bayes claissifire

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))
#Implementing KNeighbors ClaSSifier from sklearn
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))