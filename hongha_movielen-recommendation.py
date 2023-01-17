import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sb

from scipy import sparse 

import datetime 

from sklearn.model_selection import train_test_split

import sklearn.model_selection as model_selection

from pandas import DataFrame

from sklearn import preprocessing
ratings=pd.read_csv("../input/movielens-100k-dataset/ml-100k/u.data", sep='\t',names=['user_id','item_id','rating','timestamp'])

ratings.head()
users=pd.read_csv("../input/movielens-100k-dataset/ml-100k/u.user", sep='|', 

                  names=['user_id','age','gender','occupation','zip_code'])

users
# plt.hist(data=users,x=users.Gender)

# plt.show()
movies = pd.read_csv("../input/movielens-100k-dataset/ml-100k/u.item", sep='|', 

                      names=['movie_id','movie_title','release_date','video_release_date','IMDb_URL','unknown','Acton','Adventure','Animation',

                             'Childrens','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mestery','Romance',

                             'Sci-Fi','Thriller','War','Western'],

                      encoding='ISO-8859-1')

movies.head()
all_data = ratings.merge(users,left_on='user_id', right_on='user_id')

all_data = all_data.merge(movies, left_on='item_id', right_on='movie_id')



all_data.head()

categorical_var = ['occupation','gender']

numerical_var = ['timestamp','age','release_date']
# all_data['release_date'].dtype
all_data.drop(columns=['zip_code','video_release_date','user_id','item_id','movie_title','IMDb_URL','movie_id'], inplace=True)
# age_group = users.groupby('age').size()

# age_group

gender_group = users.groupby('gender').size()

gender_group
plt.hist(x=users.gender)

plt.show()
# R_data = all_data[all_data.movie_id]

# R_data.head(10)
# all_data['movie_id'].groupby(all_data['rating']).size()
sb.countplot(all_data['rating'])

plt.show()
# plt.hist(all_data['age'])

# plt.show()

sb.kdeplot(all_data['age'])

plt.show()

all_data.age.describe()
# sb.kdeplot(all_data['release_date'])
all_data = pd.get_dummies(all_data,columns= categorical_var)
all_data.head()
all_data['release_date'] = pd.to_datetime(all_data['release_date'], infer_datetime_format=True)

all_data['release_date'].unique()
all_data['release_date']=(all_data['release_date']-pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
all_data['release_date'].fillna(all_data['release_date'].mean(), inplace=True)

all_data['release_date'].unique()
# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()

# all_data[numerical_var] = scaler.fit_transform(all_data[numerical_var])

# all_data[numerical_var]
sb.kdeplot(all_data['release_date'])
x=all_data.drop(columns=['rating'])

y=all_data['rating']

# x = all_data.iloc[:,1:]

# y = all_data.iloc[:,0:1]

# print(x_train)

# print(y_lable)

# all_data.iloc[:,1:]

# all_data.iloc[:,0:1]
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x, y, train_size=0.8,stratify=y,test_size=0.2, random_state=42)

print('Train shape: ', X_train.shape)

print('Test shape: ', X_test.shape)
X_train, X_valid, Y_train, Y_valid=model_selection.train_test_split(X_train, Y_train, stratify=Y_train, random_state=42, test_size=0.2)

print('Train shape: ', X_train.shape)

print('Valid shape: ', X_valid.shape)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train=scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)

X_valid=scaler.transform(X_valid)
# from sklearn.svm import SVC

# from sklearn.metrics import accuracy_score

# # from sklearn.metrics import f1_score

# # from sklearn.metrics import classification_report

# # import sklearn.metrics as metrics

# # for c in [0.1, 0.5, 1.0]:

# #     model=LinearSVC(C=c, random_state=0,max_iter=10000)

# #     model=SVC(C=1, random_state=0,max_iter=10000)

# SVC=SVC(C=1,random_state=0)

# SVC.fit(X_train,Y_train)

# X_valid_LSVC_model = SVC.predict(X_valid)

# #     print("c: ",c)

# print("SVC X_valid: ",accuracy_score(Y_valid,X_valid_LSVC_model))

# #     print("LSVC f1_score_valid :", f1_score(Y_valid, X_valid_LSVC_model, average="micro"))

# #     print ("classification_report", metrics.classification_report(Y_valid,X_valid_LSVC_model))

# X_train_LSVC_model = SVC.predict(X_train)

# #     print("c: ",c)

# print("SVC X_valid: ",accuracy_score(Y_train,X_train_LSVC_model))

# #     print("LSVC f1_score_train :", f1_score(Y_train, X_train_LSVC_model, average="micro"))

# #     print ("classification_report", metrics.classification_report(Y_train,X_train_LSVC_model))
# from sklearn.metrics import classification_report

# from sklearn.svm import LinearSVC

# model=LinearSVC(C=1, random_state=0,max_iter=10000)

# clf=model.fit(x,y)

# predicted = cross_val_predict(clf, x, y, cv=2)

# print ("classification_report", metrics.classification_report(y, predicted))
from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.svm import LinearSVC

from sklearn.metrics import f1_score

from sklearn.metrics import classification_report

import sklearn.metrics as metrics

for c in [0.1]:

    print("c: ",c)

    model=LinearSVC(C=c, random_state=0,max_iter=1000)

    model.fit(X_train,Y_train)

    predict_X_train = model.predict(X_train)

    predict_X_valid = model.predict(X_valid)

    predict_X_test = model.predict(X_test)

    print("LSVC f1_score_valid :", f1_score(Y_valid, predict_X_valid, average="micro"))

    print ("classification_report", metrics.classification_report(Y_valid,predict_X_valid))

    print("LSVC f1_score_train :", f1_score(Y_train, predict_X_train, average="micro"))

    print ("classification_report", metrics.classification_report(Y_train,predict_X_train))

    print("LSCV f1_score_test: ",f1_score(Y_test,predict_X_test, average="micro"))

    print ("classification_report: ", metrics.classification_report(Y_test,predict_X_test))

    

    rms_train = sqrt(mean_squared_error(Y_train, predict_X_train))

    print('RMSE train: ',rms_train)

    rms_valid = sqrt(mean_squared_error(Y_valid, predict_X_valid))

    print('RMSE valid: ',rms_valid)

    rms_test = sqrt(mean_squared_error(Y_test, predict_X_test))

    print('RMSE test: ',rms_test)
# from sklearn import datasets, linear_model, metrics 

# reg = linear_model.LinearRegression() 

# reg.fit(X_train, Y_train) 

# print('Coefficients: \n', reg.coef_) 

# print('Variance score: {}'.format(reg.score(X_test, Y_test))) 

# plt.style.use('fivethirtyeight') 

  

# ## plotting residual errors in training data 

# plt.scatter(reg.predict(X_train), reg.predict(X_train) - Y_train, 

#             color = "green", s = 10, label = 'Train data') 

  

# ## plotting residual errors in test data 

# plt.scatter(reg.predict(X_test), reg.predict(X_test) - Y_test, 

#             color = "blue", s = 10, label = 'Test data') 

  

# ## plotting line for zero residual error 

# plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2) 

  

# ## plotting legend 

# plt.legend(loc = 'upper right') 

  

# ## plot title 

# plt.title("Residual errors") 

  

# ## function to show plot 

# plt.show() 
from sklearn.metrics import mean_squared_error

from math import sqrt

rms_train = sqrt(mean_squared_error(Y_train, predict_X_train))

print('RMSE train: ',rms_train)

rms_valid = sqrt(mean_squared_error(Y_valid, predict_X_valid))

print('RMSE valid: ',rms_valid)

rms_test = sqrt(mean_squared_error(Y_test, predict_X_test))

print('RMSE test: ',rms_test)