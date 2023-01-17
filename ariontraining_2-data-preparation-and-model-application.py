import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

import time



pd.set_option('display.max_columns', 500)

pd.set_option('display.max_colwidth', 500)

pd.set_option('display.max_rows', 1000)



import warnings

warnings.filterwarnings('ignore')

%matplotlib inline 
train = pd.read_csv("../input/combination.csv")
train.head()
#There is no common tube_assembly_id between train and test data. So we drop this variable.

train.drop("tube_assembly_id", axis=1, inplace=True)
train.head().transpose()
train.quote_date = pd.to_datetime(train.quote_date)
#add new numeric time features



train["year"] = train.quote_date.dt.year

train["month"] = train.quote_date.dt.month

train["day"] = train.quote_date.dt.day

train["day_of_week"] = train.quote_date.dt.dayofweek
#only use numeric data

data = train.select_dtypes(include=['int', 'float'])
#fill null by 0

data.replace(np.nan, 0, inplace=True)
train_data, valid_data = train_test_split(data, test_size = 0.3)
label = "cost"
data_labels = train_data.columns.tolist()

data_labels.remove(label)
train_df = train_data[data_labels]

valid_df = valid_data[data_labels]

train_label = train_data[label]

valid_label = valid_data[label]
#define a evaluation function



def rmse_score(preds, true):

    rmse_score = (np.sum((np.log1p(preds)-np.log1p(true))**2)/len(true))**0.5

    return rmse_score
# define a function for comparing predictions and true data.



def compare_result(preds, true):

    compare = pd.DataFrame({"test_id": true.index,

                           "real_cost": true,

                           "pred_cost": preds})

    compare = compare[["test_id", "real_cost", "pred_cost"]].reset_index(drop=True)

    

    compare["error_percent_(%)"] = np.abs(compare.real_cost - compare.pred_cost) / compare.real_cost * 100

    

    return compare
# sklearn LinearRegression

# Preprocess: do feature scaling or not 

from sklearn.linear_model import LinearRegression



def linear_learning(labels, train, test, preprocess):

    

    if preprocess == False:

        label_log=np.log1p(labels)

        linear=LinearRegression()

        model=linear.fit(train, label_log)

        preds1=model.predict(test)

        preds=np.expm1(preds1)

        

    elif preprocess == True:        

        train = preprocessing.scale(train)

        test = preprocessing.scale(test)

        

        label_log=np.log1p(labels)

        linear=LinearRegression()

        model=linear.fit(train, label_log)

        preds1=model.predict(test)

        preds=np.expm1(preds1)

        

    return preds
# sklearn svm regression 

# Preprocess: do feature scaling or not

from sklearn import svm



def svm_learning(labels, train, test, preprocess):

    

    if preprocess == False:    

        label_log=np.log1p(labels)

        clf=svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma="auto",

            kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

        model=clf.fit(train, label_log)

        preds1=model.predict(test)

        preds=np.expm1(preds1)

        

    elif preprocess == True:

        

        train = preprocessing.scale(train)

        test = preprocessing.scale(test)

        

        label_log=np.log1p(labels)

        clf=svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma="auto",

            kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

        model=clf.fit(train, label_log)

        preds1=model.predict(test)

        preds=np.expm1(preds1)

    return preds
# sklearn random forest regression

# Preprocess: do feature scaling or not

from sklearn.ensemble import RandomForestRegressor



def rf_learning(labels, train, test, preprocess):

    

    if preprocess == False:

        label_log=np.log1p(labels)

        clf=RandomForestRegressor(n_estimators=50, n_jobs=-1)

        model=clf.fit(train, label_log)

        preds1=model.predict(test)

        preds=np.expm1(preds1)

        

    elif preprocess == True:

        

        train = preprocessing.scale(train)

        test = preprocessing.scale(test)

        

        label_log=np.log1p(labels)

        clf=RandomForestRegressor(n_estimators=50, n_jobs=-1)

        model=clf.fit(train, label_log)

        preds1=model.predict(test)

        preds=np.expm1(preds1)

    return preds
# K-nearest neighbor regression

# Preprocess: do feature scaling or not

from sklearn.neighbors import KNeighborsRegressor



def knn_learning(labels, train, test, n, preprocess):

    

    if preprocess == False:

        label_log=np.log1p(labels)

        clf=KNeighborsRegressor(n_neighbors=n, n_jobs=-1)

        model=clf.fit(train, label_log)

        preds1=model.predict(test)

        preds=np.expm1(preds1)

        

    elif preprocess == True:

        

        train = preprocessing.scale(train)

        test = preprocessing.scale(test)

        

        label_log=np.log1p(labels)

        clf=KNeighborsRegressor(n_neighbors=n, n_jobs=-1)

        model=clf.fit(train, label_log)

        preds1=model.predict(test)

        preds=np.expm1(preds1)

    return preds
start = time.time()



#linear_preds = linear_learning(labels=train_label, train=train_df, test=valid_df, preprocess=False)

linear_preds = linear_learning(labels=train_label, train=train_df, test=valid_df, preprocess=True)



rmse_linear = rmse_score(linear_preds, valid_label)

print ("Linear RMSLE is : {}".format(rmse_linear))



compare_linear = compare_result(preds=linear_preds, true=valid_label)



end = time.time()

duration = end - start

print ("It takes {} seconds".format(duration))
start = time.time()



#svm_preds = svm_learning(train_label, train_df, valid_df, False)

svm_preds = svm_learning(train_label, train_df, valid_df, True)



rmse_svm = rmse_score(svm_preds, valid_label)

print ("SVM RMSLE is : {}".format(rmse_svm))



compare_svm = compare_result(preds=svm_preds, true=valid_label)



end = time.time()

duration = end - start

print ("It takes {} seconds".format(duration))
start = time.time()



#rf_preds = rf_learning(train_label, train_df, valid_df, False)

rf_preds = rf_learning(train_label, train_df, valid_df, True)



rmse_rf = rmse_score(rf_preds, valid_label)

print ("RF RMSLE is : {}".format(rmse_rf))



compare_rf = compare_result(preds=rf_preds, true=valid_label)



end = time.time()

duration = end - start

print ("It takes {} seconds".format(duration))
start = time.time()



#knn_preds = knn_learning(train_label, train_df, valid_df, 3, False)

knn_preds = knn_learning(train_label, train_df, valid_df, 3, True)



rmse_knn = rmse_score(knn_preds, valid_label)

print ("KNN RMSLE is : {}".format(rmse_knn))



compare_knn = compare_result(preds=svm_preds, true=valid_label)



end = time.time()

duration = end - start

print ("It takes {} seconds".format(duration))
compare_linear.head()
compare_svm.head()
compare_rf.head()
compare_knn.head()