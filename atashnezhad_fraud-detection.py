import numpy as np
import pandas as pd
import time
import csv
import os
import seaborn as sns
import random
import gc

from sklearn import preprocessing
from matplotlib import pyplot as plt
import matplotlib as mpl
import scipy.stats as st
from sklearn import ensemble, tree, linear_model
import missingno as msno
import math
import copy
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier

from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from matplotlib import pyplot
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split 
import gc
from numpy import loadtxt
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
%matplotlib inline
def null_table(df):
    print("Training Data\n\n")
    print(pd.isnull(df).sum()) 

def some_useful_data_insight(df):
    Num_of_line = 100
    print(Num_of_line*'=')
    print(Num_of_line*'=')
    print(df.head(5))
    print(Num_of_line*'=')
    print(df.dtypes)
    print(Num_of_line*'=')
    print(null_table(df))
    print(Num_of_line*'=')
    print('data length=', len(df))
    print(Num_of_line*'=')
    
def Plot_Hist_column(df, x):
    pyplot.hist(df[x], log = True)
    pyplot.title(x)
    pyplot.show()
    
def Plot_Hist_columns(df, xlist):
    [Plot_Hist_column(df, x) for x in xlist]  
    pyplot.show()
    
def Make_X_Y(df):
    Num_of_line = 100
    Y = pd.DataFrame()
    Y['is_attributed'] = df['is_attributed']
    X = df.copy()
    X.drop(labels = ["is_attributed"], axis = 1, inplace = True)
    print(Num_of_line*'=')
    print('X=', X.head(5))
    print(Num_of_line*'=')
    print('Y=', Y.head(5))
    return X, Y

def Train_Test_training_valid(X, Y, ratio):
    Num_of_line = 100
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=ratio)
    print ('Train X shape = ', X_train.shape, '-----Train Y shape = ', y_train.shape)
    print(Num_of_line*'=')
    print ('Test X shape = ', X_test.shape, '-----Test Y shape = ',y_test.shape)
    print(Num_of_line*'=')
    X_training, X_valid, y_training, y_valid = \
    train_test_split(X_train, y_train, test_size=ratio, random_state=0)
    print ('Training X shape = ', X_training.shape, '-----Training Y shape = ', y_training.shape)
    print(Num_of_line*'=')
    print ('Valid X shape = ', X_valid.shape, '-----Valid Y shape = ',y_valid.shape)
    
    return X_training, y_training, X_valid, y_valid

def Drop_cols(df, x):
    Num_of_line = 100
    print(Num_of_line*'=')
    print('Before drop =\n', df.head(3))
    print(Num_of_line*'=')
    df.drop(labels = x, axis = 1, inplace = True)
    print('After drop =\n', df.head(3))
    return df

def Normalized(df):

    df_col_names = df.columns
    x = df.values 
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    df.columns = df_col_names
    
    
    return df


def Parse_time(df):
    df['day'] = df['click_time'].dt.day.astype('uint8')
    df['hour'] = df['click_time'].dt.hour.astype('uint8')
    df['minute'] = df['click_time'].dt.minute.astype('uint8')
    df['second'] = df['click_time'].dt.second.astype('uint8')
    
def Merge_data(df1, df2):
    frames = [df1, df2]
    df = pd.concat(frames)
    return df



def read_csv_random(address_train, address_test, p):
    
    
    #p = 0.01  # 1% of the lines
    # keep the header, then take only 1% of lines
    # if random from [0,1] interval is greater than 0.01 the row will be skipped
    df_train = pd.read_csv(
         address_train, parse_dates=['click_time'],
         header=0, 
         skiprows=lambda i: i>0 and random.random() > p)
    
    df_test = pd.read_csv(address_test, parse_dates=['click_time'])

    return df_train, df_test


def read_train_test_data(address_train, train_nrows, address_test, test_nrows, Skip_range_low, Skip_range_Up, nrows):
    
    df_train = pd.read_csv(address_train, parse_dates=['click_time'], skiprows=range(Skip_range_low,Skip_range_Up), nrows = nrows)
    df_test = pd.read_csv(address_test, parse_dates=['click_time'])#, nrows = 100)#, nrows = test_nrows)
    return df_train, df_test


def read_train_test_data_balanced(address_train, address_test):
    
    #Read Training data, all class 1 and add same amount 0
    iter_csv = pd.read_csv(address_train, iterator=True, chunksize=10000000, parse_dates=['click_time'])
    df_train_1 = pd.concat([chunk[chunk['is_attributed'] > 0] for chunk in iter_csv])
    iter_csv = pd.read_csv(address_train, iterator=True, chunksize=10000000, parse_dates=['click_time'], nrows=2000000)
    df_train_0 = pd.concat([chunk[chunk['is_attributed'] == 0] for chunk in iter_csv])
    #seperate same number values as train data with class 1
    df_train_0 = df_train_0.head(len(df_train_1))
    #Merge 0 and 1 data
    df_train = Merge_data(df_train_1, df_train_0)
    
    #Read Test data
    df_test = pd.read_csv(address_test, parse_dates=['click_time'])
    return df_train, df_test

def read_train_test_data_balanced_oversample1(address_train, address_test):
    
    #Read Training data, all class 1 and add same amount 0
    iter_csv = pd.read_csv(address_train, iterator=True, chunksize=10000000, parse_dates=['click_time'])
    df_train_1 = pd.concat([chunk[chunk['is_attributed'] > 0] for chunk in iter_csv])
    iter_csv = pd.read_csv(address_train, iterator=True, chunksize=10000000, parse_dates=['click_time'], nrows=5000000)
    df_train_0 = pd.concat([chunk[chunk['is_attributed'] == 0] for chunk in iter_csv])
    
    count_class_0 = len(df_train_0)
    
    df_train_1_over = df_train_1.sample(count_class_0, replace=True)
    df_train_over = pd.concat([df_train_1_over, df_train_0], axis=0)
    print('Random over-sampling:')
    print(df_train_over.is_attributed.value_counts())
    #Read Test data
    df_test = pd.read_csv(address_test, parse_dates=['click_time'])

    return df_train_over, df_test

def check_memory():
    
    mem=str(os.popen('free -t -m').readlines())
    T_ind=mem.index('T')
    mem_G=mem[T_ind+14:-4]
    S1_ind=mem_G.index(' ')
    mem_T=mem_G[0:S1_ind]
    mem_G1=mem_G[S1_ind+8:]
    S2_ind=mem_G1.index(' ')
    mem_U=mem_G1[0:S2_ind]
    mem_F=mem_G1[S2_ind+8:]
    print('Free Memory = ' + mem_F +' MB')


def Feature_engineering(df, ip_count):
    
    # Count the number of clicks by ip
    #ip_count = df.groupby(['ip'])['channel'].count().reset_index()
    #ip_count.columns = ['ip', 'clicks_by_ip']
    df = pd.merge(df, ip_count, on='ip', how='left', sort=False)
    df['clicks_by_ip'] = df['clicks_by_ip'].astype('uint16')
    #df.drop('ip', axis=1, inplace=True)
    return df


def predict_And_Submit_using_xgb(df, Trained_Model):
    

    Num_of_line = 100
    print(Num_of_line*'=')
    #sub = pd.DataFrame()
    #sub['click_id'] = df['click_id'].astype('int')
    #df['clicks_by_ip'] = df['clicks_by_ip'].astype('uint16')
    
    data_to_submit = pd.DataFrame()
    data_to_submit['click_id'] = range(0, len(df))
    dtest = xgb.DMatrix(df)
    del df
    predict = Trained_Model.predict(dtest, ntree_limit=Trained_Model.best_ntree_limit)
    data_to_submit['is_attributed'] = predict

    print(Num_of_line*'=')
    print('data_to_submit = \n', data_to_submit.head(5))
    pyplot.hist(data_to_submit['is_attributed'], log = True)
    #data_to_submit.to_csv('Amin_csv_to_submit.csv', index = False)
    return data_to_submit


def predict_And_Submit(df, Trained_Model):

    Num_of_line = 100
    print(Num_of_line*'=')
    pred = Trained_Model.predict(df)
    print('pred Done.')
    predict = pd.DataFrame(pred)
    data_to_submit = pd.DataFrame()
    data_to_submit['click_id'] = range(0, len(df))
    data_to_submit['is_attributed'] = predict
    print(Num_of_line*'=')
    print('data_to_submit = \n', data_to_submit.head(5))
    pyplot.hist(data_to_submit['is_attributed'], log = True)
    #data_to_submit.to_csv('Amin_csv_to_submit.csv', index = False)
    return data_to_submit
def Train_KNN(X_training, y_training, X_valid, y_valid):
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_training, y_training)
    pred_knn = knn_clf.predict(X_valid)
    KNN_accuracy = accuracy_score(y_valid, pred_knn)
    print('KNN_accuracy=\n', KNN_accuracy)
    Trained_KNN_Model = knn_clf
    return Trained_KNN_Model, KNN_accuracy
    
def Train_Decision_tree(X_training, y_training, X_valid, y_valid):
    dt_clf = DecisionTreeClassifier()
    dt_clf.fit(X_training, y_training)
    pred_dt = dt_clf.predict(X_valid)
    Decision_tree_accuracy = accuracy_score(y_valid, pred_dt)
    print('Decision_tree_accuracy=\n', Decision_tree_accuracy)
    Trained_Decision_tree_Model = dt_clf
    return Trained_Decision_tree_Model, Decision_tree_accuracy
    
def Train_Random_forest(X_training, y_training, X_valid, y_valid):
    rf_clf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
    rf_model = rf_clf.fit(X_training, y_training)
    pred_rf = rf_clf.predict(X_valid)
    Random_forest_accuracy = accuracy_score(y_valid, pred_rf)
    print('Random_forest_accuracy=\n', Random_forest_accuracy)
    Trained_Random_forest_Model = rf_clf
    return Trained_Random_forest_Model, Random_forest_accuracy
    
def Train_logistic_regression(X_training, y_training, X_valid, y_valid):
    logreg_clf = LogisticRegression()
    logreg_clf.fit(X_training, y_training)
    pred_logreg = logreg_clf.predict(X_valid)
    logistic_regression_accuracy = accuracy_score(y_valid, pred_logreg)
    print('logistic_regression_accuracy=\n', logistic_regression_accuracy)
    Trained_logistic_regression_Model = logreg_clf
    return Trained_logistic_regression_Model, logistic_regression_accuracy

def Train_Gaussian_Naive_Bayes(X_training, y_training, X_valid, y_valid):
    gnb_clf = GaussianNB()
    gnb_clf.fit(X_training, y_training)
    pred_gnb = gnb_clf.predict(X_valid)
    Gaussian_Naive_Bayes_accuracy = accuracy_score(y_valid, pred_gnb)
    print('Gaussian_Naive_Bayes_accuracy=\n', Gaussian_Naive_Bayes_accuracy)
    Trained_Gaussian_Naive_Bayes_Model = gnb_clf
    return Trained_Gaussian_Naive_Bayes_Model, Gaussian_Naive_Bayes_accuracy
    
def Train_support_vector_machine(X_training, y_training, X_valid, y_valid):
    linsvc_clf = LinearSVC()
    linsvc_clf.fit(X_training, y_training)
    pred_linsvc = linsvc_clf.predict(X_valid)
    support_vector_machine_accuracy = accuracy_score(y_valid, pred_linsvc)
    print('support_vector_machine_accuracy=\n', support_vector_machine_accuracy)
    Trained_support_vector_machine_Model = linsvc_clf
    return Trained_support_vector_machine_Model, support_vector_machine_accuracy
    
def Train_XGBoost(X_training, y_training, X_valid, y_valid):

    XGB_model = XGBClassifier(learning_rate =0.1,
                              n_estimators=1000,
                              max_depth=15,
                              min_child_weight=1,
                              gamma=0,
                              subsample=0.8,
                              colsample_bytree=0.8,
                              objective= 'binary:logistic',
                              nthread=4,
                              scale_pos_weight=1)
    
   
    
    
    XGB_model.fit(X_training, y_training)
    pred_XGB = XGB_model.predict(X_valid)
    pred_XGB = [round(value) for value in pred_XGB]
    XGBoost_accuracy = accuracy_score(y_valid, pred_XGB)
    print('XGBoost_accuracy=\n', XGBoost_accuracy)
    Trained_XGBoost_Model = XGB_model
    return Trained_XGBoost_Model, XGBoost_accuracy

def Train_BBC(X_training, y_training, X_valid, y_valid):
    #Create an object of the classifier.
    bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                    sampling_strategy='auto',
                                    replacement=False,
                                    random_state=0)

    BBC = bbc.fit(X_training, y_training)
    pred_BBC = bbc.predict(X_valid)
    BBC_accuracy = accuracy_score(y_valid, pred_BBC)
    print('BBC accuracy = ', BBC_accuracy)
    return BBC, BBC_accuracy


def xgb2(X_training, y_training, X_valid, y_valid):
    
    params = {'eta': 0.3,
          'tree_method': "hist",
          'grow_policy': "lossguide",
          'max_leaves': 1400,  
          'max_depth': 0, 
          'subsample': 0.9, 
          'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
          'min_child_weight':0,
          'alpha':4,
          'objective': 'binary:logistic', 
          'scale_pos_weight':9,
          'eval_metric': 'auc', 
          'nthread':8,
          'random_state': 99, 
          'silent': True}
    
    dtrain = xgb.DMatrix(X_training, y_training)
    dvalid = xgb.DMatrix(X_valid, y_valid)
    
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    
    model = xgb.train(params, dtrain, 200, watchlist, maximize=True, early_stopping_rounds = 25, verbose_eval=5)

    return model


def model_performance(Models, Accuracy):
    model_performance = pd.DataFrame({
    "Model": Models,
    "Accuracy": Accuracy
    })
    print(model_performance.sort_values(by="Accuracy", ascending=False))
    
def generate_ip_count(df_train, df_test):
    
    
    df_train2 = df_train.copy()
    df_test2 = df_test.copy()
    # Drop the IP and the columns from target
    y = df_train2['is_attributed']
    df_train2.drop(['is_attributed'], axis=1, inplace=True)
    # Drop IP and ID from test rows
    sub = pd.DataFrame()
    #sub['click_id'] = test['click_id'].astype('int')
    df_test2.drop(['click_id'], axis=1, inplace=True)
    gc.collect()
    nrow_df_train2 = df_train2.shape[0]
    merge = pd.concat([df_train2, df_test2])

    del df_train2, df_test2
    gc.collect()
    
    # Count the number of clicks by ip
    ip_count = merge.groupby(['ip'])['channel'].count().reset_index()
    ip_count.columns = ['ip', 'clicks_by_ip']
    merge = pd.merge(merge, ip_count, on='ip', how='left', sort=False)
    merge['clicks_by_ip'] = merge['clicks_by_ip'].astype('uint16')
    merge.drop('ip', axis=1, inplace=True)

    df_train2 = merge[:nrow_df_train2]
    df_test2 = merge[nrow_df_train2:]
    del df_test2, merge
    gc.collect()
    
    return ip_count
def Run_Kernel(Skip_range_low, Skip_range_Up, nrows):
    
    Start_time = time.time()
    #Address to data
    address_train = '../input/talkingdata-adtracking-fraud-detection/train.csv'
    address_test = '../input/talkingdata-adtracking-fraud-detection/test.csv'
    address_train_sample = '../input/talkingdata-adtracking-fraud-detection/train_sample.csv'
    address_test_supplement = '../input/talkingdata-adtracking-fraud-detection/test_supplement.csv'
    
    print('Reading data...!'); check_memory()
    nrows_read_train = 100; nrows_read_test = 100
    #df_train, df_test = read_train_test_data(address_train, nrows_read_train, address_test, nrows_read_test)
    #df_train, df_test = read_train_test_data(address_train, nrows_read_train, address_test, nrows_read_test, Skip_range_low, Skip_range_Up, nrows)
    df_train, df_test = read_train_test_data_balanced(address_train_sample, address_test)
    #df_train, df_test = read_train_test_data_balanced_oversample1(address_train, address_test)
    #df_train, df_test = read_csv_random(address_train, address_test, 0.05)
    print(len(df_train)); print('Reading Done!'); check_memory()
    
    some_useful_data_insight(df_train)
    some_useful_data_insight(df_test)
    Plot_Hist_columns(df_train, ['ip', 'app','device', 'os', 'channel', 'is_attributed'])
   

    #Parse time
    print('Parse, training data...'); check_memory(); Parse_time(df_train); print('Parse, training data, Done!'); 
    check_memory()
    
    #Feature_engineering data
    ip_count = generate_ip_count(df_train, df_test)
    df_train = Feature_engineering(df_train, ip_count); df_train.head(); null_table(df_train);  df_train.head() #df_train = df_train.dropna()
    
    #Drop and normalize 
    print('Drop colum and normalize, training data...!'); check_memory()
    colmn_names = ['attributed_time','click_time', 'ip']; df_train = Drop_cols(df_train, colmn_names)
    #df_train = Normalized(df_train)
    print('Drop colum and normalize, training data, Done!'); check_memory()
    
    
    #Devide training data, X-Y
    print('Begin devide training data, X_Y...'); check_memory()
    X, Y = Make_X_Y(df_train); X_training, y_training, X_valid, y_valid = Train_Test_training_valid(X, Y, 0.1)
    print('Begin devide training data, X_Y, Done!'); check_memory()
    print('Cleaning before training'); del df_train; gc.collect(); check_memory()
    print('Begin training...'); check_memory()
    
    #Trained_XGBoost_Model, XGBoost_accuracy = Train_XGBoost(X_training, y_training, X_valid, y_valid)
    Trained_Decision_tree_Model, Decision_tree_accuracy = Train_Decision_tree(X_training, y_training, X_valid, y_valid)
    #Trained_BBC_Model, BBC_accuracy = Train_BBC(X_training, y_training, X_valid, y_valid)
    #Trained_support_vector_machine_Model, support_vector_machine_accuracy = Train_support_vector_machine(X_training, y_training, X_valid, y_valid)    
    #Trained_KNN_Model, KNN_accuracy = Train_KNN(X_training, y_training, X_valid, y_valid)
    #Trained_Random_forest_Model, Random_forest_accuracy = Train_Random_forest(X_training, y_training, X_valid, y_valid)
    #Trained_logistic_regression_Model, logistic_regression_accuracy = Train_logistic_regression(X_training, y_training, X_valid, y_valid)
    #Trained_Gaussian_Naive_Bayes_Model, Gaussian_Naive_Bayes_accuracy = Train_Gaussian_Naive_Bayes(X_training, y_training, X_valid, y_valid)
    #Trained_xgb2_Model = xgb2(X_training, y_training, X_valid, y_valid)
    
    print('training Done!'); check_memory(); print('reading test data')
    
    
    
    
    #Parse time
    print('Parse, test data...'); check_memory(); Parse_time(df_test); df_test.head(); null_table(df_test); print('Parse, test data, Done!'); check_memory()
    #Feature_engineering data
    df_test = pd.merge(df_test, ip_count, on='ip', how='left', sort=False)
    #df_test, ip_count = Feature_engineering(df_test)
    #Drop and normalize
    print('Drop colum and normalize, test data...!'); check_memory()
    colmn_names = ["click_time", "click_id", "ip"]; df_test = Drop_cols(df_test, colmn_names); df_test.head(); null_table(df_test);
    #df_test = Normalized(df_test)
    print('Drop colum and normalize, test data, Done!'); check_memory()
    print('Cleaning before prediction'); del X, Y, X_training, y_training, X_valid, y_valid, ip_count; gc.collect(); check_memory()

    #Begin Prediction
    print('Begin Prediction...')
    check_memory()
    #data_to_submit = predict_And_Submit(df_test, Trained_XGBoost_Model)
    data_to_submit = predict_And_Submit(df_test, Trained_Decision_tree_Model)
    #data_to_submit = predict_And_Submit(df_test, Trained_BBC_Model)
    #data_to_submit = predict_And_Submit(df_test, Trained_support_vector_machine_Model)
    #data_to_submit = predict_And_Submit(df_test, Trained_KNN_Model)
    #data_to_submit = predict_And_Submit(df_test, Trained_Random_forest_Model)
    #data_to_submit = predict_And_Submit(df_test, Trained_logistic_regression_Model)
    #data_to_submit = predict_And_Submit(df_test, Trained_Gaussian_Naive_Bayes_Model)
    #data_to_submit = predict_And_Submit_using_xgb(df_test, Trained_xgb2_Model)    
    
    print('Prediction Done!'); check_memory(); print('Cleaning RAM'); del df_test; gc.collect(); check_memory()
    print('Program ran for {} seconds'.format(time.time()-Start_time)); print(50*'=','\n',50*'=')
    
    return data_to_submit
                #Run_Kernel(Skip_range_low, Skip_range_Up, nrows)
data_to_submit = Run_Kernel(1, 1200000, 3000000) # Note that in case of using balanced data reading the used values in the Run_Kernel function are ignored.
data_to_submit.to_csv('Amin_csv_to_submit.csv', index = False)
data_to_submit.head()