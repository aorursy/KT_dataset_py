import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))

import warnings

warnings.filterwarnings('ignore')

import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

import seaborn as sns



#-----------------------------------------------------------



#Train data

app_train = pd.read_csv('../input/mydataaa/application_train.csv')

#Test data

app_test = pd.read_csv("../input/mydataaa/application_test.csv")



#-----------------------------------------------------------



# Khảo sát dữ liệu tróng (NULL) của mỗi cột

mis_val= app_train.isnull().sum() # Trả về tổng các dữ liệu trống (NULL)



# Khảo sát để lựa chọn phương pháp mã hóa dữ liệu phù hợp để xây dựng mô hình

app_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0) # Đếm các giá trị bị trùng của mỗi cột

# =>Đối với cá giá trị duy nhất của mỗi cột mà  <= 2 thì ta sẽ mã hóa nhãn (LabelEncoding) và > 2 thì ta sẽ mã hóa 1 nóng (One-Hot Encoding)

# Đối với LebelEncoding sẽ có tạo thêm cột mới tương ứng với cột được mã hóa



#-----------------------------------------------------------



# Label Encoding:Chuyển đổi những cột có dtype là chuỗi và <= 2

le = LabelEncoder()

le_count = 0;

for i in app_train:

    if app_train[i].dtype == 'object':

        if len(list(app_train[i].unique()))<=2 and i != "HOUSETYPE_MODE" :

            # Bắt đầu mã hóa

            le.fit(app_train[i])

            # Chuyển đổi dữ liệu các cột đã xét qua

            app_train[i] = le.fit_transform(app_train[i])

            app_test[i] = le.transform(app_test[i])

            # Đếm các cột đã mẫ hóa nhãn

            le_count+=1;





# One-hot Encoding: Chuyển đổi những cột có dtype là số

app_train = pd.get_dummies(app_train)

app_test = pd.get_dummies(app_test)



#-----------------------------------------------------------



## Căn chỉnh dữ liệu Training và Test



train_label = app_train['TARGET']

# Vì tệp training thì có TARGET còn test thì không, nên ta sẽ giữ lại TARGET của tệp training

# Ta sẽ loại bỏ các cột mà cả 2 tệp Trining và Test sao cho nó đồng nhất để tạo MODEL

app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)



#Gán lại TARGET

app_train['TARGET']= train_label



#-----------------------------------------------------------



# Xử lý các cột có giá trị dị thường tệp Training

# Ta thấy, 365243 là con số không hợp lý cho số ngày làm việc của 1 nhân viên, nên ta cần loại bỏ giá trị này để tranh việc xây dựng mô hình sai

app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] 

app_train['DAYS_EMPLOYED_ANOM'] == 365243

# Thay nó với nan

app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)



# Tương tự với tệp Tesing

app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"]

app_test['DAYS_EMPLOYED_ANOM']  == 365243

app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)



#-----------------------------------------------------------



# Ta thấy các cột EXT_SOURCE, DAYS_BIRTH có ảnh hưởng đến khả năng chi trả, ta xây dựng 1 Polynomial Features để xây dựng mô hình này



poly_features = app_train[["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3","DAYS_BIRTH","TARGET"]]

poly_features_test = app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]



from sklearn.impute import SimpleImputer 

imputer = SimpleImputer(strategy='median') # Sẽ tự điền trung bình của tưng cột tương ứng vào dữ liệu trống



poly_target = poly_features['TARGET']

poly_features = poly_features.drop(columns = ['TARGET']) # Tạo ra 1 bảng sao poly_features mà không có Nhãn

# Thêm giá trị vào chỗ thiếu dữ liệu

poly_features = imputer.fit_transform(poly_features)

poly_features_test = imputer.transform(poly_features_test)



#Tạo đối tượng đa thức với mức độ = 3

from sklearn.preprocessing import PolynomialFeatures

poly_transformer = PolynomialFeatures(degree=3)



# Huấn luyện các tính năng đa thức



poly_transformer.fit(poly_features)



#Chuyển đổi các Polynomial Features đã được mã hóa

poly_features = poly_transformer.transform(poly_features)

poly_features_test = poly_transformer.transform(poly_features_test)

# Đặt tên các cột chứa các 

poly_transformer.get_feature_names(input_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'])[:15]



#-----------------------------------------------------------

# ÁP DỤNG CÁC Polynomial Features ĐÃ TẠO VÀO DATA DÙNG ĐỂ TRAINING



# Tạo khung dữ liệu cho các tính năng được tạo

poly_features = pd.DataFrame(poly_features, columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']))



# Thêm TARGET vào

poly_features["TARGET"] = poly_target 



#-----------------------------------------------------------



# Tương tự tạo với khung dữ liệu thử

poly_features_test = pd.DataFrame(poly_features_test,columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2','EXT_SOURCE_3', 'DAYS_BIRTH']))



# Hợp nhất các tính năng vào bộ Training

# Trainig

poly_features['SK_ID_CURR'] = app_train["SK_ID_CURR"]

app_train_poly = app_train.merge(poly_features, on ="SK_ID_CURR", how = "left")

#Test

poly_features_test['SK_ID_CURR'] = app_test['SK_ID_CURR']

app_test_poly = app_test.merge(poly_features_test, on = 'SK_ID_CURR', how = 'left')

# Căn chỉnh

app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join = 'inner', axis = 1)



app_train_domain = app_train.copy()

app_test_domain = app_test.copy()



app_train_domain['CREDIT_INCOME_PERCENT'] = app_train_domain['AMT_CREDIT'] / app_train_domain['AMT_INCOME_TOTAL']

app_train_domain['ANNUITY_INCOME_PERCENT'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_INCOME_TOTAL']

app_train_domain['CREDIT_TERM'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_CREDIT']

app_train_domain['DAYS_EMPLOYED_PERCENT'] = app_train_domain['DAYS_EMPLOYED'] / app_train_domain['DAYS_BIRTH']



app_test_domain['CREDIT_INCOME_PERCENT'] = app_test_domain['AMT_CREDIT'] / app_test_domain['AMT_INCOME_TOTAL']

app_test_domain['ANNUITY_INCOME_PERCENT'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_INCOME_TOTAL']

app_test_domain['CREDIT_TERM'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_CREDIT']

app_test_domain['DAYS_EMPLOYED_PERCENT'] = app_test_domain['DAYS_EMPLOYED'] / app_test_domain['DAYS_BIRTH']



#-----------------------------------------------------------



# Tiền xử lý dữ liệu

from sklearn.preprocessing import MinMaxScaler

# Loại bỏ cột có TARGET trong tập Training

if 'TARGET' in app_train:

    train = app_train.drop(columns = ['TARGET'])

else:

    train = app_train.copy()

    

features = list(train.columns) # Tạo danh sách các tên cột

test = app_test.copy() 

imputer = SimpleImputer(strategy = "median")



# Chia tỷ lệ các tính năng từ 0 đến 1

scaler = MinMaxScaler(feature_range= (0,1))



imputer.fit(train)

# chuyển cả training và test

train = imputer.transform(train)

test = imputer.transform(app_test)



# Làm lại lại với Scaler MỤC ĐÍCH để giới hạn lại các giá trị trong khoảng (0 ,1)

scaler.fit(train)

train = scaler.transform(train)

test = scaler.transform(test)



# MÔ HÌNH ĐẠT 70%



# DÙNG RANDOM FOREST ĐỂ HUẤN LUYỆN MÔ HÌNH

from sklearn.ensemble import RandomForestClassifier

# Tạo bộ phân loại RF

random_forest = RandomForestClassifier()



random_forest.fit(train,train_label)

feature_importance_values = random_forest.feature_importances_



feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})

predictions = random_forest.predict_proba(test)[:, 1]



submit = app_test[['SK_ID_CURR']]

submit['TARGET'] = predictions

submit.to_csv('Random_Forest.csv', index = False)





#-----------------------------------------------------------



'''



import random

import csv

N_FOLDS = 5

MAX_EVALS = 5

from sklearn.model_selection import train_test_split



train_features, test_features, train_labels, test_labels = train_test_split(train, train_label, test_size = 6000, random_state = 50)



train_set = lgb.Dataset(data = train_features, label = train_labels)

test_set = lgb.Dataset(data = test_features, label = test_labels)









param_grid = {

    'boosting_type': ['gbdt', 'goss', 'dart'],

    'num_leaves': list(range(20, 50)),

    'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base = 1, num = 100)),

    'subsample_for_bin': list(range(500, 500, 500)),

    'min_child_samples': list(range(20, 500, 5)),

    'reg_alpha': list(np.linspace(0, 1)),

    'reg_lambda': list(np.linspace(0, 1)),

    'colsample_bytree': list(np.linspace(0.6, 1, 10)),

    'subsample': list(np.linspace(0.5, 1, 50)),

    'is_unbalance': [True, False]

}



out_file = 'random_search_trials.csv'





def objective(hyperparameters, iteration):

    """Objective function for grid and random search. Returns

       the cross validation score from a set of hyperparameters."""

    

    # Number of estimators will be found using early stopping

    if 'n_estimators' in hyperparameters.keys():

        del hyperparameters['n_estimators']

    

     # Perform n_folds cross validation

    cv_results = lgb.cv(hyperparameters, train_set, num_boost_round = 200, nfold = N_FOLDS, 

                        early_stopping_rounds = 2, metrics = 'auc', seed = 4)

    

    # results to retun

    score = cv_results['auc-mean'][-1]

    estimators = len(cv_results['auc-mean'])

    hyperparameters['n_estimators'] = estimators 

    

    return [score, hyperparameters, iteration]





def random_search(param_grid, out_file, max_evals = MAX_EVALS):

    """Random search for hyperparameter optimization. 

       Writes result of search to csv file every search iteration."""

    

    

    # Dataframe for results

    results = pd.DataFrame(columns = ['score', 'params', 'iteration'],

                                  index = list(range(MAX_EVALS)))

    for i in range(MAX_EVALS):

        

        # Choose random hyperparameters

        random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}

        random_params['subsample'] = 1.0 if random_params['boosting_type'] == 'goss' else random_params['subsample']



        # Evaluate randomly selected hyperparameters

        eval_results = objective(random_params, i)

        results.loc[i, :] = eval_results



        # open connection (append option) and write results

        of_connection = open(out_file, 'a')

        writer = csv.writer(of_connection)

        writer.writerow(eval_results)

        

        # make sure to close connection

        of_connection.close()

        

        

    # Sort with best score on top

    results.sort_values('score', ascending = False, inplace = True)

    results.reset_index(inplace = True)



    return results 





random_results = random_search(param_grid,out_file)







train_set = lgb.Dataset(train, label = train_labels)



hyperparameters = dict(**random_results.loc[0, 'hyperparameters'])

del hyperparameters['n_estimators']



# Xác thực chéo

cv_results = lgb.cv(hyperparameters, train_set,num_boost_round = 200, early_stopping_rounds = 2, metrics = 'auc', nfold = N_FOLDS)



# Traing

model = lgb.LGBMClassifier(n_estimators = len(cv_results['auc-mean']), **hyperparameters)

model.fit(train, train_labels)

                        

# Dự đoán

preds = model.predict_proba(test)[:, 1]



submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': preds})

submission.to_csv('RF_SUBMIT.csv', index = False)



'''