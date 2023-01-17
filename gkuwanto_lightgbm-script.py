# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/train.csv")
IP = pd.read_csv("/kaggle/input/IP_country.csv")
df_test = pd.read_csv("/kaggle/input/test.csv")
import datetime
from bisect import bisect_left
def feature_engineering_simple(df):
    # Country
    dic = {}
    countries = []
    for lower_bound, country in zip(IP["lower_bound_ip_address"], IP["country"]):
        dic[lower_bound] = country
        countries.append(country)
    keys = list(dic.keys())
    country_list = []
    for ip in df["ip_address"]:
        country_list.append(countries[bisect_left(keys, ip)-1])
    df["country"] = country_list
    
    # purchase - signup
    selisih = []
    for signup, purchase in zip(df["signup_time"], df["purchase_time"]):
        delta = (datetime.datetime.strptime(purchase, "%Y-%m-%d %H:%M:%S") - datetime.datetime.strptime(signup, "%Y-%m-%d %H:%M:%S"))
        selisih.append(delta.total_seconds())
    df["selisih"] = selisih
    
    # get week of year from purchase
    df['purchase_time'] = pd.to_datetime(df['purchase_time'], format='%Y-%m-%d %H:%M:%S')
    df['week'] = [df['purchase_time'][i].week for i in range(df.shape[0])]

def feature_engineering_with_total(df, df_test):
    # Device Count
    total = pd.concat([df_test, df.drop("class", axis = 1)])
    user_per_device = []
    device_dict = total.groupby("device_id").count()["user_id"]
    for dev in df["device_id"]:
        user_per_device.append(device_dict[dev])
    df["user_per_device"] = user_per_device
    user_per_device_test = []
    for dev in df_test["device_id"]:
        user_per_device_test.append(device_dict[dev])
    df_test["user_per_device"] = user_per_device_test
    
    # Get different combination of three columns
    df["sourcebrowsersex"] = df["source"] + df["browser"] + df["sex"]
    df_test["sourcebrowsersex"] = df_test["source"] + df_test["browser"] + df_test["sex"]
    
    #Mean Encoding all of it
    mean_encode(df,df_test, "country")
    mean_encode(df,df_test, "device_id")
    mean_encode(df,df_test, "source")
    mean_encode(df,df_test, "browser")
    mean_encode(df,df_test, "sex")
    mean_encode(df,df_test, "sourcebrowsersex")
    mean_encode(df,df_test, "ip_address")
def mean_encode(df,df_test, column):
    helper_dict = df.groupby(column).mean()["class"].sort_values()
    mean_encoded = []
    mean_encoded_test = []
    for i in df[column]:
        mean_encoded.append(helper_dict[i])
    for i in df_test[column]:
        if i in helper_dict:
            mean_encoded_test.append(helper_dict[i])
        else:
            mean_encoded_test.append(0)
    df[column] = mean_encoded
    df_test[column] = mean_encoded_test
def all_features(df,df_test):
    feature_engineering_simple(df)
    feature_engineering_simple(df_test)
    feature_engineering_with_total(df,df_test)
all_features(df,df_test)
df_test
df

df.to_csv("train.csv")
df_test.to_csv("test.csv")
helper_dict = df.groupby("country").mean()["class"].sort_values()
mean_encoded = []
mean_encoded_test = []
for i in df["country"]:
    mean_encoded.append(helper_dict[i])
for i in df_test["country"]:
    if i in helper_dict:
        mean_encoded_test.append(helper_dict[i])
    else:
        mean_encoded_test.append(0)
df["country"] = mean_encoded
df_test["country"] = mean_encoded_test
target = "class"
dropped_columns = ['user_id', 'signup_time', 'purchase_time', 'ip_address']
X_sub = df_test.drop(dropped_columns, axis = 1).copy()
X = df.drop(dropped_columns+[target],axis = 1).copy()
y = df[target].copy()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=314)
def learning_rate_010_decay_power_099(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3

def learning_rate_010_decay_power_0995(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.995, current_iter)
    return lr if lr > 1e-3 else 1e-3

def learning_rate_005_decay_power_099(current_iter):
    base_learning_rate = 0.05
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3
import lightgbm as lgb
fit_params={"early_stopping_rounds":30, 
            "eval_metric" : 'auc', 
            "eval_set" : [(X_test,y_test)],
            'eval_names': ['valid'],
#             'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],
            'verbose': 100,
            'categorical_feature': 'auto'}
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
param_test ={'num_leaves': sp_randint(6, 50), 
             'min_child_samples': sp_randint(100, 500), 
             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': sp_uniform(loc=0.2, scale=0.8), 
             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}
#This parameter defines the number of HP points to be tested
n_HP_points_to_test = 100

import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

#n_estimators is set to a "large value". The actual number of trees build will depend on early stopping and 5000 define only the absolute maximum
clf = lgb.LGBMClassifier(max_depth=-1, random_state=314, silent=True, metric='f1', n_jobs=4, n_estimators=5000)
gs = RandomizedSearchCV(
    estimator=clf, param_distributions=param_test, 
    n_iter=n_HP_points_to_test,
    scoring='f1',
    cv=3,
    refit=True,
    random_state=314,
    verbose=True)
gs.fit(X_train, y_train, **fit_params)
print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))
opt_parameters = gs.best_params_
gs.best_score_
clf_sw = lgb.LGBMClassifier(**clf.get_params())
#set optimal parameters
clf_sw.set_params(**opt_parameters)
gs_sample_weight = GridSearchCV(estimator=clf_sw, 
                                param_grid={'scale_pos_weight':[1,2,6,12]},
                                scoring='f1',
                                cv=5,
                                refit=True,
                                verbose=True)
gs_sample_weight.fit(X_train, y_train, **fit_params)
print('Best score reached: {} with params: {} '.format(gs_sample_weight.best_score_, gs_sample_weight.best_params_))
gs.cv_results_['mean_test_score']
print("Valid+-Std      :   Parameters")
for i in np.argsort(gs.cv_results_['mean_test_score'])[-5:]:
    print('{1:.3f}+-{2:.3f}    :  {0}'.format(gs.cv_results_['params'][i], 
                                   gs.cv_results_['mean_test_score'][i], 
                                   gs.cv_results_['std_test_score'][i]))
    print()
clf_final = lgb.LGBMClassifier(**clf.get_params())
#set optimal parameters
clf_final.set_params(**opt_parameters)

#Train the final model with learning rate decay
clf_final.fit(X_train, y_train, **fit_params, 
              callbacks=[lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_0995)]
             )
clf_final.feature_importances_
feat_imp = pd.Series(clf_final.feature_importances_, index=X.columns)
feat_imp.nlargest(20).plot(kind='barh', figsize=(8,10))
clf_final.feature_importances_
probabilities = clf_final.predict_proba(X)
submission = pd.DataFrame({
    target:     [ row[1] for row in probabilities]
})
sum(clf_final.predict(X_test))
from sklearn.metrics import f1_score
f1_score(clf_final.predict(X_test), y_test)
sum(clf_final.predict(X_sub))

