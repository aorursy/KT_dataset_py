from xgboost import XGBClassifier

import pandas as pd

import numpy as np

from sklearn.model_selection import RandomizedSearchCV

#from sklearn.tree import DecisionTreeClassifier

from sklearn.calibration import CalibratedClassifierCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import log_loss

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestRegressor

from tqdm import tqdm

from math import radians, sin, cos, acos

from collections import Counter

import warnings

warnings.filterwarnings("ignore")
zomato_data = pd.read_csv("../input/zomato_data_with_ratings_complete.csv")
zomato_data['New_Id'] = range(1, 1+len(zomato_data))
zomato_data.set_index('New_Id', inplace=True)
zomato_data
zomato_data['menu_item']
zomato_data.drop(columns=['dish_liked','reviews_list','menu_item','listed_in(type)'], inplace  =True)
zomato_data.shape
zomato_data.drop(zomato_data[zomato_data['rate'] == "NEW"].index, inplace = True) 

zomato_data.drop(zomato_data[zomato_data['rate'] == "-"].index, inplace = True)

zomato_data.dropna(how = 'any', inplace = True)
zomato_data['online_order']= pd.get_dummies(zomato_data.online_order, drop_first=True)

zomato_data['book_table']= pd.get_dummies(zomato_data.book_table, drop_first=True)
zomato_data['rest_type'] = zomato_data['rest_type'].str.replace(',' , '') 

zomato_data['rest_type'] = zomato_data['rest_type'].astype(str).apply(lambda x: ' '.join(sorted(x.split())))
zomato_data['cuisines'] = zomato_data['cuisines'].str.replace(',' , '') 

zomato_data['cuisines'] = zomato_data['cuisines'].astype(str).apply(lambda x: ' '.join(sorted(x.split())))
zomato_data.shape
from sklearn.preprocessing import LabelEncoder

T = LabelEncoder()                 

zomato_data['location_encoded'] = T.fit_transform(zomato_data['location'])

zomato_data['rest_type_encoded'] = T.fit_transform(zomato_data['rest_type'])

zomato_data['cuisines_encoded'] = T.fit_transform(zomato_data['cuisines'])

zomato_data['geo_loc_encoded'] = T.fit_transform(zomato_data['geo_loc'])
zomato_data["approx_cost(for two people)"] = zomato_data["approx_cost(for two people)"].str.replace(',' , '')

zomato_data["approx_cost(for two people)"] = zomato_data["approx_cost(for two people)"].astype('float')
zomato_data.shape
zomato_data.head()
x = zomato_data.drop(['rate','name', 'location', 'rest_type', 'cuisines', 'geo_loc'],axis = 1)
y = zomato_data['rate']
x.shape
y.shape
x
# def normalize(dataframe):

#     #print("Here")

#     test = dataframe.copy()

#     for col in test.columns:

#         if(col != "online_order" and col !="book_table"):

#             max_val = max(dataframe[col])

#             min_val = min(dataframe[col])

#             test[col] = (dataframe[col] - min_val) / (max_val-min_val)

#     return test
# x_final = normalize(x)

x_final = x
# split the data into test and train by maintaining same distribution of output varaible .[stratify=data_y]

x_train,x_test,y_train,y_test = train_test_split(x_final,y,stratify=y,test_size=0.20)

# split the data into train and cv by maintaining same distribution of output varaible. [stratify=y_train]

x_train,x_cv,y_train,y_cv = train_test_split(x_train,y_train,stratify=y_train,test_size=0.20)
def perform_hyperparam_tuning(list_of_hyperparam, model_name,  x_train, y_train, x_cv, y_cv):

    cv_log_error_array = []

    for i in tqdm(list_of_hyperparam):

        model = RandomForestRegressor(n_estimators = i,random_state = 42,n_jobs = -1)

        model.fit(x_train, y_train)

        cv_log_error_array.append(model.score(x_cv, y_cv))

    for i in range(len(cv_log_error_array)):

        print ('accuracy for hyper_parameter = ',list_of_hyperparam[i],'is',cv_log_error_array[i])

    return cv_log_error_array



def get_best_hyperparam(list_of_hyperparam, cv_log_error_array):

    index = np.argmin(cv_log_error_array)

    best_hyperparameter = list_of_hyperparam[index]

    return best_hyperparameter





def perform_on_best_hyperparam(model_name, best_hyperparameter, cv_log_error_array,x_train,y_train,x_cv,y_cv,x_test,y_test):

    

    model = RandomForestRegressor(n_estimators = best_hyperparameter,random_state = 42,n_jobs = -1)           

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print("The test accuracy for best hyperparameter is", model.score(x_test, y_test)*100)

    return (y_test, y_pred, model)
list_of_hyperparam = [10,50,100,500,1000,2000,3000]

model_name = "rf"

cv_log_error_array = perform_hyperparam_tuning(list_of_hyperparam, model_name,  x_train, y_train, x_cv, y_cv)
best_hyperparameter = get_best_hyperparam(list_of_hyperparam, cv_log_error_array)
(y_test, y_pred, model) = perform_on_best_hyperparam(model_name, best_hyperparameter, cv_log_error_array,x_train,y_train,x_cv,y_cv,x_test,y_test)
Randpred = pd.DataFrame({"actual": y_test, "pred": y_pred })

Randpred
print(model.predict(x_test[1001:1002]))
print(y_test[1001:1002])