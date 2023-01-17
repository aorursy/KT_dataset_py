import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')

test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')

submission = pd.read_csv('/kaggle/input/bike-sharing-demand/sampleSubmission.csv')



def transform(dataset):

    dataset.loc[:,'hour'] = pd.to_datetime(dataset['datetime']).dt.hour    #  ->0.72

    dataset.loc[:,'day'] = pd.to_datetime(dataset['datetime']).dt.day    #  ->0.72

    dataset.loc[:,'month'] = pd.to_datetime(dataset['datetime']).dt.month    #  ->0.72

    dataset.loc[:,'year'] = pd.to_datetime(dataset['datetime']).dt.year    #  ->0.72

    dataset.loc[:,'weekday'] = pd.to_datetime(dataset['datetime']).dt.dayofweek # -0.02p

    dataset.loc[:,'hour_sin'] = np.sin(2 * np.pi * dataset['hour']/24.0)  # -0.02

    dataset.loc[:,'hour_cos'] = np.cos(2 * np.pi * dataset['hour']/24.0)





    numeric_cols = [cname for cname in dataset.columns if dataset[cname].dtype in ['int64', 'float64', 'bool']]

    return dataset[numeric_cols]



from sklearn.model_selection import train_test_split

import xgboost as xgb

from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error

import statistics

from sklearn.linear_model import LinearRegression, LogisticRegression 

from sklearn.svm import LinearSVR, SVR

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor 

from sklearn.model_selection import GridSearchCV

 



X, y = transform(df[df.columns[:-3]]).copy(), df['count'].apply(np.log1p)

X = X.fillna(0)

N, results, predictions = 30, [], []



for i in range(N):

    print(i,end='') 

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)



#     model = xgb.XGBRegressor(n_estimators=150,max_depth=5, learning_fate=0.1,

#                              reg_lambda=4.0,reg_alpha=4.0, min_child_weight=10, 

#                              min_split_loss=20)

#     model.fit(X_train, y_train, eval_metric='rmse', early_stopping_rounds=10, 

#               eval_set=[(X_val, y_val)], verbose=False)



#     model = LinearRegression() # rmsle =1.1

#     model = SVR() 

#     model = LinearSVR()          

#     model = GradientBoostingRegressor() # rmsle =0.75

#     model = DecisionTreeRegressor() #0.433!!!

#     model = DecisionTreeRegressor(max_depth=13, min_samples_split=10,

#         min_samples_leaf=10) #0.39!!!

    model = RandomForestRegressor(max_depth=13, n_estimators=35, #0.34

                                  min_samples_split=10, min_samples_leaf=10)

   

    model.fit(X_train, y_train) 

    y_prediction = np.maximum(model.predict(X_val), 0)

    

    results.append(np.sqrt(mean_squared_error(y_val,y_prediction)))

    predictions.append(model.predict(transform(test)))

    

print(' ',sum(results)/N, mean_absolute_error(results,[sum(results)/N]*len(results)))



rating = []

for i in range(len(test)):

     rating.append(sum([pred[i] for pred in predictions])/N) 





y_prediction = np.array([np.max(np.exp(r)-1.0,0) for r in rating])         

submission.loc[:,'count'] = y_prediction  

submission.to_csv('my_results.csv',index=False)
# 1.323130701880204 0.016642945847354915

# date 0.7216162589226854 0.020419579868874948

# cos  0.69               0.01 



#     model = RandomForestRegressor(max_depth=10, n_estimators=200, 0.358

#                                   min_samples_split=13, min_samples_leaf=13)



# линейная регрессия

# 1.0846787414019234 0.014829748101813759

# линейная регрессия, после шкалирования

# 0.8587908097119896 0.01109453999189203

# линейная регрессия, после сигмоиды

# 1.07052150656394 0.014058840347841736



# 0.16700500975878763 0.0025353183758171955



# svm

# 291.0228793109133916 0.027358918592929153

X.columns
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')

test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')

submission = pd.read_csv('/kaggle/input/bike-sharing-demand/sampleSubmission.csv')



def transform(dataset):

    dataset.loc[:,'hour'] = pd.to_datetime(dataset['datetime']).dt.hour    #  ->0.72

    dataset.loc[:,'day'] = pd.to_datetime(dataset['datetime']).dt.day    #  ->0.72

    dataset.loc[:,'month'] = pd.to_datetime(dataset['datetime']).dt.month    #  ->0.72

    dataset.loc[:,'year'] = pd.to_datetime(dataset['datetime']).dt.year    #  ->0.72

    dataset.loc[:,'weekday'] = pd.to_datetime(dataset['datetime']).dt.dayofweek # -0.02p

    dataset.loc[:,'hour_sin'] = np.sin(2 * np.pi * dataset['hour']/24.0)  # -0.02

    dataset.loc[:,'hour_cos'] = np.cos(2 * np.pi * dataset['hour']/24.0)





    numeric_cols = [cname for cname in dataset.columns if dataset[cname].dtype in ['int64', 'float64', 'bool']]

    return dataset[numeric_cols]



import xgboost as xgb

from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error

import statistics

from sklearn.tree import DecisionTreeRegressor 

from sklearn.model_selection import GridSearchCV

 



X, y = transform(df[df.columns[:-3]]).copy(), df['count'].apply(np.log1p)

X = X.fillna(0)



model = DecisionTreeRegressor()

parameters = {'max_depth':[10, 13, 15, 18, 20],

              'min_samples_split':[10],

             'min_samples_leaf':[10], 'max_leaf_nodes':[15,100,1000]}



find_it = GridSearchCV(model, parameters)

find_it.fit(X, y) 



print(find_it.best_params_)

 


