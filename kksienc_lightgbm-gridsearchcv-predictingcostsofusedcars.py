# import libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import lightgbm as lgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)



# Any results you write to the current directory are saved as output.
train  = pd.read_excel("../input/Data_Train.xlsx")

test = pd.read_excel("../input/Data_Test.xlsx")
train.isnull().sum()
test.isnull().sum()
# copy train & test data 

traincopy = train.copy() 

testcopy = test.copy()
# droping New_Price as majority of values are missing

traincopy.drop(('New_Price'), axis =1 , inplace = True)

testcopy.drop(('New_Price'), axis = 1 , inplace = True)
# Fetch car name from Name

traincopy['Car'] = traincopy.Name.str.split().str.get(0)+" " +traincopy.Name.str.split().str.get(1)

testcopy['Car'] = testcopy.Name.str.split().str.get(0)+" " +testcopy.Name.str.split().str.get(1)
# replace missing value with mode for dataframe extract having identical Car value 

# Maruti Estilo car has no entry  for all its records so we feed it manually 

# also correct the mileage  in case it is 0



traincopy.loc[traincopy['Car'] == 'Maruti Estilo', 'Seats'] = 5.0

traincopy.loc[traincopy['Car'] == 'Maruti Estilo', 'Power'] = '64 bhp'



dataset = [traincopy,testcopy]



for data in dataset:

    car_list = data['Car'].unique() 

    for carname in car_list:

        subdef = data[data['Car'] ==  carname ]

        subdef.fillna(subdef.mode().iloc[0], inplace=True)

        if '0.0 kmpl' in subdef['Mileage']:

            subdef.replace('0.0 kmpl',subdef.Mileage.mode().iloc[0] ,inplace=True, )

        data.update(subdef)



# We corrected the mileage  in case its 0

# Milage is still missing in train (not in test ), we will delete these records 

traincopy.dropna(axis=0,inplace = True)
# features type conversion



# for Owner_Type mapping to int

convrt_to_num =  { 'First' :1 , 'Second' :2 ,'Third' :3, 'Fourth & Above' :4   }



#for Transmission mapping to int

convrt_to_binary =  { 'Manual' : 0 , 'Automatic' : 1 }



dataset = [traincopy,testcopy]



for data in dataset:

    

    data['Owner_Type'] = data['Owner_Type'].map(convrt_to_num)

    data['Transmission'] = data['Transmission'].map(convrt_to_binary)

    

    # Milegae to float

    data['Mileage'] = data.Mileage.str.split().str.get(0).astype(float)

    

    #Engine to int 

    data['Engine'] = data.Engine.str.split().str.get(0).astype(int)

    

    #power to float 

    data['Power'][data.Power == 'null bhp'] = '0'

    data['Power'] = data.Power.str.split().str.get(0).astype(float)

    

    # Seats as int 

    data.Seats = data.Seats.astype(int)

    

    # Year, Kilometers_Driven from float to int 

    data['Year']=data['Year'].astype(int)

    data['Kilometers_Driven']=data['Kilometers_Driven'].astype(int)

    

    # from Year  get the age of car 

    data['Age'] =  (2019 - data['Year']).astype(int)

    

    #from Car fetch the brand of car

    data['Car_Brand'] = data.Car.str.split().str.get(0)

    

    #lets convert objects to categorical type 

    data['Location'] = data.Location.astype('category')

    data['Fuel_Type'] = data.Fuel_Type.astype('category')

    data['Car_Brand'] = data.Car_Brand.astype('category')

    data['Seats'] = data.Seats.astype('category')

    data['Owner_Type'] = data.Owner_Type.astype('category')

    data['Transmission'] = data.Transmission.astype('category')

    data['Age'] = data.Age.astype('category')  

    

    # droping not so useful features 

    data.drop( (['Year','Car', 'Name']), axis = 1 , inplace =True )

    
# Segregating target & Features 

X= traincopy.drop('Price',axis=1)

y= traincopy['Price']



# test train split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1234)
lgb_model = lgb.LGBMRegressor(

    categorical_feature= [0,2,3,4,8,9,10],

    task = 'predict',

    application = 'regression',

    objective = 'root_mean_squared_error',

    boosting_type="gbdt",

    num_iterations = 2500,

    learning_rate = 0.05,

    num_leaves=15,

    tree_learner='feature',

    max_depth =10,

    min_data_in_leaf=7,

    bagging_fraction = 1,

    bagging_freq = 100,

    reg_sqrt='True',

    metric ='rmse',

    feature_fraction = 0.6,

    random_state=42)



lgb_model.fit(X_train,y_train) 





preds_lgb_model = lgb_model.predict(X_test)

rmse_lgb = np.sqrt(mean_squared_error(y_test, preds_lgb_model))

print(" RMSE: %f" % (rmse_lgb ))
#Cross Validation

with warnings.catch_warnings():

    #just to supress warning

    warnings.filterwarnings("ignore")

    # Cross Validation score

    scores = cross_val_score(lgb_model,X,y,scoring = "neg_mean_squared_error",cv =10,verbose=1)

    rmse_scores = np.sqrt(-scores)

    print(rmse_scores.mean())
# grid search  hyperparameter tuning



# parameters = {

#     'task' : ['predict'],

#     'boosting': ['gbdt' ],

#     'objective': ['root_mean_squared_error'],

#     'num_iterations': [  1500, 2000,5000  ],

#     'learning_rate':[  0.05, 0.005 ],

#    'num_leaves':[ 7, 15, 31  ],

#    'max_depth' :[ 10,15,25],

#    'min_data_in_leaf':[15,25 ],

#   'feature_fraction': [ 0.6, 0.8,  0.9],

#     'bagging_fraction': [  0.6, 0.8 ],

#     'bagging_freq': [   100, 200, 400  ],

     

# }



# gsearch_lgb = GridSearchCV(lgb_model, param_grid = parameters, n_jobs=6,iid=False, verbose=10)

# gsearch_lgb.fit(X_train,y_train)

 



# print('best params')

# print (gsearch_lgb.best_params_)

# preds_lgb_model = gsearch_lgb.predict(X_test)

# rmse_lgb = np.sqrt(mean_squared_error(y_test, preds_lgb_model))

# print(" RMSE: %f" % (rmse_lgb ))
predictions = lgb_model.predict(testcopy) 

dataset = pd.DataFrame({'Price': predictions[:] })

dataset = round(dataset['Price'],2)

dataset.to_excel('out.xlsx', index = False, float_format ='%.2f' )