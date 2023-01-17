#Importing important modules

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import random

from sklearn.preprocessing import LabelEncoder

%matplotlib inline
#Loading data

stores = pd.read_csv("../input/lojas.csv")

stores_train_data = pd.read_csv("../input/dataset_treino.csv")

stores_test_data = pd.read_csv("../input/dataset_teste.csv")

stores_y_train_data = stores_train_data['Sales']
#Stores data

stores[:10]
#Data from Stores Train

stores_train_data[:10]
#Data from Target

stores_y_train_data[:10]
#Quantity of rows

qtd_rows = len(stores_train_data)

print(qtd_rows)
#Quantidade de rows

len(stores_test_data)
#In the description of the problem it is described that sales are influenced by promotions

#competitors, school and state holidays, seasonality and locality, this way

#we can then assign new attributes of the store data set

#these new attributes have been added through the ID parameter, each store has a unique ID.

for x in range(0,len(stores)):

    stores_train_data.loc[stores_train_data['Store'] == x+1, 'StoreType'] = stores.iloc[x,1]   

    stores_train_data.loc[stores_train_data['Store'] == x+1, 'Assortment'] = stores.iloc[x,2]   

    stores_train_data.loc[stores_train_data['Store'] == x+1, 'CompetitionDistance'] = stores.iloc[x,3]   

    stores_train_data.loc[stores_train_data['Store'] == x+1, 'CompetitionOpenSinceMonth'] = stores.iloc[x,4]   

    stores_train_data.loc[stores_train_data['Store'] == x+1, 'CompetitionOpenSinceYear'] = stores.iloc[x,5]   

    stores_train_data.loc[stores_train_data['Store'] == x+1, 'Promo2'] = stores.iloc[x,6]

    if (x % 50 == 0):

        print(x)

print("Complete!")
#The same process done previously is done for the Test Store data set, since they must have the

#same number of attributes

for x in range(0,len(stores)):

    stores_test_data.loc[stores_test_data['Store'] == x+1, 'StoreType'] = stores.iloc[x,1]   

    stores_test_data.loc[stores_test_data['Store'] == x+1, 'Assortment'] = stores.iloc[x,2]   

    stores_test_data.loc[stores_test_data['Store'] == x+1, 'CompetitionDistance'] = stores.iloc[x,3]   

    stores_test_data.loc[stores_test_data['Store'] == x+1, 'CompetitionOpenSinceMonth'] = stores.iloc[x,4]   

    stores_test_data.loc[stores_test_data['Store'] == x+1, 'CompetitionOpenSinceYear'] = stores.iloc[x,5]   

    stores_test_data.loc[stores_test_data['Store'] == x+1, 'Promo2'] = stores.iloc[x,6]

    if (x % 50 == 0):

        print(x)

print("Complete!")
#Checking out the new data set for Train Data

stores_train_data[:3]
#Checking out the new data set for Test Data

stores_test_data[:3]
#Columns from Train Data

stores_train_data.columns
#Columns from Test Data

stores_test_data.columns
#Data description - Train Data

stores_train_data.describe()
#Number of rows from Train Data

len(stores_train_data)
#Checking the amount of missing values for each attribute (column) in the Train Data

stores_train_data.isnull().sum()
#Replacing null values by mode

stores_train_data['CompetitionDistance'].fillna(stores_train_data.CompetitionDistance.mode()[0], inplace=True)

stores_train_data['CompetitionOpenSinceMonth'].fillna(stores_train_data.CompetitionOpenSinceMonth.mode()[0], inplace=True)

stores_train_data['CompetitionOpenSinceYear'].fillna(stores_train_data.CompetitionOpenSinceYear.mode()[0], inplace=True)
#Checking the amount of missing values for each attribute (column) in the Train Data - Again

stores_train_data.isnull().sum()
#Checking out the Test Data

stores_test_data.isnull().sum()
#Replacing null values by mode in Test Data

stores_test_data['Open'].fillna(stores_test_data.Open.mode()[0], inplace=True)

stores_test_data['CompetitionDistance'].fillna(stores_test_data.CompetitionDistance.mode()[0], inplace=True)

stores_test_data['CompetitionOpenSinceMonth'].fillna(stores_test_data.CompetitionOpenSinceMonth.mode()[0], inplace=True)

stores_test_data['CompetitionOpenSinceYear'].fillna(stores_test_data.CompetitionOpenSinceYear.mode()[0], inplace=True)
#Checking the amount of missing values for each attribute (column) in the Test Data

stores_test_data.isnull().sum()
#Checking out the types - Train Data

stores_train_data.dtypes
#Checking out the types - Train Data

stores_test_data.dtypes
#Separating the Date column into three new columns (Year, Month, and Day)

#Separating for Train Data

stores_train_data["Year"] = [x.split('-')[0] for x in stores_train_data.Date]

stores_train_data["Year"] = stores_train_data["Year"].astype(float)



stores_train_data["Month"] = [x.split('-')[1] for x in stores_train_data.Date]

stores_train_data["Month"] = stores_train_data["Month"].astype(float)



stores_train_data["Day"] = [x.split('-')[2] for x in stores_train_data.Date]

stores_train_data["Day"] = stores_train_data["Day"].astype(float)





#Separating for Test Data

stores_test_data["Year"] = [x.split('-')[0] for x in stores_test_data.Date]

stores_test_data["Year"] = stores_test_data["Year"].astype(float)



stores_test_data["Month"] = [x.split('-')[1] for x in stores_test_data.Date]

stores_test_data["Month"] = stores_test_data["Month"].astype(float)



stores_test_data["Day"] = [x.split('-')[2] for x in stores_test_data.Date]

stores_test_data["Day"] = stores_test_data["Day"].astype(float)
#Checking out the first rows for Train Data

stores_train_data[:3]
#Checking out the first rows for Test Data

stores_test_data[:3]
#Let's check the relationship between DayOfWeek and Sales

#Before, I must separate the days of the week into unique values

days_of_week = stores_train_data['DayOfWeek'].unique()

days_of_week
#For each day of the week I make the sum of the total sales volume

sales_by_week_day = {}

for x in days_of_week:

    selected_row = stores_train_data.loc[stores_train_data["DayOfWeek"] == x, "Sales"].sum().astype(float)    

    sales_by_week_day[x] = selected_row    
#Values (should be organized by day of the week)

sales_by_week_day
#Values are organized by the days of the week, which are the dictionary keys

tuple_days_of_week_sales = sorted(sales_by_week_day.items())
#Much better now

tuple_days_of_week_sales
#The data are separated for plotting

axis_days_of_week, axis_sales = zip(*tuple_days_of_week_sales) 
#Simple plot

plt.figure(figsize=(10,7))



#Color for colorblind people

color_new = (255/255, 128/255, 14/255)

plt.plot(axis_days_of_week,axis_sales, c=color_new, linewidth=3)

plt.title('Sales Volume per day of the week', fontsize=21)

plt.xlabel('Day of the week', fontsize=18)

plt.ylabel('Sales Volume', fontsize=18)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)
#Let's check the relation between the days of the week with the number of customers

customers_by_week_day = {}

for x in days_of_week:

    selected_row = stores_train_data.loc[stores_train_data["DayOfWeek"] == x, "Customers"].sum().astype(float)    

    customers_by_week_day[x] = selected_row    
#Values (should be organized by day of the week)

customers_by_week_day
#Values are organized by the days of the week, which are the dictionary keys

tuple_days_of_week_customers = sorted(customers_by_week_day.items())
#Much better now

tuple_days_of_week_customers
#The data are separated for plotting

axis_days_of_week_customers, axis_customers = zip(*tuple_days_of_week_sales) 
#Simple plot

plt.figure(figsize=(10,7))



#Color for colorblind people

color_new = (255/255, 128/255, 14/255)

plt.plot(axis_days_of_week_customers, axis_customers, c=color_new, linewidth=3)

plt.title('Number of customers per day of the week', fontsize=21)

plt.xlabel('Day of the week', fontsize=18)

plt.ylabel('Number of customers', fontsize=18)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)
#Let's check the ratio of the number of sales to stores that do promotion and stores that do not

stores_promo_or_not = {}

for x in range(0,2):

    selected_row = stores_train_data.loc[stores_train_data["Promo"] == x, "Sales"].sum().astype(float)    

    stores_promo_or_not[x] = selected_row  

    

stores_promo_or_not    
tuple_promo_or_not = sorted(stores_promo_or_not.items())

tuple_promo_or_not
#The data are separated for plotting

axis_promo_or_not, axis_sales_promo_or_not = zip(*tuple_promo_or_not) 
#Simple plot

plt.figure(figsize=(10,7))



#Color for colorblind people

color_new = (255/255, 128/255, 14/255)



plt.bar(axis_promo_or_not, axis_sales_promo_or_not)

plt.xticks(axis_promo_or_not, ['No promo', 'With promo'])

plt.title('Sales volume and promo', fontsize=21)

plt.ylabel('Sales volume', fontsize=18)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)
#Unique values for transformation - StoreType

print(sorted(pd.Series(stores_train_data["StoreType"]).unique()))
#Unique values for transformation - Assortment

print(sorted(pd.Series(stores_train_data["Assortment"]).unique()))
#StoreType - Before - Store Train Data

stores_train_data["StoreType"][:5]
#StoreType - Before - Store Test Data

stores_test_data["StoreType"][:5]
#Assortment - Before - Store Train Data

stores_train_data["Assortment"][:5]
#Assortment - Before - Store Test Data

stores_test_data["Assortment"][:5]
#Using LabelEncoder

label_encoder = LabelEncoder()



#Store Train Data

stores_train_data["StoreType"] = label_encoder.fit_transform(stores_train_data["StoreType"])

stores_train_data["Assortment"] = label_encoder.fit_transform(stores_train_data["Assortment"])



#Store Test Data

stores_test_data["StoreType"] = label_encoder.fit_transform(stores_test_data["StoreType"])

stores_test_data["Assortment"] = label_encoder.fit_transform(stores_test_data["Assortment"])
#StoreType - After - Store Train Data

stores_train_data["StoreType"][:5]
#Assortment - After - Store Test Data

stores_test_data["StoreType"][:5]
#Assortment - After - Store Train Data

stores_train_data["Assortment"][:5]
#Assortment - After - Store Test Data

stores_test_data["Assortment"][:5]
#Unique values for transformation - Train Data - StateHoliday

print(pd.Series(stores_train_data["StateHoliday"]).unique())
#Unique values for transformation - Test Data - StateHoliday

print(pd.Series(stores_test_data["StateHoliday"]).unique())
#Unique values for transformation - Train Data - SchoolHoliday

print(pd.Series(stores_train_data["SchoolHoliday"]).unique())
#Unique values for transformation - Test Data - SchoolHoliday

print(pd.Series(stores_test_data["SchoolHoliday"]).unique())
#Replacing for StateHoliday - Train Data

stores_train_data.loc[stores_train_data["StateHoliday"] == '0', "StateHoliday"] = 0

stores_train_data.loc[stores_train_data["StateHoliday"] == 'a', "StateHoliday"] = 1

stores_train_data.loc[stores_train_data["StateHoliday"] == 'b', "StateHoliday"] = 2

stores_train_data.loc[stores_train_data["StateHoliday"] == 'c', "StateHoliday"] = 3



#Replacing for StateHoliday - Test Data

stores_test_data.loc[stores_test_data["StateHoliday"] == '0', "StateHoliday"] = 0

stores_test_data.loc[stores_test_data["StateHoliday"] == 'a', "StateHoliday"] = 1
#Unique values for transformation - Train Data - StateHoliday

print(pd.Series(stores_train_data["StateHoliday"]).unique())
#Unique values for transformation - Test Data - StateHoliday

print(pd.Series(stores_test_data["StateHoliday"]).unique())
#Finnaly, we will verify the correlation between the data

corr = stores_train_data.corr()

corr
#Heatmap

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
#Checking the attributes with strong correlations in relation to Sales

cor_target = abs(corr['Sales'])



relevant = sorted(cor_target[cor_target > 0.01], reverse=True)

relevant
#Showing attributes greater than 0.01

#This value was an arbitrary choice



relevant_text = cor_target[cor_target > 0.01]

relevant_text
#Dropping columns - Train Data

stores_train_data.drop(['Store','Date', 'Sales', 'Customers'], axis=1, inplace=True)
#Checking

stores_train_data[:3]
#Dropping columns from Test Data

stores_test_data.drop(['Store','Date', 'Id'], axis=1, inplace=True)
#Checking

stores_test_data[:3]
#Checking columns

print('Train:', len(stores_train_data.columns))

print('Test:', len(stores_test_data.columns))
#I make one last check if the Train columns match Test columns

pd.unique(stores_train_data.columns == stores_test_data.columns)
#The features

cols_features = ['DayOfWeek','Open','Promo','StateHoliday','SchoolHoliday','StoreType','Assortment','CompetitionDistance',

                 'CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2','Year','Month', 'Day']

#Number of features

len(cols_features)
#Store the position of the columns present in the Train dataset

list_colums_stores_train_data = [stores_train_data.columns.get_loc(c) for c in cols_features if c in stores_train_data]
#Separating the columns for Training

store_train_data_values = stores_train_data.iloc[:,list_colums_stores_train_data].values



#It was defined at the beginning

y_train_data = stores_y_train_data.values
#Store the position of the columns present in the Test dataset

list_colums_stores_test_data = [stores_test_data.columns.get_loc(c) for c in cols_features if c in stores_test_data]
#I store the values of the columns that were located through the index. The index was stored in the

#previous step

store_test_data_values = stores_test_data.iloc[:,list_colums_stores_test_data].values
#Saving the new dataset

save_train_data = pd.DataFrame(store_train_data_values)

save_train_data.columns=cols_features



save_test_data = pd.DataFrame(store_test_data_values)

save_test_data.columns=cols_features



save_y_train_data = pd.DataFrame(y_train_data)

save_y_train_data.columns=['y_train']



save_train_data.to_csv('train_data.csv', index=False)

save_test_data.to_csv('test_data.csv', index=False)

save_y_train_data.to_csv('y_train_data.csv', index=False)
#Loading the new dataset

train_data_new = pd.read_csv('train_data.csv', header=0)

test_data_new = pd.read_csv('test_data.csv', header=0)

y_train_data_new = pd.read_csv('y_train_data.csv', index_col=False).values



#From Array of arrays to only array

y_train_data_new = y_train_data_new.reshape(-1)
#Module Sklearn to separate into Test and Training

from sklearn.model_selection import train_test_split



#Data Split for Training, Testing, and Validation

n = random.randrange(45,90)

X_train, X_test, y_train, y_test = train_test_split(train_data_new.values, y_train_data_new, test_size=0.2, random_state=n)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.25, random_state=n)
print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)

print(X_val.shape, y_val.shape)
def calc_rmsep(y, y_pred):

    rmsep = np.sqrt(np.mean(np.square(((y - y_pred) / y_pred)), axis=0))  

    return rmsep
#Main module for DecisionTreeRegressor

from sklearn.tree import DecisionTreeRegressor
#Instantiating the DecisioTree object

decisionTree_regressor = DecisionTreeRegressor(random_state = 0)
#Training

decisionTree_regressor.fit(X_train, y_train) 
#Checking Score

decisionTree_regressor.score(X_val, y_val)
#Prediction for X_test

y_pred_decision_tree = decisionTree_regressor.predict(X_val)
y_pred_decision_tree
#Changing rows with zero values

y_pred_decision_tree[y_pred_decision_tree == 0] = 0.0001
#Evaluation

calc_rmsep(y_val, y_pred_decision_tree)
#Prediction for the data test <- stores_test_data

y_pred_decision_tree_teste = decisionTree_regressor.predict(test_data_new.values)
y_pred_decision_tree_teste[:10]
#File to submit 

submission = pd.read_csv('../input/sample_submission.csv')

submission['Sales'] = y_pred_decision_tree_teste

submission.to_csv('submission_next_tree.csv', index=False)
#Main module

from sklearn.ensemble import RandomForestRegressor
#Parameter setting

randomForest_regressor = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,

           max_features='auto',

           min_impurity_decrease=0.0,

           min_samples_leaf=1, min_samples_split=2,

           min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=-1,

           oob_score=False, random_state=42, verbose=0)
#Training

randomForest_regressor.fit(X_train, y_train)
#Checking score

randomForest_regressor.score(X_val, y_val)
#Prediction for X_test

y_pred_random_forest = randomForest_regressor.predict(X_val)
#Changing rows with zero values

y_pred_random_forest[y_pred_random_forest == 0] = 0.0001
#Evaluation

calc_rmsep(y_val, y_pred_random_forest)
#Prediction for the data test <- stores_test_data

y_pred_random_forest_teste = randomForest_regressor.predict(test_data_new.values)
y_pred_random_forest_teste[:10]
#Evaluation between prediction from DecisionTree and RandomForest

calc_rmsep(y_pred_decision_tree, y_pred_random_forest)
#File to submit.

submission = pd.read_csv('../input/sample_submission.csv')

submission['Sales'] = y_pred_random_forest_teste

submission.to_csv('submission_next_forest.csv', index=False)
#Main module

from sklearn.ensemble import GradientBoostingRegressor

import timeit
#Using the huber metric, it is less sensitive to outliers

# The code will be run n_rounds for a better evaluation of the parameters of n_estimators and max_depth

# There will be two tests: the first the n_estimators and the max_depth range together and in the second the n_estimators

#permance in the default (100) and max_depth only varies





#TEST 01 - Varying n_estimators and max_depth

#TEST 02 - Only changing max_depth



#Number of rounds

n_rounds = 12



#N_estimators

n_manual_estimators = 1



#Max_depth

n_manual_max_depth = 1



#Learning rate

n_manual_learning_rate = 0.09



#Type Test

test_types = 3 



rmspe_errors = []

log_final = pd.DataFrame()



#Rounds

for type_test in range(1, test_types):

    n_manual_max_depth = 1 

    for x in range(1, n_rounds+1):

        print('Round: ', x)

        print('Type Test: ', type_test)

        

        if (type_test == 2):

            n_manual_estimators = 100            



        

        gboost_regressor = GradientBoostingRegressor(loss='huber', learning_rate=n_manual_learning_rate, n_estimators=n_manual_estimators,

                                           min_samples_split=10, min_samples_leaf=15,max_depth=n_manual_max_depth,

                                           random_state=5, max_features='sqrt', verbose=0, validation_fraction=0.3)

        #Fitting the data

        print("Fitting the data... - 1/6")

        start = timeit.default_timer()

        gboost_regressor.fit(X_train, y_train)





        #Prediction for X_train

        print('Making predictions for Train Data.. - 2/6')

        y_pred_gboost_train = gboost_regressor.predict(X_train)



        #Prediction for X_val

        print('Making predictions for Validation Data.. - 3/6')

        y_pred_gboost_val = gboost_regressor.predict(X_val)



        #Prediction for Test Data

        print('Making predictions for Test Data.. - 4/6')

        y_pred_gboost_test = gboost_regressor.predict(test_data_new.values)



        #Evaluation

        print('Evaluation.. - 6/6')

        erro_calc_train = calc_rmsep(y_train, y_pred_gboost_train) 

        erro_calc_val = calc_rmsep(y_val, y_pred_gboost_val) 

        rmspe_errors.append([])

        print('Erro Train: ', erro_calc_train)  

        print('Erro Val: ', erro_calc_val)   

        print('N_estimators: ', n_manual_estimators)   

        print('Max_depth: ', n_manual_max_depth)      

        stop = timeit.default_timer()

        time = stop - start





        #Info for Log Final

        print('Generating log.. - 6/6')

        rmspe_errors.append([x, type_test, time, round(erro_calc_train, 4), round(erro_calc_val, 4), n_manual_learning_rate, n_manual_estimators, n_manual_max_depth])



        print('--------------------------------------------------------------')



        #Varying values

        if (type_test == 1):

            n_manual_estimators += 1

            n_manual_max_depth += 1

        else:

            n_manual_max_depth += 1





        #File to submit

        submission = pd.read_csv('../input/sample_submission.csv')

        submission['Sales'] = y_pred_gboost_test

        submission.to_csv('submission_next_Gboost_Test0'+str(type_test)+'_'+str(x)+'.csv', index=False)



#Generating pandas

log_final = pd.DataFrame(rmspe_errors, columns=['Id','Type','Time (s)','Train_Error','Validation_Error','Learning_Rate','N_estimators','Max_Depth'])

log_final = log_final.sort_values('Validation_Error')

log_final = log_final[:(n_rounds)*2]

log_final.to_csv('log_final_gboost.csv', index=False)

print('Final Log generated!')
#I check the final log and which was the ID that obtained the least error in the validation, it will be this file that will be submitted

log_final
#Main Module

from xgboost import XGBRegressor

import timeit
# The code will be run n_rounds for a better evaluation of the parameters of n_estimators and max_depth

# There will be two tests: the first the n_estimators and the max_depth range together and in the second the n_estimators

#permance in the default (100) and max_depth only varies





#TEST 01 - Varying n_estimators and max_depth

#TEST 02 - Only changing max_depth



#Number of rounds

n_rounds = 24



#N_estimators

n_manual_estimators = 1



#Max_depth

n_manual_max_depth = 1



#Learning rate

n_manual_learning_rate = 0.09



#Type Test

test_types = 3 



rmspe_errors = []

log_final = pd.DataFrame()



#Rounds

for type_test in range(1, test_types):

    n_manual_max_depth = 1 

    for x in range(1, n_rounds+1):

        print('Round: ', x)

        print('Type Test: ', type_test)

        

        if (type_test == 2):

            n_manual_estimators = 100            



        

        xgboost_regressor = XGBRegressor(learning_rate=n_manual_learning_rate, 

                                     n_estimators=n_manual_estimators,

                                     max_depth=n_manual_max_depth, min_child_weight=0,

                                     gamma=0.0003, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective='reg:linear', nthread=4,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006, random_state=42)

        

        

        #Fitting the data

        print("Fitting the data... - 1/7")

        start = timeit.default_timer()

        xgboost_regressor.fit(X_train, y_train, eval_set=[(X_val, y_val)])





        #Prediction for X_train

        print('Making predictions for Train Data.. - 2/7')

        y_pred_xgboost_train = xgboost_regressor.predict(X_train)



        #Prediction for X_val

        print('Making predictions for Validation Data.. - 3/7')

        y_pred_xgboost_val = xgboost_regressor.predict(X_val)



        #Prediction for X_test

        print('Making predictions for X_test.. - 4/7')

        y_pred_xgboost_test_set = xgboost_regressor.predict(X_test)



        #Prediction for Test Data

        print('Making predictions for Test Data.. - 5/7')

        y_pred_xgboost_test = xgboost_regressor.predict(test_data_new.values)



        #Evaluation

        print('Evaluation.. - 6/7')

        erro_calc_train = calc_rmsep(y_train, y_pred_xgboost_train) 

        erro_calc_val = calc_rmsep(y_val, y_pred_xgboost_val) 

        erro_calc_test = calc_rmsep(y_test, y_pred_xgboost_test_set) 

        rmspe_errors.append([])

        print('Erro Train: ', erro_calc_train)  

        print('Erro Val: ', erro_calc_val)   

        print('Erro Test: ', erro_calc_test)  

        print('N_estimators: ', n_manual_estimators)   

        print('Max_depth: ', n_manual_max_depth)      

        stop = timeit.default_timer()

        time = stop - start





        #Info for Log Final

        print('GeneratingGerando log.. - 7/7')

        rmspe_errors.append([x, type_test, time, round(erro_calc_train, 4), round(erro_calc_val, 4), round(erro_calc_test, 4), n_manual_learning_rate, n_manual_estimators, n_manual_max_depth])



        print('--------------------------------------------------------------')



        #Varying values

        if (type_test == 1):

            n_manual_estimators += 1

            n_manual_max_depth += 1

        else:

            n_manual_max_depth += 1

            





            



        #Arquivo para submeter

        submission = pd.read_csv('../input/sample_submission.csv')

        submission['Sales'] = y_pred_xgboost_test

        submission.to_csv('submission_next_Xgboost_Test0'+str(type_test)+'_'+str(x)+'.csv', index=False)



#Generating pandas

log_final = pd.DataFrame(rmspe_errors, columns=['Id','Type','Time (s)','Train_Error','Validation_Error', 'Test_Error','Learning_Rate','N_estimators','Max_Depth'])

log_final = log_final.sort_values('Validation_Error')

log_final = log_final[:(n_rounds)*2]

log_final.to_csv('log_final_xgboost.csv', index=False)

print('Final Log generated!')
#I check the final log and which was the ID that obtained the least error in the validation, it will be this file that will be submitted

log_final