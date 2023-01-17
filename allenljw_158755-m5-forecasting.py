# import required libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import datetime

from math import sqrt

import scipy.stats  as stats

import seaborn as sns



#regression library

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn import svm, preprocessing

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error

from sklearn.svm import SVR

import lightgbm as lgb

import gc
# define the function to change the category column from string to int16

def chg_cat_int(df, cat_col):

    

    for col, col_dtype in cat_col.items():

        if col_dtype == 'category':

            df[col] = df[col].cat.codes.astype('int16')

            df[col] -= df[col].min()

            

    return df
# loaded the dataset of the sales training

sales_catcol = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']

sales_numcol = [f"d_{day}" for day in range(1,1914)]

sales_cat_dtype = {col:'category' for col in sales_catcol if col != 'id'}



# load the data

df_sales = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv", dtype=sales_cat_dtype)



df_sales.head()
# check if null values in the dataframe

df_sales.columns[df_sales.isnull().any()]
# loaded the data of the calendar

calendar_catcol = ['weekday', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI']

calendar_dtype = {col:'category' for col in calendar_catcol}



# load the data

df_calendar = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv", dtype=calendar_dtype)

#df_calendar = chg_cat_int(df_calendar, calendar_dtype)

        

df_calendar.head()
# check if null values in the dataframe

df_calendar.columns[df_calendar.isnull().any()]
# loaded the dataset with information of price for each sku and each week

price_catcol = ['store_id', 'item_id']

price_dtype = {col:'category' for col in price_catcol}



df_price = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv", dtype=price_dtype)



df_price.head()
# check if null values in the dataframe

df_price.columns[df_price.isnull().any()]
# merge all the dataset as a final one for analysis

sales_catcol = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']

sales_numcol = [f"d_{day}" for day in range(1,1914)]

sales_cat_dtype = {col:'category' for col in sales_catcol if col != 'id'}



final_data = df_sales[sales_catcol+sales_numcol]



final_data = final_data.melt(id_vars=sales_catcol, value_vars=sales_numcol, var_name='d', value_name='sales')



final_data = final_data.merge(df_calendar, on='d')

final_data = final_data.merge(df_price, on=['store_id', 'item_id', 'wm_yr_wk'])



# change category from string to int16

final_data = chg_cat_int(final_data, sales_cat_dtype)



final_data.head()
week_day_sales = final_data.groupby(['wday'])['sales'].sum()

weeklabel = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]



month_sales = final_data.groupby(['month'])['sales'].sum()

monthlabel = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']



plt.figure(figsize=(16,4))

plt.subplot(1, 2, 1)

plt.bar(week_day_sales.index, week_day_sales.values, width=0.5)

plt.xticks(week_day_sales.index, weeklabel, rotation=30)

plt.title('Sales distribution in week days')



plt.subplot(1, 2, 2)

plt.bar(month_sales.index, month_sales.values, width=0.5)

plt.xticks(month_sales.index, monthlabel, rotation=30)

plt.title('Sales distribution in months')
f, axes = plt.subplots(1, 2, figsize=(20, 7))

sns.countplot(x='event_type_2', hue='event_name_2', data=final_data, ax=axes[0])



sns.countplot(y='event_name_1', data=final_data, ax=axes[1])
# delete the dataframe for visualization to save the memory

del final_data

del week_day_sales

del month_sales

gc.collect()
# change the null value as 0 and also update the string category to int

df_calendar = chg_cat_int(df_calendar, calendar_dtype)

df_calendar.columns[df_calendar.isnull().any()]
# define the function to shift the column values with lags parameter

def shift_days(df, lags=[28, 35, 42, 49]):

    

    lag_cols = [f'lag_{lag}' for lag in lags]



    for lag, lag_col in zip(lags, lag_cols):

        df[lag_col] = df[['id', 'sales']].groupby('id')['sales'].shift(lag)



    df.dropna(inplace=True)

    

    return df
# function to create the 3 dataset with the source data. It can proudced the data by giving the batch size and the start day



def create_dataset(df_sales, df_calendar, df_price, id_list, batch_size=1000, batch_start=0, start_day=500):

       

    total_size = len(id_list)

    # only select data with the selected batch sku item id 

    if batch_start+batch_size < total_size:

        batch_id_list = id_list[batch_start:batch_start + batch_size]

    else:

        batch_id_list = id_list[batch_start:]

    

    #----------------------------------------------

    # produce training data set

    sales_catcol = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']

    sales_numcol = [f"d_{day}" for day in range(start_day,1914)]

    sales_cat_dtype = {col:'category' for col in sales_catcol if col != 'id'}

    

    batch_train_data = df_sales[sales_catcol+sales_numcol][df_sales['id'].isin(batch_id_list)]

    batch_train_data = batch_train_data.melt(id_vars=sales_catcol, value_vars=sales_numcol, var_name='d', value_name='sales')

    

    batch_train_data = batch_train_data.merge(df_calendar, on='d')

    batch_train_data = batch_train_data.merge(df_price, on=['store_id', 'item_id', 'wm_yr_wk'])

    

    # change category from string to int16

    batch_train_data = chg_cat_int(batch_train_data, sales_cat_dtype)

    

    # give lag parameter. Put the day sales of 1 month ago as the supportive features, and predict today's sales number. 

    # the reason plus 1 month （28 days) is that when preparing the validation dataset, you need to predict the next 28 days, 

    lags = [28, 35, 42, 49]   

    batch_train_data = shift_days(batch_train_data, lags)

    #----------------------------------------------

    # end of produce training data set

    

    

    #----------------------------------------------

    # produce validate data set

    lag_numcol = [f"d_{day}" for day in range(1914-49,1914)]

    predict_numcol = [f"d_{day}" for day in range(1914,1914+28)]

  

    batch_predict_data = df_sales[df_sales['id'].isin(batch_id_list)]

    batch_predict_data = batch_predict_data[sales_catcol+lag_numcol]



    for num_col in predict_numcol:

        batch_predict_data[num_col] = 0     #prefill with 0 values

    

    batch_predict_data = batch_predict_data.melt(id_vars=sales_catcol, value_vars=lag_numcol+predict_numcol, var_name='d', value_name='sales')

    batch_predict_data = batch_predict_data.merge(df_calendar, on='d')

    batch_predict_data = batch_predict_data.merge(df_price, on=['store_id', 'item_id', 'wm_yr_wk'])

    

    # change category from string to int16

    batch_predict_data = chg_cat_int(batch_predict_data, sales_cat_dtype)

    

    # give lag parameter. Put the day sales 1 month ago as the training data, and predict today's sales number

    lags = [28, 35, 42, 49]

    batch_predict_data = shift_days(batch_predict_data, lags)

    # end of produce validate data set

    #----------------------------------------------

    

    #----------------------------------------------

    # produce evaluation data set

    lag_numcol = [f"d_{day}" for day in range(1914-49,1914)]

    evaluate_numcol = [f"d_{day}" for day in range(1914+28,1914+56)]

  

    batch_evaluate_data = df_sales[df_sales['id'].isin(batch_id_list)]

    batch_evaluate_data = batch_evaluate_data[sales_catcol+lag_numcol]



    for num_col in evaluate_numcol:

        batch_evaluate_data[num_col] = 0

    

    batch_evaluate_data = batch_evaluate_data.melt(id_vars=sales_catcol, value_vars=lag_numcol+evaluate_numcol, var_name='d', value_name='sales')

    batch_evaluate_data = batch_evaluate_data.merge(df_calendar, on='d')

    batch_evaluate_data = batch_evaluate_data.merge(df_price, on=['store_id', 'item_id', 'wm_yr_wk'])

    batch_evaluate_data = chg_cat_int(batch_evaluate_data, sales_cat_dtype)

    

    # give lag parameter. Put the day sales 1 month ago as the training data, and predict today's sales number

    lags = [28, 35, 42, 49]

    batch_evaluate_data = shift_days(batch_evaluate_data, lags)

    

    batch_evaluate_data["id"] = batch_evaluate_data["id"].str.replace("validation", "evaluation")

    #----------------------------------------------

    # end of produce evaluation data set

    

    

    return batch_train_data, batch_predict_data, batch_evaluate_data
# Define the category features and the numeric features



#cat_features = ['item_id', 'dept_id', 'store_id', 'cat_id', 'state_id', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI', 'weekday', #'wday', 'month', 'year']

#cat_features = ['item_id', 'store_id', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI', 'wday', 'month', 'year']

cat_features = ['item_id', 'store_id', 'event_name_1', 'event_name_2', 'snap_CA', 'snap_TX', 'snap_WI', 'wday', 'month', 'year']

num_features = ['sell_price', 'lag_28', 'lag_35', 'lag_42', 'lag_49']
id_list = df_sales['id'].unique()

id_list = id_list[0:50]  #limit the total id list for KNN due to computational issue



batch_size = 60             # The size for the number of item id list for each batch

batch_start = 0             # The starting place for the item id list batch



total_size = len(id_list)   # Total length of the item id list



# The paramter for the data start day, range from 0 to 1913. It means select the data from first_day to the 1913th day for training. 

# Note: small start day would cost large memory and long time

start_day = 1000     
%%time

# create the training dataset and validation dataset

train_data, validate_data, evaluate_data = create_dataset(df_sales, df_calendar, df_price, id_list, batch_size, batch_start, start_day)    



X = train_data[cat_features + num_features]

y = train_data['sales']

#X_validate = validate_data[cat_features + num_features]

#X_evaluate = evaluate_data[cat_features + num_features]



# feed the training data (X, y) and train the model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



# create dummy value for the category features via OneHotEncoder function

cat_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),

    ('onehot', preprocessing.OneHotEncoder(handle_unknown='ignore'))])



preprocessor = ColumnTransformer(

    transformers=[('cat', cat_transformer, cat_features)])



clf = Pipeline(steps=[('preprocessor', preprocessor),

                  ('KNN', KNeighborsRegressor())])



k_range = [5, 50, 100, 200]

weight_options = ['uniform', 'distance']

metric_options = ['euclidean', 'minkowski', 'mahalanobis']



param_grid = {'KNN__n_neighbors':k_range, 'KNN__weights':weight_options, 'KNN__metric':metric_options}

grid = GridSearchCV(clf, param_grid, n_jobs=-1)



grid.fit(X_train, y_train)



print("Best Parameter: ", grid.best_params_)

print("KNN Model with best parameter training RMSE: %.4f" %(sqrt(mean_squared_error(y_train, grid.best_estimator_.predict(X_train)))))

print("KNN Model with best parameter testing RMSE: %.4f" %(sqrt(mean_squared_error(y_test, grid.best_estimator_.predict(X_test)))))

# define the function to create the model. The category features would be preprocessed before the training step. 

def create_model(name, model, X, y):

    # split the training and testing dataset with testing size is 20%

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    

    # create dummy value for the category features via OneHotEncoder function

    cat_transformer = Pipeline(steps=[

        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),

        ('onehot', preprocessing.OneHotEncoder(handle_unknown='ignore'))])



    preprocessor = ColumnTransformer(

        transformers=[('cat', cat_transformer, cat_features)])

    

    clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('classifier', model)])

    

    clf.fit(X_train, y_train)

    

    tr_RMSE = {'model':name, 'RMSE':sqrt(mean_squared_error(y_train, clf.predict(X_train)))}

    te_RMSE = {'model':name, 'RMSE':sqrt(mean_squared_error(y_test, clf.predict(X_test)))}

    

    print("model: %s, training RMSE: %.4f" % (name, tr_RMSE['RMSE']))

    print("model: %s, test RMSE: %.4f" % (name, te_RMSE['RMSE']))

    print("")

    

    return clf, tr_RMSE, te_RMSE
%%time

# dataframe to store the training and test RMSE for each model

train_RMSE = pd.DataFrame()

test_RMSE = pd.DataFrame()



# Fit estimators

ESTIMATORS = {

    "Random Forest": RandomForestRegressor(n_estimators=100, max_features=32, random_state=0),

    "Linear regression": LinearRegression(),

    "Logistics regression": LogisticRegression(solver='saga'),

    "SVM": SVR(),}



for name, estimator in ESTIMATORS.items():

    

    clf, tr_RMSE, te_RMSE = create_model(name, estimator, X, y)

    train_RMSE = train_RMSE.append(tr_RMSE, ignore_index=True)

    test_RMSE = test_RMSE.append(te_RMSE, ignore_index=True)

    

test_RMSE
id_list = df_sales['id'].unique()

id_list = id_list[0:1000]  #give larger number of id list for lightGBM model



batch_size = 1100             # > 1000 to let the batch to run once only

batch_start = 0             # The starting place for the item id list batch



start_day = 600    # give larger start date as lightGBM support large dataset
# use the lightGBM model, the category features is not required to performe the OneHotEnconder as the lightGBM have the parameter to indicate the category features. 

def create_GBMmodel(X, y):

    # split the training and testing dataset with testing size is 20%

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)

    validate_data = lgb.Dataset(X_test, label=y_test, categorical_feature=cat_features)

    

    params = {

        "objective" : "poisson",

        "metric" :"rmse",

        "force_row_wise" : True,

        'verbosity': 1,

    }

    

    del X_train, y_train; gc.collect()

    

    num_round = 1500

    m_lgb = lgb.train(params, train_data, num_round, valid_sets = [validate_data], early_stopping_rounds=5, verbose_eval=200) 

        

    return m_lgb
%%time

train_data, validate_data, evaluate_data = create_dataset(df_sales, df_calendar, df_price, id_list, batch_size, batch_start, start_day)    

    

X = train_data[cat_features + num_features]

y = train_data['sales']



# feed the training data (X, y) and train the model

bst = create_GBMmodel(X, y)  
id_list = df_sales['id'].unique()



batch_size = 20000             # The size for the number of item id list for each batch

batch_start = 0             # The starting place for the item id list batch



total_size = len(id_list)   # Total length of the item id list



# The paramter for the data start day, range from 0 to 1912. It means select the data from first_day to the 1913th day for training. 

# Note: small start day would cost large memory and long time

start_day = 500     
batch_num = 1   # batch number indicator

validate_df = pd.DataFrame() # dataframe used to calculate the uncertainty

evaluate_df = pd.DataFrame() # dataframe used to calculate the uncertainty



for val in range(batch_start, total_size, batch_size):

    

    # print the batch start time

    #print('batch %d started from ' % batch_num, datetime.datetime.now())

    

    # create the training dataset and validation dataset

    train_data, validate_data, evaluate_data = create_dataset(df_sales, df_calendar, df_price, id_list, batch_size, val, start_day)    

    

    X = train_data[cat_features + num_features]

    y = train_data['sales']

    X_validate = validate_data[cat_features + num_features]

    X_evaluate = evaluate_data[cat_features + num_features]

    

    # feed the training data (X, y) and train the model

    bst = create_GBMmodel(X, y)

    #bst.save_model('model.txt', num_iteration=bst.best_iteration)

            

    # use the trained model to predict the target submission

    y_validate = bst.predict(X_validate, num_iteration=bst.best_iteration)



    validate_data['sales'] = y_validate

    

    result = validate_data[['id', 'd', 'sales']]

    result = result.pivot(index='id', columns='d', values='sales')

    result = result.reset_index()

     

    out_col = [f'F{d}' for d in range(1, 29)]

    result.rename(columns=dict(zip(result.columns[1:], out_col)), inplace=True)

    

    validate_df = validate_df.append(result, ignore_index=True)

   

    

    # use the trained model to predict the evaluate data

    y_evaluate = bst.predict(X_evaluate, num_iteration=bst.best_iteration)

    

    evaluate_data['sales'] = y_evaluate

    

    result2 = evaluate_data[['id', 'd', 'sales']]

    result2 = result2.pivot(index='id', columns='d', values='sales')

    result2 = result2.reset_index()

    

    result2.rename(columns=dict(zip(result2.columns[1:], out_col)), inplace=True)

    

    evaluate_df = evaluate_df.append(result2, ignore_index=True)

    

    result = pd.concat([result, result2], axis=0, sort=False)

    

    if val == 0:

        result.to_csv('accuracy-submission.csv', index=False, mode='w')

    else:

        result.to_csv('accuracy-submission.csv', index=False, header=False, mode='a')

        

    batch_num += 1

validate_df = validate_df.merge(df_sales[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]], on = "id")

validate_df['Total'] = 'Total'

validate_df.head()
evaluate_df["id"] = evaluate_df["id"].str.replace("evaluation", "validation")

evaluate_df = evaluate_df.merge(df_sales[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]], on = "id")

evaluate_df['Total'] = 'Total'

evaluate_df.head()
qs = np.array([0.005,0.025,0.165,0.25, 0.5, 0.75, 0.835, 0.975, 0.995])



# the ratios calculation is learned from the kaggle notebook "From point to uncertainty prediction"

qs2 = np.log(qs/(1-qs))*.065



ratios = stats.norm.cdf(qs2)

ratios /= ratios[4]

ratios = pd.Series(ratios, index=qs)

ratios
def quantile_coefs(q):

    return ratios.loc[q].values
# define the function to come out the uncertainty distribution with one level grouping

def get_group_preds(pred, level, cols):

    df = pred.groupby(level)[cols].sum()

    q = np.repeat(qs, len(df))

    df = pd.concat([df]*9, axis=0, sort=False)

    df.reset_index(inplace = True)

    df[cols] *= quantile_coefs(q).reshape(-1, 1)

    if level != "id":

        df["id"] = [f"{lev}_X_{q:.3f}_validation" for lev, q in zip(df[level].values, q)]

    else:

        df["id"] = [f"{lev.replace('_validation', '')}_{q:.3f}_validation" for lev, q in zip(df[level].values, q)]

    df = df[["id"]+list(cols)]

    return df
# define the function to come out the uncertainty distribution with two level grouping

def get_couple_group_preds(pred, level1, level2, cols):

    df = pred.groupby([level1, level2])[cols].sum()

    q = np.repeat(qs, len(df))

    df = pd.concat([df]*9, axis=0, sort=False)

    df.reset_index(inplace = True)

    df[cols] *= quantile_coefs(q).reshape(-1, 1)

    df["id"] = [f"{lev1}_{lev2}_{q:.3f}_validation" for lev1,lev2, q in 

                zip(df[level1].values,df[level2].values, q)]

    df = df[["id"]+list(cols)]

    return df
# the levels parameter prepared for grouping calculation

levels = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "Total"]

couples = [("state_id", "item_id"),  ("state_id", "dept_id"),("store_id","dept_id"),

                            ("state_id", "cat_id"),("store_id","cat_id")]

cols = [f"F{i}" for i in range(1, 29)]
# produce the output file for uncertainty

uncertainty_df = pd.DataFrame(columns=['id']+list(cols))

uncertainty_df2 = pd.DataFrame(columns=['id']+list(cols))



for level in levels:

    uncertainty_df = uncertainty_df.append(get_group_preds(validate_df, level, cols))

    uncertainty_df2 = uncertainty_df2.append(get_group_preds(evaluate_df, level, cols))



for level1,level2 in couples:

    uncertainty_df = uncertainty_df.append(get_couple_group_preds(validate_df, level1, level2, cols))

    uncertainty_df2 = uncertainty_df2.append(get_couple_group_preds(evaluate_df, level1, level2, cols))

    

uncertainty_df2['id'] = uncertainty_df2["id"].str.replace("validation", "evaluation")



output = pd.concat([uncertainty_df, uncertainty_df2], axis=0, ignore_index=True)

output.to_csv("uncertainty_submission.csv", index = False)

output.head()
plt.figure(figsize=(15,4))

plt.subplot(1, 2, 1)

accuracy_score = [4.21835, 2.61307, 0.77669, 0.73193, 0.70767, 0.68186,0.68060]

plt.plot(accuracy_score)

plt.xlabel('number of times')

plt.ylabel('Weighted Root Mean Squared Scaled Error')

plt.title('Accuracy score trend')



plt.subplot(1, 2, 2)

uncertainty_score = [0.24640, 0.22897, 0.21921, 0.21045, 0.20218]

plt.plot(uncertainty_score)

plt.xlabel('number of times')

plt.ylabel('Weighted Scaled Pinball Loss')

plt.title('Uncertainty score trend')
import matplotlib.image as mpimg

plt.figure(figsize = (20,2))

img=mpimg.imread('../input/score-image/accuracy-score.png')

imgplot = plt.imshow(img)

plt.show()



plt.figure(figsize = (20,2))

img=mpimg.imread('../input/score-image/uncertainty-score.png')

imgplot = plt.imshow(img)

plt.show()